use crate::bp_base::{BPBase, BPBuffer};
use crate::bp_like::{BPLike, RandBPLike};
use crate::dmembp_core::run_dmembp_in_relay;
use crate::relaybp_core::run_random_relays;
use crate::utils::{is_all_zeros, spawn_seeds};
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::distr::Uniform;
use rayon::prelude::*;

/// MultiRelayBP decoder: `num_chains` chains sharing a deterministic first stage,
/// then forking into independent random-relay sequences.
/// We say that a chain has converged if it has found at least one candidate error pattern.
/// The output of that chain is the min-LLR-weight candidate.
/// We say that the overall MultiRelayBP decoder has converged if at least one chain has
/// converged. In this case, the chain that finished decoding with fewest iterations wins,
/// with ties broken by lowest chain index. If no chain converged, output the estimated
/// error pattern in the final iteration of the first chain.
#[pyclass]
pub struct MultiRelayBPDecoderRust {
    /// Base struct for BP-based decoders, which stores parity-check matrix and prior error probabilities.
    base: BPBase,
    /// Per-VN memory strength for the shared initial DMemBP stage.
    gamma0: Array1<f64>,
    /// Uniform distribution from which to sample gamma vectors at each relay stage.
    gamma_dist: Uniform<f64>,
    /// Number of independent chains.
    num_chains: usize,
    /// Number of DMemBP relays beyond the initial stage.
    num_relays: usize,
    /// Max number of iterations for the shared initial DMemBP stage.
    pre_iter: usize,
    /// Max number of iterations for each relay stage.
    max_iter_per_relay: usize,
    /// Stop a chain after that chain has collected this many converged candidates.
    stop_nconv: usize,
}

impl BPLike for MultiRelayBPDecoderRust {
    fn base(&self) -> &BPBase {
        &self.base
    }
}

impl RandBPLike for MultiRelayBPDecoderRust {
    fn run(
        &self,
        buffer: &mut BPBuffer,
        synd: ArrayView1<u8>,
        seed: Option<u64>,
    ) -> (Array1<u8>, bool, usize) {
        let base = &self.base;

        // Return immediately if syndrome is all zeros.
        if is_all_zeros(synd) {
            return (Array1::zeros(base.num_vars), true, 0);
        }

        // ===== Shared stage 0: user-provided gamma0 =====
        let mut llr0 = base.prior_llr.to_vec();
        let mut ehat0 = vec![0; base.num_vars];
        let (conv0, it0) = run_dmembp_in_relay(
            base,
            self.gamma0.view(),
            1.0,
            self.pre_iter,
            buffer,
            &mut llr0,
            &mut ehat0,
            synd,
        );

        // Shortcut: stage 0 alone meets the stopping criterion.
        if conv0 && self.stop_nconv == 1 {
            return (Array1::from_vec(ehat0), true, it0);
        }

        let initial_candidates = if conv0 {
            vec![ehat0.clone()]
        } else {
            Vec::new()
        };

        // ===== Generate per-chain seeds from `seed` =====
        let chain_seeds = spawn_seeds(seed, self.num_chains);

        // ===== Fork into chains (parallel) =====
        // The post-stage-0 buffer serves as a template each chain clones.
        let buffer_template = &*buffer;
        let llr_template: &[f64] = &llr0;
        let ehat_template: &[u8] = &ehat0;
        let candidates_template = &initial_candidates;

        let mut chain_results: Vec<(Array1<u8>, bool, usize)> = chain_seeds
            .into_par_iter()
            .map(|chain_seed| {
                let mut buffer = buffer_template.clone();
                let mut llr = llr_template.to_vec();
                let mut ehat = ehat_template.to_vec();
                run_random_relays(
                    base,
                    &self.gamma_dist,
                    self.num_relays,
                    self.max_iter_per_relay,
                    self.stop_nconv,
                    &mut buffer,
                    &mut llr,
                    &mut ehat,
                    synd,
                    candidates_template.clone(),
                    it0,
                    chain_seed,
                )
            })
            .collect();

        // ===== Cross-chain selection =====
        // Among converged chains, pick the one with the smallest total iteration count.
        // If no chain converged, fallback to chain 0.
        let idx = chain_results
            .iter()
            .enumerate()
            .filter(|(_, (_, conv, _))| *conv)
            .min_by_key(|(_, (_, _, it))| *it)
            .map(|(i, _)| i)
            .unwrap_or(0);
        chain_results.swap_remove(idx)
    }
}

#[pymethods]
impl MultiRelayBPDecoderRust {
    /// Create a MultiRelayBP decoder.
    ///
    /// Parameters:
    /// - `pcm`: Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
    /// - `prior`: Prior error probabilities, shape `(num_vars,)`.
    /// - `gamma0`: Per-VN memory strength for the shared initial DMemBP stage, shape `(num_vars,)`.
    /// - `gamma_dist_interval`: `(low, high)` uniform distribution for sampling gamma vectors
    /// at each relay stage.
    /// - `num_chains`: Number of independent chains (≥1).
    /// - `num_relays`: Number of DMemBP relays beyond the initial stage.
    /// - `pre_iter`: Max iterations for the shared initial DMemBP stage.
    /// - `max_iter_per_relay`: Max number of iterations for each relay stage.
    /// - `stop_nconv`: Stop a chain after that chain has collected this many converged candidates.
    /// Must satisfy `1 ≤ stop_nconv ≤ num_relays + 1`.
    #[new]
    #[pyo3(signature = (pcm, prior, *, gamma0, gamma_dist_interval, num_chains, num_relays, pre_iter, max_iter_per_relay, stop_nconv))]
    pub fn new(
        pcm: PyReadonlyArray2<'_, u8>,
        prior: PyReadonlyArray1<'_, f64>,
        gamma0: PyReadonlyArray1<'_, f64>,
        gamma_dist_interval: (f64, f64),
        num_chains: usize,
        num_relays: usize,
        pre_iter: usize,
        max_iter_per_relay: usize,
        stop_nconv: usize,
    ) -> PyResult<Self> {
        let base = BPBase::new(pcm.as_array(), prior.as_array())?;
        let (gamma_low, gamma_high) = gamma_dist_interval;
        let gamma_dist = Uniform::new_inclusive(gamma_low, gamma_high).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid gamma_dist_interval: {}", e))
        })?;
        Ok(Self {
            base,
            gamma0: gamma0.as_array().to_owned(),
            gamma_dist,
            num_chains,
            num_relays,
            pre_iter,
            max_iter_per_relay,
            stop_nconv,
        })
    }

    /// Decode a syndrome vector. Chains run in parallel via rayon.
    ///
    /// Parameters:
    /// - `syndrome`: Syndrome vector.
    /// - `seed`: Optional RNG seed for reproducibility. If provided, this will be the
    /// master seed used to generate child seeds for each chain.
    ///
    /// Returns:
    /// - `ehat`: Estimated error vector.
    /// - `converged`: Whether the decoder converged.
    /// - `num_iter`: Total BP iterations of the winning chain.
    #[pyo3(signature = (syndrome, *, seed=None))]
    pub fn decode_detailed<'py>(
        &self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
        seed: Option<u64>,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, bool, usize)> {
        let synd = syndrome.as_array();
        let mut buf = BPBuffer::new(&self.base);
        let (ehat, converged, num_iter) = py.allow_threads(|| self.run(&mut buf, synd, seed));
        Ok((PyArray1::from_owned_array(py, ehat), converged, num_iter))
    }

    /// Decode a batch of syndrome vectors.
    ///
    /// Parameters:
    /// - `syndrome_batch`: Batch of syndrome vectors.
    /// - `parallel`: Whether to parallelize at the **batch** level. Chain-level
    /// parallelism is always on.
    /// - `seed`: Optional RNG seed for reproducibility. Derives per-shot seeds which are
    /// in turn used to derive per-chain seeds inside each shot.
    ///
    /// Returns:
    /// - `ehat_batch`: Batch of estimated error vectors.
    /// - `converged_mask`: Whether the decoder converged in each shot.
    /// - `decoding_iters`: Total BP iterations of the winning chain in each shot.
    #[pyo3(signature = (syndrome_batch, *, parallel, seed=None))]
    pub fn decode_batch_detailed<'py>(
        &self,
        py: Python<'py>,
        syndrome_batch: PyReadonlyArray2<'_, u8>,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<i64>>,
    )> {
        let synd = syndrome_batch.as_array();
        let (ehat_batch, converged_mask, decoding_iters) =
            py.allow_threads(|| self.run_batch(synd, parallel, seed));
        Ok((
            PyArray2::from_owned_array(py, ehat_batch),
            PyArray1::from_owned_array(py, converged_mask),
            PyArray1::from_owned_array(py, decoding_iters),
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MultiRelayBPDecoderRust>()?;
    Ok(())
}
