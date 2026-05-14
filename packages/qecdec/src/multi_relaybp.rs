use crate::bp_base::{alloc_msg_buffers, BPBase};
use crate::dmembp_core::run_dmembp_in_relay;
use crate::relaybp_core::run_random_relays;
use crate::utils::{is_all_zeros, spawn_seeds};
use numpy::ndarray::{Array1, Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::distr::Uniform;
use rayon::prelude::*;

/// Run MultiRelayBP decoding algorithm. Return `(ehat, converged, num_iter)`.
///
/// `chk_inmsg` and `var_inmsg` serve as scratch space for stage 0 and are mutated in
/// place; afterward they are also used as templates that each chain clones.
fn run_multi_relaybp(
    base: &BPBase,
    gamma0: ArrayView1<f64>,
    gamma_dist: &Uniform<f64>,
    num_chains: usize,
    num_relays: usize,
    pre_iter: usize,
    max_iter_per_relay: usize,
    stop_nconv: usize,
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    synd: ArrayView1<u8>,
    seed: Option<u64>,
) -> (Array1<u8>, bool, usize) {
    // Return immediately if syndrome is all zeros.
    if is_all_zeros(synd) {
        return (Array1::zeros(base.num_vars), true, 0);
    }

    // ===== Shared stage 0: user-provided gamma0 =====
    let mut llr0 = base.prior_llr.to_vec();
    let mut ehat0 = vec![0; base.num_vars];
    let (conv0, it0) = run_dmembp_in_relay(
        base, gamma0, 1.0, pre_iter, chk_inmsg, var_inmsg, &mut llr0, &mut ehat0, synd,
    );

    // Shortcut: stage 0 alone meets the stopping criterion.
    if conv0 && stop_nconv == 1 {
        return (Array1::from_vec(ehat0), true, it0);
    }

    let initial_candidates = if conv0 {
        vec![ehat0.clone()]
    } else {
        Vec::new()
    };

    // ===== Generate per-chain seeds from `seed` =====
    let chain_seeds = spawn_seeds(seed, num_chains);

    // ===== Fork into chains (parallel) =====
    // Reborrow the post-stage-0 buffers as shared slices to use as clone templates.
    let chk_template: &[Vec<f64>] = chk_inmsg;
    let var_template: &[Vec<f64>] = var_inmsg;
    let llr_template: &[f64] = &llr0;
    let ehat_template: &[u8] = &ehat0;
    let candidates_template = &initial_candidates;

    let mut chain_results: Vec<(Array1<u8>, bool, usize)> = chain_seeds
        .into_par_iter()
        .map(|chain_seed| {
            let mut chk_inmsg = chk_template.to_vec();
            let mut var_inmsg = var_template.to_vec();
            let mut llr = llr_template.to_vec();
            let mut ehat = ehat_template.to_vec();
            run_random_relays(
                base,
                gamma_dist,
                num_relays,
                max_iter_per_relay,
                stop_nconv,
                &mut chk_inmsg,
                &mut var_inmsg,
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
    /// `chk_inmsg[i]` stores the incoming messages at CN `i` from its neighboring VNs during the current BP iteration.
    chk_inmsg: Vec<Vec<f64>>,
    /// `var_inmsg[j]` stores the incoming messages at VN `j` from its neighboring CNs during the current BP iteration.
    var_inmsg: Vec<Vec<f64>>,
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
        let pcm = pcm.as_array();
        let prior = prior.as_array();
        let base = BPBase::new(pcm, prior)?;
        let (gamma_low, gamma_high) = gamma_dist_interval;
        let gamma_dist = Uniform::new_inclusive(gamma_low, gamma_high).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid gamma_dist_interval: {}", e))
        })?;
        let (chk_inmsg, var_inmsg) = alloc_msg_buffers(&base);

        Ok(Self {
            base,
            gamma0: gamma0.as_array().to_owned(),
            gamma_dist,
            num_chains,
            num_relays,
            pre_iter,
            max_iter_per_relay,
            stop_nconv,
            chk_inmsg,
            var_inmsg,
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
        &mut self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
        seed: Option<u64>,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, bool, usize)> {
        let synd = syndrome.as_array();
        let (ehat, converged, num_iter) = py.allow_threads(|| {
            run_multi_relaybp(
                &self.base,
                self.gamma0.view(),
                &self.gamma_dist,
                self.num_chains,
                self.num_relays,
                self.pre_iter,
                self.max_iter_per_relay,
                self.stop_nconv,
                &mut self.chk_inmsg,
                &mut self.var_inmsg,
                synd,
                seed,
            )
        });
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
        &mut self,
        py: Python<'py>,
        syndrome_batch: PyReadonlyArray2<'_, u8>,
        parallel: bool,
        seed: Option<u64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<i64>>,
    )> {
        let syndrome_batch = syndrome_batch.as_array();
        let batch_size = syndrome_batch.nrows();
        let mut ehat_batch = Array2::zeros((batch_size, self.base.num_vars));
        let mut converged_mask = Array1::default(batch_size);
        let mut decoding_iters = Array1::zeros(batch_size);

        let child_seeds = spawn_seeds(seed, batch_size);

        let results: Vec<(Array1<u8>, bool, usize)> = if parallel {
            py.allow_threads(|| {
                child_seeds
                    .into_par_iter()
                    .enumerate()
                    .map(|(i, child_seed)| {
                        let mut chk_inmsg = self.chk_inmsg.clone();
                        let mut var_inmsg = self.var_inmsg.clone();
                        run_multi_relaybp(
                            &self.base,
                            self.gamma0.view(),
                            &self.gamma_dist,
                            self.num_chains,
                            self.num_relays,
                            self.pre_iter,
                            self.max_iter_per_relay,
                            self.stop_nconv,
                            &mut chk_inmsg,
                            &mut var_inmsg,
                            syndrome_batch.row(i),
                            child_seed,
                        )
                    })
                    .collect()
            })
        } else {
            // Wrap the serial batch loop in allow_threads so the inner chain-level
            // par_iter in run_multi_relaybp can use other CPU cores without the GIL.
            py.allow_threads(|| {
                child_seeds
                    .iter()
                    .enumerate()
                    .map(|(i, &child_seed)| {
                        run_multi_relaybp(
                            &self.base,
                            self.gamma0.view(),
                            &self.gamma_dist,
                            self.num_chains,
                            self.num_relays,
                            self.pre_iter,
                            self.max_iter_per_relay,
                            self.stop_nconv,
                            &mut self.chk_inmsg,
                            &mut self.var_inmsg,
                            syndrome_batch.row(i),
                            child_seed,
                        )
                    })
                    .collect()
            })
        };

        for (i, (ehat, converged, num_iter)) in results.into_iter().enumerate() {
            ehat_batch.row_mut(i).assign(&ehat);
            converged_mask[i] = converged;
            decoding_iters[i] = num_iter as i64;
        }

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
