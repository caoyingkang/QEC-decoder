//! Ensembled serial-schedule min-sum BP decoder.
//!
//! Runs `ensemble_size` serial-schedule BP decoders with different `vn_order`
//! permutations in lockstep (one global iteration at a time, parallel across
//! members via Rayon). Once `topk` members have converged, the remaining
//! still-active members are stopped, and the most-likely candidate among the
//! converged members (lowest prior-LLR weight) is returned.

use crate::bp_base::{alloc_msg_buffers, init_v2c_msg, BPBase};
use crate::serial_bp_kernel::run_serial_bp_iteration;
use numpy::ndarray::{Array1, Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

struct MemberState {
    /// `chk_inmsg[i]` stores the incoming messages at CN `i` from its neighboring VNs.
    chk_inmsg: Vec<Vec<f64>>,
    /// `var_inmsg[j]` stores the incoming messages at VN `j` from its neighboring CNs.
    var_inmsg: Vec<Vec<f64>>,
    /// Posterior LLR values.
    llr: Vec<f64>,
    /// Estimated error vector.
    ehat: Vec<u8>,
    /// Set to `Some(iter)` as soon as this member converges.
    converged_at_iter: Option<usize>,
}

/// Ensemble of serial-schedule BP decoders with per-member `vn_order` permutations.
#[pyclass]
pub struct EnsSerialBPDecoderRust {
    /// Base struct for BP-based decoders, which stores parity-check matrix and prior error probabilities.
    base: BPBase,
    /// Number of member decoders.
    ensemble_size: usize,
    /// Stops decoding once `topk` members converge.
    topk: usize,
    /// Length = ensemble_size; each entry is a permutation of variable nodes.
    vn_orders: Vec<Vec<usize>>,
    /// Maximum number of iterations (one iteration = one full pass over `vn_order`).
    max_iter: usize,
    /// All-zeros template (with correct sizes); cloned per member.
    chk_inmsg_template: Vec<Vec<f64>>,
    /// All-zeros template (with correct sizes); cloned per member.
    var_inmsg_template: Vec<Vec<f64>>,
}

impl EnsSerialBPDecoderRust {
    /// Run the ensemble on a single syndrome.
    /// Returns (ehat, converged, iters_completed).
    fn decode_one(&self, synd: ArrayView1<u8>) -> (Array1<u8>, bool, usize) {
        let base = &self.base;
        let vn_orders = &self.vn_orders;
        let num_vars = base.num_vars;
        let topk = self.topk;

        // Build per-member states for this syndrome.
        let mut members: Vec<MemberState> = (0..self.ensemble_size)
            .map(|_| {
                let mut chk_inmsg = self.chk_inmsg_template.clone();
                init_v2c_msg(base, &mut chk_inmsg);
                let var_inmsg = self.var_inmsg_template.clone();
                MemberState {
                    chk_inmsg: chk_inmsg,
                    var_inmsg: var_inmsg,
                    llr: base.prior_llr.to_vec(),
                    ehat: vec![0_u8; num_vars],
                    converged_at_iter: None,
                }
            })
            .collect();

        // Iteration-synchronous step lock.
        let mut iters_completed = 0_usize;
        for iter in 0..self.max_iter {
            members.par_iter_mut().enumerate().for_each(|(i, m)| {
                if m.converged_at_iter.is_some() {
                    return;
                }
                let conv = run_serial_bp_iteration(
                    base,
                    &vn_orders[i],
                    &mut m.chk_inmsg,
                    &mut m.var_inmsg,
                    &mut m.llr,
                    &mut m.ehat,
                    synd,
                );
                if conv {
                    m.converged_at_iter = Some(iter);
                }
            });
            iters_completed = iter + 1;

            let count = members
                .iter()
                .filter(|m| m.converged_at_iter.is_some())
                .count();
            if count >= topk {
                break;
            }
        }

        // Score: sum_i e_i * prior_llr_i. Smaller = more likely.
        let prior_llr = base.prior_llr.as_slice().unwrap();
        let score = |e: &[u8]| -> f64 {
            e.iter()
                .zip(prior_llr.iter())
                .map(|(&b, &l)| (b as f64) * l)
                .sum()
        };

        // Pick the most-likely candidate among all converged members
        // (could be > topk if multiple members converged in the same iteration).
        let best_idx = members
            .iter()
            .enumerate()
            .filter(|(_, m)| m.converged_at_iter.is_some())
            .min_by(|(_, a), (_, b)| score(&a.ehat).partial_cmp(&score(&b.ehat)).unwrap())
            .map(|(i, _)| i);

        match best_idx {
            Some(i) => (Array1::from(members[i].ehat.clone()), true, iters_completed),
            None => (
                Array1::from(members[0].ehat.clone()),
                false,
                iters_completed,
            ),
        }
    }
}

#[pymethods]
impl EnsSerialBPDecoderRust {
    /// Create an ensembled serial-schedule BP decoder.
    ///
    /// Parameters:
    /// - `pcm`: Parity-check matrix. Every row (check) must have at least 2 nonzero entries.
    /// Every column (variable) must have at least 1 nonzero entry.
    /// - `prior`: Prior error probabilities (dtype=np.float64).
    /// - `vn_orders`: Ensemble of variable node permutations, shape
    /// `(ensemble_size, num_vars)`, (dtype=np.int64).
    /// - `max_iter`: Maximum number of iterations (one iteration = one full pass over `vn_order`).
    /// - `topk`: Number of converged members required before terminating the remaining members.
    /// Must satisfy `1 <= topk <= ensemble_size`.
    #[new]
    #[pyo3(signature = (pcm, prior, *, vn_orders, max_iter, topk))]
    pub fn new(
        pcm: PyReadonlyArray2<'_, u8>,
        prior: PyReadonlyArray1<'_, f64>,
        vn_orders: PyReadonlyArray2<'_, i64>,
        max_iter: usize,
        topk: usize,
    ) -> Self {
        let pcm = pcm.as_array();
        let prior = prior.as_array();
        let base = BPBase::new(pcm, prior);
        let vn_orders_arr = vn_orders.as_array();
        let ensemble_size = vn_orders_arr.nrows();
        assert!(
            vn_orders_arr.ncols() == base.num_vars,
            "vn_orders must have {} columns",
            base.num_vars
        );
        assert!(
            topk >= 1 && topk <= ensemble_size,
            "Require 1 <= topk <= ensemble_size"
        );

        let vn_orders_vecs: Vec<Vec<usize>> = (0..ensemble_size)
            .map(|i| {
                vn_orders_arr
                    .row(i)
                    .iter()
                    .map(|&x| x as usize)
                    .collect::<Vec<usize>>()
            })
            .collect();

        let (chk_inmsg_template, var_inmsg_template) = alloc_msg_buffers(&base);

        Self {
            base: base,
            ensemble_size: ensemble_size,
            topk: topk,
            vn_orders: vn_orders_vecs,
            max_iter: max_iter,
            chk_inmsg_template: chk_inmsg_template,
            var_inmsg_template: var_inmsg_template,
        }
    }

    /// Decode a syndrome vector.
    ///
    /// Parameters:
    /// - `syndrome`: Syndrome vector.
    ///
    /// Return: The decoded error vector.
    pub fn decode<'py>(
        &self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
    ) -> Bound<'py, PyArray1<u8>> {
        let syndrome = syndrome.as_array();
        let (ehat, _, _) = py.allow_threads(|| self.decode_one(syndrome));
        PyArray1::from_owned_array(py, ehat)
    }

    /// Decode a syndrome vector with detailed diagnostics.
    ///
    /// Parameters:
    /// - `syndrome`: Syndrome vector.
    ///
    /// Returns:
    /// - `ehat`: The decoded error vector.
    /// - `converged`: Whether the decoder converged (i.e. the syndrome was satisfied).
    /// - `num_iter`: The number of iterations actually run.
    pub fn decode_detailed<'py>(
        &self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
    ) -> (Bound<'py, PyArray1<u8>>, bool, usize) {
        let syndrome = syndrome.as_array();
        let (ehat, converged, num_iter) = py.allow_threads(|| self.decode_one(syndrome));
        (PyArray1::from_owned_array(py, ehat), converged, num_iter)
    }

    /// Decode a batch of syndrome vectors.
    ///
    /// Parameters:
    /// - `syndrome_batch`: Batch of syndrome vectors.
    ///
    /// Return: Batch of decoded error vectors.
    pub fn decode_batch<'py>(
        &self,
        py: Python<'py>,
        syndrome_batch: PyReadonlyArray2<'_, u8>,
    ) -> Bound<'py, PyArray2<u8>> {
        let syndrome_batch = syndrome_batch.as_array();
        let batch_size = syndrome_batch.nrows();

        let ehat_batch: Array2<u8> = py.allow_threads(|| {
            let mut out = Array2::<u8>::zeros((batch_size, self.base.num_vars));
            for i in 0..batch_size {
                let (ehat, _, _) = self.decode_one(syndrome_batch.row(i));
                out.row_mut(i).assign(&ehat);
            }
            out
        });

        PyArray2::from_owned_array(py, ehat_batch)
    }

    /// Decode a batch of syndrome vectors with detailed diagnostics.
    ///
    /// Parameters:
    /// - `syndrome_batch`: Batch of syndrome vectors.
    ///
    /// Returns:
    /// - `ehat_batch`: Batch of decoded error vectors.
    /// - `converged_mask`: Whether the decoder converged in each shot.
    /// - `decoding_iters`: Number of iterations actually run in each shot.
    pub fn decode_batch_detailed<'py>(
        &self,
        py: Python<'py>,
        syndrome_batch: PyReadonlyArray2<'_, u8>,
    ) -> (
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<i64>>,
    ) {
        let syndrome_batch = syndrome_batch.as_array();
        let batch_size = syndrome_batch.nrows();

        let (ehat_batch, converged_mask, decoding_iters): (Array2<u8>, Vec<bool>, Vec<i64>) = py
            .allow_threads(|| {
                let mut out = Array2::<u8>::zeros((batch_size, self.base.num_vars));
                let mut conv_mask = Vec::<bool>::with_capacity(batch_size);
                let mut iters = Vec::<i64>::with_capacity(batch_size);
                for i in 0..batch_size {
                    let (ehat, converged, num_iter) = self.decode_one(syndrome_batch.row(i));
                    out.row_mut(i).assign(&ehat);
                    conv_mask.push(converged);
                    iters.push(num_iter as i64);
                }
                (out, conv_mask, iters)
            });

        (
            PyArray2::from_owned_array(py, ehat_batch),
            PyArray1::from_vec(py, converged_mask),
            PyArray1::from_vec(py, decoding_iters),
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EnsSerialBPDecoderRust>()?;
    Ok(())
}
