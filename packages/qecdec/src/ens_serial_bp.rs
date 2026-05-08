//! Ensemble decoder built on top of the serial-schedule BP kernel.
//!
//! Runs `ensemble_size` SerialBP-style decoders with different `vn_order`
//! permutations in lockstep (one global iteration at a time, parallel across
//! members via Rayon). Once `topk` members have converged, the remaining
//! still-active members are stopped at the next iteration boundary, and the
//! most-likely candidate among the converged members (lowest prior-LLR weight)
//! is returned.

use crate::bp_base::BPBase;
use crate::serial_bp_kernel::run_serial_bp_iteration;
use numpy::ndarray::{Array1, Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

struct MemberState {
    chk_inmsg: Vec<Vec<f64>>,
    var_inmsg: Vec<Vec<f64>>,
    llr: Vec<f64>,
    ehat: Vec<u8>,
    /// Set to `Some(iter)` the first time this member's run_iteration returns true.
    converged_at_iter: Option<usize>,
}

/// Ensemble of SerialBP decoders with per-member `vn_order` permutations.
#[pyclass]
pub struct EnsSerialBPDecoderRust {
    base: BPBase,
    /// Length = ensemble_size; each entry is a permutation of [0..num_vars).
    vn_orders: Vec<Vec<usize>>,
    max_iter: usize,
    ensemble_size: usize,
    topk: usize,
    /// All-zeros templates (correct sizes); cloned per syndrome per member.
    chk_inmsg_template: Vec<Vec<f64>>,
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
                base.init_messages(&mut chk_inmsg);
                MemberState {
                    chk_inmsg,
                    var_inmsg: self.var_inmsg_template.clone(),
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

        // Score: sum_i e_i * prior_llr_i. Smaller = more likely under
        // independent bit priors. prior_llr_i = ln((1-p_i)/p_i) > 0 for p_i < 0.5.
        let prior_llr = base.prior_llr.as_slice().unwrap();
        let score = |e: &[u8]| -> f64 {
            e.iter()
                .zip(prior_llr.iter())
                .map(|(&b, &llr)| (b as f64) * llr)
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
    /// Create an ensemble serial-schedule BP decoder.
    ///
    /// Parameters:
    /// - `pcm`: Parity-check matrix (uint8). Every row must have at least 2
    ///   nonzero entries; every column at least 1.
    /// - `prior`: Prior error probabilities (float64).
    /// - `vn_orders`: Stack of variable-node permutations, shape
    ///   `(ensemble_size, num_vars)`, dtype int64.
    /// - `max_iter`: Maximum number of global iterations.
    /// - `topk`: Number of converged members required before terminating
    ///   the remaining (still-unconverged) members. Must satisfy
    ///   `1 <= topk <= ensemble_size`.
    #[new]
    #[pyo3(signature = (pcm, prior, *, vn_orders, max_iter, topk))]
    pub fn new(
        pcm: PyReadonlyArray2<'_, u8>,
        prior: PyReadonlyArray1<'_, f64>,
        vn_orders: PyReadonlyArray2<'_, i64>,
        max_iter: usize,
        topk: usize,
    ) -> PyResult<Self> {
        let pcm = pcm.as_array();
        let prior = prior.as_array();
        let base = BPBase::new(pcm, prior);

        let vn_orders_arr = vn_orders.as_array();
        let ensemble_size = vn_orders_arr.nrows();
        if ensemble_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ensemble_size (vn_orders.shape[0]) must be >= 1",
            ));
        }
        if vn_orders_arr.ncols() != base.num_vars {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vn_orders must have shape (ensemble_size, {}), got ({}, {})",
                base.num_vars,
                ensemble_size,
                vn_orders_arr.ncols(),
            )));
        }
        if topk < 1 || topk > ensemble_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Require 1 <= topk <= ensemble_size",
            ));
        }

        let vn_orders_vec: Vec<Vec<usize>> = (0..ensemble_size)
            .map(|i| {
                vn_orders_arr
                    .row(i)
                    .iter()
                    .map(|&x| x as usize)
                    .collect::<Vec<usize>>()
            })
            .collect();

        let mut var_inmsg_template = Vec::with_capacity(base.num_vars);
        for j in 0..base.num_vars {
            var_inmsg_template.push(vec![0.0; base.var_nbrs[j].len()]);
        }
        let mut chk_inmsg_template = Vec::with_capacity(base.num_chks);
        for i in 0..base.num_chks {
            chk_inmsg_template.push(vec![0.0; base.chk_nbrs[i].len()]);
        }

        Ok(Self {
            base,
            vn_orders: vn_orders_vec,
            max_iter,
            ensemble_size,
            topk,
            chk_inmsg_template,
            var_inmsg_template,
        })
    }

    /// Decode a single syndrome vector. Returns the estimated error vector.
    pub fn decode<'py>(
        &self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
    ) -> Bound<'py, PyArray1<u8>> {
        let synd = syndrome.as_array();
        let (ehat, _, _) = py.allow_threads(|| self.decode_one(synd));
        PyArray1::from_owned_array(py, ehat)
    }

    /// Decode a single syndrome vector with detailed diagnostics.
    ///
    /// Returns: (ehat, converged, num_iter).
    pub fn decode_detailed<'py>(
        &self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
    ) -> (Bound<'py, PyArray1<u8>>, bool, usize) {
        let synd = syndrome.as_array();
        let (ehat, converged, num_iter) = py.allow_threads(|| self.decode_one(synd));
        (PyArray1::from_owned_array(py, ehat), converged, num_iter)
    }

    /// Decode a batch of syndrome vectors. Outer loop is sequential; ensemble
    /// parallelism happens within each syndrome via Rayon.
    pub fn decode_batch<'py>(
        &self,
        py: Python<'py>,
        syndrome_batch: PyReadonlyArray2<'_, u8>,
    ) -> Bound<'py, PyArray2<u8>> {
        let syndrome_batch = syndrome_batch.as_array();
        let batch_size = syndrome_batch.nrows();
        let num_vars = self.base.num_vars;

        let ehat_batch: Array2<u8> = py.allow_threads(|| {
            let mut out = Array2::<u8>::zeros((batch_size, num_vars));
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
    /// Returns: (ehat_batch, converged_mask, decoding_iters).
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
        let num_vars = self.base.num_vars;

        let (ehat_batch, converged_mask, decoding_iters): (Array2<u8>, Vec<bool>, Vec<i64>) = py
            .allow_threads(|| {
                let mut out = Array2::<u8>::zeros((batch_size, num_vars));
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
