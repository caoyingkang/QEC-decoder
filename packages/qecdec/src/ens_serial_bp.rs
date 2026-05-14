//! Ensembled serial-schedule min-sum BP decoder.
//!
//! Runs `ensemble_size` serial-schedule BP decoders with different `vn_order`
//! permutations in lockstep (one global iteration at a time, parallel across
//! members via Rayon). Once `topk` members have converged, the remaining
//! still-active members are stopped, and the most-likely candidate among the
//! converged members (lowest prior-LLR weight) is returned.

use crate::bp_base::{alloc_msg_buffers, init_v2c_msg, BPBase};
use crate::serial_bp_core::run_serial_bp_one_iteration;
use crate::utils::{is_all_zeros, pick_most_likely};
use numpy::ndarray::{Array1, Array2, ArrayView1, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::mem;

struct MemberState {
    /// `chk_inmsg[i]` stores the incoming messages at CN `i` from its neighboring VNs.
    chk_inmsg: Vec<Vec<f64>>,
    /// `var_inmsg[j]` stores the incoming messages at VN `j` from its neighboring CNs.
    var_inmsg: Vec<Vec<f64>>,
    /// Posterior LLR values.
    llr: Vec<f64>,
    /// Estimated error vector.
    ehat: Vec<u8>,
    /// Set to `Some(num_iter)` as soon as this member converges.
    num_iter_on_conv: Option<usize>,
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
    /// All-zeros template (with correct sizes); to be cloned per member.
    chk_inmsg_template: Vec<Vec<f64>>,
    /// All-zeros template (with correct sizes); to be cloned per member.
    var_inmsg_template: Vec<Vec<f64>>,
}

impl EnsSerialBPDecoderRust {
    /// Run the ensemble on a single syndrome vector.
    /// Return `(ehat, converged, num_iter)`.
    fn _run(&self, synd: ArrayView1<u8>) -> (Array1<u8>, bool, usize) {
        // Return immediately if syndrome is all zeros.
        if is_all_zeros(synd) {
            return (Array1::zeros(self.base.num_vars), true, 0);
        }

        // Build per-member states for this syndrome.
        let mut members: Vec<MemberState> = (0..self.ensemble_size)
            .map(|_| {
                let mut chk_inmsg = self.chk_inmsg_template.clone();
                init_v2c_msg(&self.base, &mut chk_inmsg);
                MemberState {
                    chk_inmsg,
                    var_inmsg: self.var_inmsg_template.clone(),
                    llr: vec![0.0; self.base.num_vars],
                    ehat: vec![0_u8; self.base.num_vars],
                    num_iter_on_conv: None,
                }
            })
            .collect();

        // Main BP iteration loop.
        let mut num_iter = 0;
        let mut at_least_one_converged = false;
        while num_iter < self.max_iter {
            num_iter += 1;

            members.par_iter_mut().enumerate().for_each(|(i, m)| {
                if m.num_iter_on_conv.is_some() {
                    return;
                }
                let conv = run_serial_bp_one_iteration(
                    &self.base,
                    &self.vn_orders[i],
                    &mut m.chk_inmsg,
                    &mut m.var_inmsg,
                    &mut m.llr,
                    &mut m.ehat,
                    synd,
                );
                if conv {
                    m.num_iter_on_conv = Some(num_iter);
                }
            });

            let count = members
                .iter()
                .filter(|m| m.num_iter_on_conv.is_some())
                .count();
            if count > 0 {
                at_least_one_converged = true;
            }
            if count >= self.topk {
                break;
            }
        }

        if !at_least_one_converged {
            return (
                Array1::from_vec(mem::take(&mut members[0].ehat)),
                false,
                num_iter,
            );
        }

        // Pick the most-likely candidate among all converged members.
        let candidates: Vec<Vec<u8>> = members
            .into_iter()
            .filter(|m| m.num_iter_on_conv.is_some())
            .map(|m| m.ehat)
            .collect();
        let best_ehat = pick_most_likely(candidates, self.base.prior_llr.view());

        (Array1::from_vec(best_ehat), true, num_iter)
    }
}

#[pymethods]
impl EnsSerialBPDecoderRust {
    /// Create an ensembled serial-schedule BP decoder.
    ///
    /// Parameters:
    /// - `pcm`: Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
    /// - `prior`: Prior error probabilities.
    /// - `vn_orders`: Ensemble of variable node permutations, shape `(ensemble_size, num_vars)`.
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
    ) -> PyResult<Self> {
        let pcm = pcm.as_array();
        let prior = prior.as_array();
        let base = BPBase::new(pcm, prior)?;
        let vn_orders = vn_orders.as_array();
        let ensemble_size = vn_orders.nrows();
        if vn_orders.ncols() != base.num_vars {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vn_orders must have {} columns",
                base.num_vars
            )));
        }
        if topk < 1 || topk > ensemble_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Require 1 <= topk <= ensemble_size",
            ));
        }
        let vn_orders_vecs = vn_orders
            .axis_iter(Axis(0))
            .map(|row| row.iter().map(|&x| x as usize).collect())
            .collect();
        let (chk_inmsg_template, var_inmsg_template) = alloc_msg_buffers(&base);

        Ok(Self {
            base,
            ensemble_size,
            topk,
            vn_orders: vn_orders_vecs,
            max_iter,
            chk_inmsg_template,
            var_inmsg_template,
        })
    }

    /// Decode a syndrome vector.
    ///
    /// Parameters:
    /// - `syndrome`: Syndrome vector.
    ///
    /// Returns:
    /// - `ehat`: Estimated error vector.
    /// - `converged`: Whether at least one ensemble member converged.
    /// - `num_iter`: The number of iterations actually run.
    pub fn decode_detailed<'py>(
        &self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
    ) -> (Bound<'py, PyArray1<u8>>, bool, usize) {
        let syndrome = syndrome.as_array();
        let (ehat, converged, num_iter) = py.allow_threads(|| self._run(syndrome));
        (PyArray1::from_owned_array(py, ehat), converged, num_iter)
    }

    /// Decode a batch of syndrome vectors.
    ///
    /// Parameters:
    /// - `syndrome_batch`: Batch of syndrome vectors.
    ///
    /// Returns:
    /// - `ehat_batch`: Batch of estimated error vectors.
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

        let (ehat_batch, converged_mask, decoding_iters) = py.allow_threads(|| {
            let mut ehat_batch = Array2::zeros((batch_size, self.base.num_vars));
            let mut converged_mask = Array1::default(batch_size);
            let mut decoding_iters = Array1::zeros(batch_size);
            for i in 0..batch_size {
                let (ehat, converged, num_iter) = self._run(syndrome_batch.row(i));
                ehat_batch.row_mut(i).assign(&ehat);
                converged_mask[i] = converged;
                decoding_iters[i] = num_iter as i64;
            }
            (ehat_batch, converged_mask, decoding_iters)
        });

        (
            PyArray2::from_owned_array(py, ehat_batch),
            PyArray1::from_owned_array(py, converged_mask),
            PyArray1::from_owned_array(py, decoding_iters),
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EnsSerialBPDecoderRust>()?;
    Ok(())
}
