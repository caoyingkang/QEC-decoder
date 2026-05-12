use crate::bp_base::{alloc_msg_buffers, init_v2c_msg, BPBase};
use crate::utils::is_all_zeros;
use numpy::ndarray::{Array1, Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Run BP decoding algorithm. Return `(ehat, converged, num_iter, None)` (if
/// `record_llr_history` is true) or `(ehat, converged, num_iter, llr_hist)`
/// (if `record_llr_history` is false).
///
/// `chk_inmsg` and `var_inmsg` only need to be sized correctly; their
/// initial values will be overwritten.
fn run_bp(
    base: &BPBase,
    norm: f64,
    max_iter: usize,
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    synd: ArrayView1<u8>,
    record_llr_history: bool,
) -> (Array1<u8>, bool, usize, Option<Array2<f64>>) {
    // Return immediately if syndrome is all zeros.
    if is_all_zeros(synd) {
        return (
            Array1::zeros(base.num_vars),
            true,
            0,
            if record_llr_history {
                Some(Array2::zeros((0, base.num_vars)))
            } else {
                None
            },
        );
    }

    init_v2c_msg(base, chk_inmsg);
    // Estimated error vector at current iteration.
    let mut ehat = Array1::zeros(base.num_vars);
    // Posterior LLR values at current iteration.
    let mut llr = vec![0.0; base.num_vars];
    // History of posterior LLR values, stored as a flattened vector.
    let mut llr_hist_flattened = if record_llr_history {
        Some(Vec::new())
    } else {
        None
    };

    // Main BP iteration loop.
    let mut num_iter = 0;
    let mut converged = false;
    while num_iter < max_iter {
        num_iter += 1;

        // Message processing at CNs.
        for i in 0..base.num_chks {
            // List of incoming messages.
            let inmsg = &chk_inmsg[i];
            // List of sign parities of the incoming messages (0 for positive, 1 for negative).
            let inmsg_sgnpar: Vec<u8> =
                inmsg.iter().map(|&x| if x < 0.0 { 1 } else { 0 }).collect();
            // Total sign parity of the incoming messages (i.e. XOR of the entries in inmsg_sgnpar).
            let total_sgnpar = inmsg_sgnpar.iter().fold(0, |acc, &x| acc ^ x);
            // Minimum absolute value of the incoming messages.
            let mut minabs1 = f64::MAX;
            // Second minimum absolute value of the incoming messages.
            let mut minabs2 = f64::MAX;
            // Index of the incoming message with minimum absolute value.
            let mut minidx = 0;
            for (k, &val) in inmsg.iter().enumerate() {
                let val_abs = val.abs();
                if val_abs < minabs1 {
                    minabs2 = minabs1;
                    minabs1 = val_abs;
                    minidx = k;
                } else if val_abs < minabs2 {
                    minabs2 = val_abs;
                }
            }
            // Calculate the outgoing messages.
            for (k, &j) in base.chk_nbrs[i].iter().enumerate() {
                let msg_sgnpar = synd[i] ^ total_sgnpar ^ inmsg_sgnpar[k];
                let msg_abs = if k == minidx { minabs2 } else { minabs1 };
                let msg = if msg_sgnpar == 0 { msg_abs } else { -msg_abs };
                var_inmsg[j][base.chk_nbr_pos[i][k]] = norm * msg;
            }
        }

        // Message processing at VNs.
        for j in 0..base.num_vars {
            // List of incoming messages.
            let inmsg = &var_inmsg[j];
            // Get posterior LLR.
            llr[j] = base.prior_llr[j] + inmsg.iter().sum::<f64>();
            // Hard decision.
            ehat[j] = if llr[j] < 0.0 { 1 } else { 0 };
            // Calculate the outgoing messages.
            for (k, &i) in base.var_nbrs[j].iter().enumerate() {
                chk_inmsg[i][base.var_nbr_pos[j][k]] = llr[j] - inmsg[k];
            }
        }

        // Record LLR values.
        if let Some(ref mut v) = llr_hist_flattened {
            v.extend_from_slice(&llr);
        }

        // Check if the syndrome is satisfied. If so, early stop.
        let mut satisfied = true;
        for i in 0..base.num_chks {
            let mut parity = 0_u8;
            for &j in base.chk_nbrs[i].iter() {
                parity ^= ehat[j];
            }
            if parity != synd[i] {
                satisfied = false;
                break;
            }
        }
        if satisfied {
            converged = true;
            break;
        }
    }

    // Convert the flattened LLR history vector into a 2D array.
    let llr_hist = if let Some(v) = llr_hist_flattened {
        Some(Array2::from_shape_vec((num_iter, base.num_vars), v).unwrap())
    } else {
        None
    };

    (ehat, converged, num_iter, llr_hist)
}

/// Belief Propagation decoder (min-sum variant).
#[pyclass]
pub struct BPDecoderRust {
    /// Base struct for BP-based decoders, which stores parity-check matrix and prior error probabilities.
    base: BPBase,
    /// Normalization factor. For no normalization, set to 1.0.
    norm: f64,
    /// Maximum number of iterations.
    max_iter: usize,
    /// `chk_inmsg[i]` stores the incoming messages at CN `i` from its neighboring VNs during the current BP iteration.
    chk_inmsg: Vec<Vec<f64>>,
    /// `var_inmsg[j]` stores the incoming messages at VN `j` from its neighboring CNs during the current BP iteration.
    var_inmsg: Vec<Vec<f64>>,
}

#[pymethods]
impl BPDecoderRust {
    /// Create a BP decoder.
    ///
    /// Parameters:
    /// - `pcm`: Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
    /// - `prior`: Prior error probabilities.
    /// - `norm`: Message normalization factor. Default is 1.0, meaning no normalization.
    /// - `max_iter`: Maximum number of BP iterations.
    #[new]
    #[pyo3(signature = (pcm, prior, *, norm=None, max_iter))]
    pub fn new(
        pcm: PyReadonlyArray2<'_, u8>,
        prior: PyReadonlyArray1<'_, f64>,
        norm: Option<f64>,
        max_iter: usize,
    ) -> Self {
        let pcm = pcm.as_array();
        let prior = prior.as_array();
        let base = BPBase::new(pcm, prior);
        let norm = norm.unwrap_or(1.0);
        let (chk_inmsg, var_inmsg) = alloc_msg_buffers(&base);

        Self {
            base: base,
            norm: norm,
            max_iter: max_iter,
            chk_inmsg: chk_inmsg,
            var_inmsg: var_inmsg,
        }
    }

    /// Decode a syndrome vector.
    ///
    /// Parameters:
    /// - `syndrome`: Syndrome vector.
    /// - `record_llr_history`: Whether to return the history of posterior LLR values.
    ///
    /// Returns:
    /// - `ehat`: Estimated error vector.
    /// - `converged`: Whether the decoder converged (i.e. the syndrome was satisfied).
    /// - `num_iter`: The number of BP iterations actually run.
    /// - `llr_hist`: The history of posterior LLR values if `record_llr_history` is true;
    /// otherwise, `None`.
    #[pyo3(signature = (syndrome, *, record_llr_history))]
    pub fn decode_detailed<'py>(
        &mut self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
        record_llr_history: bool,
    ) -> PyResult<(
        Bound<'py, PyArray1<u8>>,
        bool,
        usize,
        Option<Bound<'py, PyArray2<f64>>>,
    )> {
        let (ehat, converged, num_iter, llr_hist) = run_bp(
            &self.base,
            self.norm,
            self.max_iter,
            &mut self.chk_inmsg,
            &mut self.var_inmsg,
            syndrome.as_array(),
            record_llr_history,
        );
        let llr_hist_py = llr_hist.map(|arr| PyArray2::from_owned_array(py, arr));
        Ok((
            PyArray1::from_owned_array(py, ehat),
            converged,
            num_iter,
            llr_hist_py,
        ))
    }

    /// Decode a batch of syndrome vectors.
    ///
    /// Parameters:
    /// - `syndrome_batch`: Batch of syndrome vectors.
    /// - `parallel`: Whether to use multithreaded decoding.
    ///
    /// Returns:
    /// - `ehat_batch`: Batch of estimated error vectors.
    /// - `converged_mask`: Whether the decoder converged in each shot.
    /// - `decoding_iters`: Number of BP iterations actually run in each shot.
    #[pyo3(signature = (syndrome_batch, *, parallel))]
    pub fn decode_batch_detailed<'py>(
        &mut self,
        py: Python<'py>,
        syndrome_batch: PyReadonlyArray2<'_, u8>,
        parallel: bool,
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

        if parallel {
            let syndrome_batch = syndrome_batch.to_owned();
            let base = &self.base;
            let norm = self.norm;
            let max_iter = self.max_iter;
            let chk_inmsg_template = &self.chk_inmsg;
            let var_inmsg_template = &self.var_inmsg;

            let results: Vec<(Array1<u8>, bool, usize)> = py.allow_threads(|| {
                (0..batch_size)
                    .into_par_iter()
                    .map(|i| {
                        let mut chk_inmsg = chk_inmsg_template.clone();
                        let mut var_inmsg = var_inmsg_template.clone();
                        let (ehat, converged, num_iter, _) = run_bp(
                            base,
                            norm,
                            max_iter,
                            &mut chk_inmsg,
                            &mut var_inmsg,
                            syndrome_batch.row(i),
                            false,
                        );
                        (ehat, converged, num_iter)
                    })
                    .collect()
            });

            for (i, (ehat, converged, num_iter)) in results.into_iter().enumerate() {
                ehat_batch.row_mut(i).assign(&ehat);
                converged_mask[i] = converged;
                decoding_iters[i] = num_iter as i64;
            }
        } else {
            for i in 0..batch_size {
                let (ehat, converged, num_iter, _) = run_bp(
                    &self.base,
                    self.norm,
                    self.max_iter,
                    &mut self.chk_inmsg,
                    &mut self.var_inmsg,
                    syndrome_batch.row(i),
                    false,
                );
                ehat_batch.row_mut(i).assign(&ehat);
                converged_mask[i] = converged;
                decoding_iters[i] = num_iter as i64;
            }
        }

        Ok((
            PyArray2::from_owned_array(py, ehat_batch),
            PyArray1::from_owned_array(py, converged_mask),
            PyArray1::from_owned_array(py, decoding_iters),
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BPDecoderRust>()?;
    Ok(())
}
