use crate::bp_base::BPBase;
use numpy::ndarray::{Array1, Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Decode a single syndrome vector using disordered-memory min-sum BP.
fn decode_single(
    base: &BPBase,
    gamma: &Array1<f64>,
    norm: f64,
    max_iter: usize,
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    synd: ArrayView1<u8>,
) -> Array1<u8> {
    base.init_messages(chk_inmsg);
    // Estimated error vector at the current iteration.
    let mut ehat = Array1::<u8>::zeros(base.num_vars);
    // Posterior LLR values at the current iteration.
    let mut llr = base.prior_llr.to_vec();

    // Main BP iteration loop.
    for _ in 0..max_iter {
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
            llr[j] = (1.0 - gamma[j]) * base.prior_llr[j]
                + gamma[j] * llr[j]
                + inmsg.iter().sum::<f64>();
            // Hard decision.
            ehat[j] = if llr[j] < 0.0 { 1 } else { 0 };
            // Calculate the outgoing messages.
            for (k, &i) in base.var_nbrs[j].iter().enumerate() {
                chk_inmsg[i][base.var_nbr_pos[j][k]] = llr[j] - inmsg[k];
            }
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
            break;
        }
    }
    ehat
}

/// Decode a single syndrome vector with detailed diagnostics.
fn decode_single_detailed(
    base: &BPBase,
    gamma: &Array1<f64>,
    norm: f64,
    max_iter: usize,
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    synd: ArrayView1<u8>,
    record_llr_history: bool,
) -> (Array1<u8>, bool, usize, Option<Array2<f64>>) {
    base.init_messages(chk_inmsg);
    // Estimated error vector at the current iteration.
    let mut ehat = Array1::<u8>::zeros(base.num_vars);
    // Posterior LLR values at the current iteration.
    let mut llr = base.prior_llr.to_vec();
    // History of posterior LLR values, stored as a flattened vector.
    let mut llr_hist_flattened = Vec::<f64>::new();

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
            llr[j] = (1.0 - gamma[j]) * base.prior_llr[j]
                + gamma[j] * llr[j]
                + inmsg.iter().sum::<f64>();
            // Hard decision.
            ehat[j] = if llr[j] < 0.0 { 1 } else { 0 };
            // Calculate the outgoing messages.
            for (k, &i) in base.var_nbrs[j].iter().enumerate() {
                chk_inmsg[i][base.var_nbr_pos[j][k]] = llr[j] - inmsg[k];
            }
        }

        // Record LLR values.
        if record_llr_history {
            llr_hist_flattened.extend_from_slice(&llr);
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
    let llr_hist = if record_llr_history {
        Some(Array2::from_shape_vec((num_iter, base.num_vars), llr_hist_flattened).unwrap())
    } else {
        None
    };

    (ehat, converged, num_iter, llr_hist)
}

/// Disordered-memory min-sum BP decoder.
#[pyclass]
pub struct DMemBPDecoderRust {
    /// Base struct for BP-based decoders, which stores parity-check matrix and prior error probabilities.
    base: BPBase,
    /// Memory strength for each variable node.
    gamma: Array1<f64>,
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
impl DMemBPDecoderRust {
    /// Create a DMemBP decoder.
    ///
    /// Parameters:
    /// - `pcm`: Parity-check matrix. Every row (check) must have at least 2 nonzero entries.
    /// Every column (variable) must have at least 1 nonzero entry.
    /// - `prior`: Prior error probabilities.
    /// - `gamma`: Memory strength for each variable node. The value 0.0 means no memory.
    /// - `norm`: Normalization factor. Default is 1.0, meaning no normalization.
    /// - `max_iter`: Maximum number of BP iterations.
    #[new]
    #[pyo3(signature = (pcm, prior, *, gamma, norm=None, max_iter))]
    pub fn new(
        pcm: PyReadonlyArray2<'_, u8>,
        prior: PyReadonlyArray1<'_, f64>,
        gamma: PyReadonlyArray1<'_, f64>,
        norm: Option<f64>,
        max_iter: usize,
    ) -> Self {
        let pcm = pcm.as_array();
        let prior = prior.as_array();
        let gamma = gamma.as_array();
        let base = BPBase::new(pcm, prior);
        let norm = norm.unwrap_or(1.0);

        let mut var_inmsg = Vec::new();
        for j in 0..base.num_vars {
            var_inmsg.push(vec![0.0; base.var_nbrs[j].len()]);
        }

        let mut chk_inmsg = Vec::new();
        for i in 0..base.num_chks {
            chk_inmsg.push(vec![0.0; base.chk_nbrs[i].len()]);
        }

        Self {
            base: base,
            gamma: gamma.to_owned(),
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
    ///
    /// Return: The decoded error vector.
    pub fn decode<'py>(
        &mut self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
    ) -> Bound<'py, PyArray1<u8>> {
        let syndrome = syndrome.as_array();
        let ehat = decode_single(
            &self.base,
            &self.gamma,
            self.norm,
            self.max_iter,
            &mut self.chk_inmsg,
            &mut self.var_inmsg,
            syndrome,
        );
        PyArray1::from_owned_array(py, ehat)
    }

    /// Decode a syndrome vector with detailed diagnostics.
    ///
    /// Parameters:
    /// - `syndrome`: Syndrome vector.
    /// - `record_llr_history`: Whether to return the history of posterior LLR values.
    ///
    /// Returns:
    /// - `ehat`: The decoded error vector.
    /// - `converged`: Whether the decoder converged (i.e. the syndrome was satisfied).
    /// - `num_iter`: The number of BP iterations actually run.
    /// - `llr_hist`: The history of posterior LLR values if `record_llr_history` is True;
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
        let syndrome = syndrome.as_array();
        let (ehat, converged, num_iter, llr_hist) = decode_single_detailed(
            &self.base,
            &self.gamma,
            self.norm,
            self.max_iter,
            &mut self.chk_inmsg,
            &mut self.var_inmsg,
            syndrome,
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
    /// - `parallel`: Whether to use multithreaded decoding. Default is false.
    ///
    /// Return: Batch of decoded error vectors.
    #[pyo3(signature = (syndrome_batch, *, parallel=false))]
    pub fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        syndrome_batch: PyReadonlyArray2<'_, u8>,
        parallel: bool,
    ) -> Bound<'py, PyArray2<u8>> {
        let syndrome_batch = syndrome_batch.as_array();
        let batch_size: usize = syndrome_batch.nrows();
        let mut ehat_batch = Array2::<u8>::zeros((batch_size, self.base.num_vars));

        if parallel {
            let syndrome_batch = syndrome_batch.to_owned();
            let base = &self.base;
            let gamma = &self.gamma;
            let norm = self.norm;
            let max_iter = self.max_iter;
            let chk_inmsg_template = &self.chk_inmsg;
            let var_inmsg_template = &self.var_inmsg;

            let results: Vec<Array1<u8>> = py.allow_threads(|| {
                (0..batch_size)
                    .into_par_iter()
                    .map(|i| {
                        let mut chk_inmsg = chk_inmsg_template.clone();
                        let mut var_inmsg = var_inmsg_template.clone();
                        decode_single(
                            base,
                            gamma,
                            norm,
                            max_iter,
                            &mut chk_inmsg,
                            &mut var_inmsg,
                            syndrome_batch.row(i),
                        )
                    })
                    .collect()
            });

            for (i, ehat) in results.into_iter().enumerate() {
                ehat_batch.row_mut(i).assign(&ehat);
            }
        } else {
            for i in 0..batch_size {
                let ehat = decode_single(
                    &self.base,
                    &self.gamma,
                    self.norm,
                    self.max_iter,
                    &mut self.chk_inmsg,
                    &mut self.var_inmsg,
                    syndrome_batch.row(i),
                );
                ehat_batch.row_mut(i).assign(&ehat);
            }
        }

        PyArray2::from_owned_array(py, ehat_batch)
    }

    /// Decode a batch of syndrome vectors with detailed diagnostics.
    ///
    /// Parameters:
    /// - `syndrome_batch`: Batch of syndrome vectors.
    /// - `parallel`: Whether to use multithreaded decoding. Default is false.
    ///
    /// Returns:
    /// - `ehat_batch`: Batch of decoded error vectors.
    /// - `converged_mask`: Whether the decoder converged in each shot.
    /// - `decoding_iters`: Number of BP iterations actually run in each shot.
    #[pyo3(signature = (syndrome_batch, *, parallel=false))]
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
        let batch_size: usize = syndrome_batch.nrows();
        let mut ehat_batch = Array2::<u8>::zeros((batch_size, self.base.num_vars));
        let mut converged_mask = Vec::<bool>::with_capacity(batch_size);
        let mut decoding_iters = Vec::<i64>::with_capacity(batch_size);

        if parallel {
            let syndrome_batch = syndrome_batch.to_owned();
            let base = &self.base;
            let gamma = &self.gamma;
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
                        let (ehat, converged, num_iter, _) = decode_single_detailed(
                            base,
                            gamma,
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
                converged_mask.push(converged);
                decoding_iters.push(num_iter as i64);
            }
        } else {
            for i in 0..batch_size {
                let (ehat, converged, num_iter, _) = decode_single_detailed(
                    &self.base,
                    &self.gamma,
                    self.norm,
                    self.max_iter,
                    &mut self.chk_inmsg,
                    &mut self.var_inmsg,
                    syndrome_batch.row(i),
                    false,
                );
                ehat_batch.row_mut(i).assign(&ehat);
                converged_mask.push(converged);
                decoding_iters.push(num_iter as i64);
            }
        }

        Ok((
            PyArray2::from_owned_array(py, ehat_batch),
            PyArray1::from_vec(py, converged_mask),
            PyArray1::from_vec(py, decoding_iters),
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DMemBPDecoderRust>()?;
    Ok(())
}
