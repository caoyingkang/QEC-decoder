use crate::bp_base::BPBase;
use crate::serial_bp_kernel::run_serial_bp_iteration;
use numpy::ndarray::{Array1, Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Decode a single syndrome vector using serial-schedule min-sum BP.
///
/// Convergence is checked once per iteration (iteration = one full pass over all
/// VNs in `vn_order`).
fn decode_single(
    base: &BPBase,
    vn_order: &[usize],
    max_iter: usize,
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    synd: ArrayView1<u8>,
) -> Array1<u8> {
    base.init_messages(chk_inmsg);
    // Estimated error vector
    let mut ehat = vec![0_u8; base.num_vars];
    // Posterior LLR values
    let mut llr = base.prior_llr.to_vec();

    // Main BP iteration loop.
    for _ in 0..max_iter {
        let converged = run_serial_bp_iteration(
            base, vn_order, chk_inmsg, var_inmsg, &mut llr, &mut ehat, synd,
        );
        if converged {
            break;
        }
    }

    Array1::from_vec(ehat)
}

/// Decode a single syndrome vector with detailed diagnostics.
fn decode_single_detailed(
    base: &BPBase,
    vn_order: &[usize],
    max_iter: usize,
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    synd: ArrayView1<u8>,
    record_llr_history: bool,
) -> (Array1<u8>, bool, usize, Option<Array2<f64>>) {
    base.init_messages(chk_inmsg);
    // Estimated error vector.
    let mut ehat = vec![0_u8; base.num_vars];
    // Posterior LLR values.
    let mut llr = base.prior_llr.to_vec();
    // History of posterior LLR values, stored as a flattened vector.
    let mut llr_hist_flattened = Vec::<f64>::new();

    // Main BP iteration loop.
    let mut num_iter = 0;
    let mut converged = false;
    while num_iter < max_iter {
        num_iter += 1;
        converged = run_serial_bp_iteration(
            base, vn_order, chk_inmsg, var_inmsg, &mut llr, &mut ehat, synd,
        );
        if record_llr_history {
            llr_hist_flattened.extend_from_slice(&llr);
        }
        if converged {
            break;
        }
    }

    // Convert the flattened LLR history vector into a 2D array.
    let llr_hist = if record_llr_history {
        Some(Array2::from_shape_vec((num_iter, base.num_vars), llr_hist_flattened).unwrap())
    } else {
        None
    };

    (Array1::from_vec(ehat), converged, num_iter, llr_hist)
}

/// Belief Propagation decoder with serial message passing schedule and min-sum CN update rule.
#[pyclass]
pub struct SerialBPDecoderRust {
    /// Base struct for BP-based decoders, which stores parity-check matrix and prior error probabilities.
    base: BPBase,
    /// Permutation of variable nodes.
    vn_order: Vec<usize>,
    /// Maximum number of iterations (one iteration = one full pass over `vn_order`).
    max_iter: usize,
    /// `chk_inmsg[i]` stores the incoming messages at CN `i` from its neighboring VNs.
    chk_inmsg: Vec<Vec<f64>>,
    /// `var_inmsg[j]` stores the incoming messages at VN `j` from its neighboring CNs.
    var_inmsg: Vec<Vec<f64>>,
}

#[pymethods]
impl SerialBPDecoderRust {
    /// Create a serial-schedule BP decoder.
    ///
    /// Parameters:
    /// - `pcm`: Parity-check matrix. Every row (check) must have at least 2 nonzero entries.
    /// Every column (variable) must have at least 1 nonzero entry.
    /// - `prior`: Prior error probabilities (dtype=np.float64).
    /// - `vn_order`: Permutation of variable nodes.
    /// - `max_iter`: Maximum number of iterations (one iteration = one full pass over `vn_order`).
    #[new]
    #[pyo3(signature = (pcm, prior, *, vn_order, max_iter))]
    pub fn new(
        pcm: PyReadonlyArray2<'_, u8>,
        prior: PyReadonlyArray1<'_, f64>,
        vn_order: PyReadonlyArray1<'_, i64>,
        max_iter: usize,
    ) -> Self {
        let pcm = pcm.as_array();
        let prior = prior.as_array();
        let base = BPBase::new(pcm, prior);

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
            vn_order: vn_order.as_array().iter().map(|&x| x as usize).collect(),
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
            &self.vn_order,
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
    /// - `record_llr_history`: Whether to return the history of posterior LLR values
    /// (one snapshot per iteration, taken at the end of each iteration).
    ///
    /// Returns:
    /// - `ehat`: The decoded error vector.
    /// - `converged`: Whether the decoder converged (i.e. the syndrome was satisfied).
    /// - `num_iter`: The number of iterations actually run.
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
            &self.vn_order,
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
            let vn_order = &self.vn_order;
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
                            vn_order,
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
                    &self.vn_order,
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
    /// - `decoding_iters`: Number of iterations actually run in each shot.
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
            let vn_order = &self.vn_order;
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
                            vn_order,
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
                    &self.vn_order,
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
    m.add_class::<SerialBPDecoderRust>()?;
    Ok(())
}
