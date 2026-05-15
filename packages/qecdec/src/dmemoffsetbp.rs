use crate::bp_base::{init_v2c_msg, BPBase, BPBuffer};
use crate::utils::{is_all_zeros, sign_parities, two_smallest_abs};
use numpy::ndarray::{Array1, Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Run DMemOffsetBP decoding algorithm. Return `(ehat, converged, num_iter)`.
///
/// Update `buffer` in place. The initial contents of `buffer` are overwritten.
fn run_dmemoffsetbp(
    base: &BPBase,
    gamma: &Array1<f64>,
    offset: &[Vec<f64>],
    norm: &[Vec<f64>],
    max_iter: usize,
    buffer: &mut BPBuffer,
    synd: ArrayView1<u8>,
) -> (Array1<u8>, bool, usize) {
    // Return immediately if syndrome is all zeros.
    if is_all_zeros(synd) {
        return (Array1::zeros(base.num_vars), true, 0);
    }

    init_v2c_msg(base, buffer);
    let mut ehat = Array1::zeros(base.num_vars);
    let mut llr = base.prior_llr.to_vec();

    // Main BP iteration loop.
    let mut num_iter = 0;
    let mut converged = false;
    while num_iter < max_iter {
        num_iter += 1;

        // Message processing at CNs.
        for i in 0..base.num_chks {
            let inmsg = &buffer.chk_inmsg[i];
            let (inmsg_sgnpar, total_sgnpar) = sign_parities(inmsg);
            let (minabs1, minabs2, minidx) = two_smallest_abs(inmsg);
            for (k, &j) in base.chk_nbrs[i].iter().enumerate() {
                let msg_sgnpar = synd[i] ^ total_sgnpar ^ inmsg_sgnpar[k];
                let msg_abs = if k == minidx { minabs2 } else { minabs1 };
                let msg_abs_offset = (msg_abs - offset[i][k]).max(0.0);
                let msg = if msg_sgnpar == 0 {
                    msg_abs_offset
                } else {
                    -msg_abs_offset
                };
                buffer.var_inmsg[j][base.chk_nbr_pos[i][k]] = norm[i][k] * msg;
            }
        }

        // Message processing at VNs.
        for j in 0..base.num_vars {
            // List of incoming messages.
            let inmsg = &buffer.var_inmsg[j];
            // Get posterior LLR.
            llr[j] = (1.0 - gamma[j]) * base.prior_llr[j]
                + gamma[j] * llr[j]
                + inmsg.iter().sum::<f64>();
            // Hard decision.
            ehat[j] = if llr[j] < 0.0 { 1 } else { 0 };
            // Calculate the outgoing messages.
            for (k, &i) in base.var_nbrs[j].iter().enumerate() {
                buffer.chk_inmsg[i][base.var_nbr_pos[j][k]] = llr[j] - inmsg[k];
            }
        }

        // Check if the syndrome is satisfied. If so, early stop.
        if base.syndrome_satisfied(ehat.as_slice().unwrap(), synd) {
            converged = true;
            break;
        }
    }

    (ehat, converged, num_iter)
}

/// Disordered-memory, offset-normalized min-sum BP decoder.
#[pyclass]
pub struct DMemOffsetBPDecoderRust {
    /// Base struct for BP-based decoders, which stores parity-check matrix and prior error probabilities.
    base: BPBase,
    /// Per-VN memory strength.
    gamma: Array1<f64>,
    /// Offset parameter for each CN-to-VN edge.
    offset: Vec<Vec<f64>>,
    /// Normalization factor for each CN-to-VN edge.
    norm: Vec<Vec<f64>>,
    /// Maximum number of iterations.
    max_iter: usize,
    /// Message buffer.
    buffer: BPBuffer,
}

#[pymethods]
impl DMemOffsetBPDecoderRust {
    /// Create a DMemOffsetBP decoder.
    ///
    /// Parameters:
    /// - `pcm`: Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
    /// - `prior`: Prior error probabilities.
    /// - `gamma`: Per-VN memory strength.
    /// - `offset`: Offset parameter for each CN-to-VN edge.
    /// - `norm`: Normalization factor for each CN-to-VN edge.
    /// - `max_iter`: Maximum number of BP iterations.
    #[new]
    #[pyo3(signature = (pcm, prior, *, gamma, offset, norm, max_iter))]
    pub fn new(
        pcm: PyReadonlyArray2<'_, u8>,
        prior: PyReadonlyArray1<'_, f64>,
        gamma: PyReadonlyArray1<'_, f64>,
        offset: Vec<Vec<f64>>,
        norm: Vec<Vec<f64>>,
        max_iter: usize,
    ) -> PyResult<Self> {
        let base = BPBase::new(pcm.as_array(), prior.as_array())?;
        let buffer = BPBuffer::new(&base);

        Ok(Self {
            base,
            gamma: gamma.as_array().to_owned(),
            offset,
            norm,
            max_iter,
            buffer,
        })
    }

    /// Decode a syndrome vector.
    ///
    /// Parameters:
    /// - `syndrome`: Syndrome vector.
    ///
    /// Returns:
    /// - `ehat`: Estimated error vector.
    /// - `converged`: Whether the decoder converged (i.e. the syndrome was satisfied).
    /// - `num_iter`: The number of BP iterations actually run.
    pub fn decode_detailed<'py>(
        &mut self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, bool, usize)> {
        let (ehat, converged, num_iter) = run_dmemoffsetbp(
            &self.base,
            &self.gamma,
            &self.offset,
            &self.norm,
            self.max_iter,
            &mut self.buffer,
            syndrome.as_array(),
        );
        Ok((PyArray1::from_owned_array(py, ehat), converged, num_iter))
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

        let results: Vec<(Array1<u8>, bool, usize)> = if parallel {
            py.allow_threads(|| {
                (0..batch_size)
                    .into_par_iter()
                    .map(|i| {
                        let mut buffer = self.buffer.clone();
                        run_dmemoffsetbp(
                            &self.base,
                            &self.gamma,
                            &self.offset,
                            &self.norm,
                            self.max_iter,
                            &mut buffer,
                            syndrome_batch.row(i),
                        )
                    })
                    .collect()
            })
        } else {
            py.allow_threads(|| {
                (0..batch_size)
                    .map(|i| {
                        run_dmemoffsetbp(
                            &self.base,
                            &self.gamma,
                            &self.offset,
                            &self.norm,
                            self.max_iter,
                            &mut self.buffer,
                            syndrome_batch.row(i),
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
    m.add_class::<DMemOffsetBPDecoderRust>()?;
    Ok(())
}
