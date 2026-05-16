use crate::bp_base::{init_v2c_msg, BPBase, BPBuffer};
use crate::bp_like::{BPLike, DetBPLike};
use crate::csr::Csr;
use crate::utils::{is_all_zeros, sign_parities, two_smallest_abs};
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Disordered-memory, offset-normalized min-sum BP decoder.
#[pyclass]
pub struct DMemOffsetBPDecoderRust {
    /// Base struct for BP-based decoders, which stores parity-check matrix and prior error probabilities.
    base: BPBase,
    /// Per-VN memory strength.
    gamma: Array1<f64>,
    /// Offset parameter for each CN-to-VN edge.
    offset: Csr<f64>,
    /// Normalization factor for each CN-to-VN edge.
    norm: Csr<f64>,
    /// Maximum number of iterations.
    max_iter: usize,
}

impl BPLike for DMemOffsetBPDecoderRust {
    fn base(&self) -> &BPBase {
        &self.base
    }
}

impl DetBPLike for DMemOffsetBPDecoderRust {
    fn run(&self, buffer: &mut BPBuffer, synd: ArrayView1<u8>) -> (Array1<u8>, bool, usize) {
        let base = &self.base;

        // Return immediately if syndrome is all zeros.
        if is_all_zeros(synd) {
            return (Array1::zeros(base.num_vars), true, 0);
        }

        init_v2c_msg(base, buffer);
        let mut ehat = Array1::zeros(base.num_vars);
        let mut llr = base.prior_llr.to_vec();

        let mut num_iter = 0;
        let mut converged = false;
        while num_iter < self.max_iter {
            num_iter += 1;

            // Message processing at CNs.
            for i in 0..base.num_chks {
                let inmsg = &buffer.chk_inmsg[i];
                let (inmsg_sgnpar, total_sgnpar) = sign_parities(inmsg);
                let (minabs1, minabs2, minidx) = two_smallest_abs(inmsg);
                let nbrs = &base.chk_nbrs[i];
                let pos = &base.chk_nbr_pos[i];
                for (k, (&j, &p)) in nbrs.iter().zip(pos).enumerate() {
                    let msg_sgnpar = synd[i] ^ total_sgnpar ^ inmsg_sgnpar[k];
                    let msg_abs = if k == minidx { minabs2 } else { minabs1 };
                    let msg_abs_offset = (msg_abs - self.offset[i][k]).max(0.0);
                    let msg = if msg_sgnpar == 0 {
                        msg_abs_offset
                    } else {
                        -msg_abs_offset
                    };
                    buffer.var_inmsg[j][p] = self.norm[i][k] * msg;
                }
            }

            // Message processing at VNs.
            for j in 0..base.num_vars {
                let inmsg = &buffer.var_inmsg[j];
                llr[j] = (1.0 - self.gamma[j]) * base.prior_llr[j]
                    + self.gamma[j] * llr[j]
                    + inmsg.iter().sum::<f64>();
                ehat[j] = if llr[j] < 0.0 { 1 } else { 0 };
                let nbrs = &base.var_nbrs[j];
                let pos = &base.var_nbr_pos[j];
                for ((&i, &p), &m) in nbrs.iter().zip(pos).zip(inmsg) {
                    buffer.chk_inmsg[i][p] = llr[j] - m;
                }
            }

            if base.syndrome_satisfied(ehat.as_slice().unwrap(), synd) {
                converged = true;
                break;
            }
        }

        (ehat, converged, num_iter)
    }
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
        let chk_lens: Vec<usize> = (0..base.num_chks)
            .map(|i| base.chk_nbrs.row_len(i))
            .collect();
        let offset = Csr::from_rows_with_lens(offset, &chk_lens, "offset")?;
        let norm = Csr::from_rows_with_lens(norm, &chk_lens, "norm")?;

        Ok(Self {
            base,
            gamma: gamma.as_array().to_owned(),
            offset,
            norm,
            max_iter,
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
        &self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, bool, usize)> {
        let mut buf = BPBuffer::new(&self.base);
        let (ehat, converged, num_iter) = self.run(&mut buf, syndrome.as_array());
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
        &self,
        py: Python<'py>,
        syndrome_batch: PyReadonlyArray2<'_, u8>,
        parallel: bool,
    ) -> PyResult<(
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<i64>>,
    )> {
        let synd = syndrome_batch.as_array();
        let (ehat_batch, converged_mask, decoding_iters) =
            py.allow_threads(|| self.run_batch(synd, parallel));
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
