use crate::bp_base::{alloc_msg_buffers, BPBase};
use crate::dmembp_core::run_dmembp_in_relay;
use crate::utils::is_all_zeros;
use numpy::ndarray::{Array1, Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Run disordered-memory BP decoding algorithm. Return `(ehat, converged, num_iter)`.
///
/// Update `chk_inmsg` and `var_inmsg` in place. `chk_inmsg` and `var_inmsg`
/// only need to be sized correctly; their initial values will be overwritten.
fn run_dmembp(
    base: &BPBase,
    gamma: ArrayView1<f64>,
    norm: f64,
    max_iter: usize,
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    synd: ArrayView1<u8>,
) -> (Array1<u8>, bool, usize) {
    // Return immediately if syndrome is all zeros.
    if is_all_zeros(synd) {
        return (Array1::zeros(base.num_vars), true, 0);
    }

    let mut llr = base.prior_llr.to_vec();
    let mut ehat = vec![0; base.num_vars];
    let (converged, num_iter) = run_dmembp_in_relay(
        base, gamma, norm, max_iter, chk_inmsg, var_inmsg, &mut llr, &mut ehat, synd,
    );

    (Array1::from_vec(ehat), converged, num_iter)
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
    /// - `pcm`: Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
    /// - `prior`: Prior error probabilities, shape `(num_vars,)`.
    /// - `gamma`: Per-VN memory strength, shape `(num_vars,)`.
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
        let base = BPBase::new(pcm, prior);
        let norm = norm.unwrap_or(1.0);
        let (chk_inmsg, var_inmsg) = alloc_msg_buffers(&base);

        Self {
            base,
            gamma: gamma.as_array().to_owned(),
            norm,
            max_iter,
            chk_inmsg,
            var_inmsg,
        }
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
        let (ehat, converged, num_iter) = run_dmembp(
            &self.base,
            self.gamma.view(),
            self.norm,
            self.max_iter,
            &mut self.chk_inmsg,
            &mut self.var_inmsg,
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
                        let (ehat, converged, num_iter) = run_dmembp(
                            base,
                            gamma.view(),
                            norm,
                            max_iter,
                            &mut chk_inmsg,
                            &mut var_inmsg,
                            syndrome_batch.row(i),
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
                let (ehat, converged, num_iter) = run_dmembp(
                    &self.base,
                    self.gamma.view(),
                    self.norm,
                    self.max_iter,
                    &mut self.chk_inmsg,
                    &mut self.var_inmsg,
                    syndrome_batch.row(i),
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
    m.add_class::<DMemBPDecoderRust>()?;
    Ok(())
}
