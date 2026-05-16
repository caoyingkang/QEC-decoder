use crate::bp_base::{BPBase, BPBuffer};
use crate::bp_like::{BPLike, DetBPLike};
use crate::dmembp_core::run_dmembp_in_relay;
use crate::utils::is_all_zeros;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

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
}

impl BPLike for DMemBPDecoderRust {
    fn base(&self) -> &BPBase {
        &self.base
    }
}

impl DetBPLike for DMemBPDecoderRust {
    fn run(&self, buffer: &mut BPBuffer, synd: ArrayView1<u8>) -> (Array1<u8>, bool, usize) {
        let base = &self.base;

        // Return immediately if syndrome is all zeros.
        if is_all_zeros(synd) {
            return (Array1::zeros(base.num_vars), true, 0);
        }

        let mut llr = base.prior_llr.to_vec();
        let mut ehat = vec![0; base.num_vars];
        let (converged, num_iter) = run_dmembp_in_relay(
            base,
            self.gamma.view(),
            self.norm,
            self.max_iter,
            buffer,
            &mut llr,
            &mut ehat,
            synd,
        );

        (Array1::from_vec(ehat), converged, num_iter)
    }
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
    ) -> PyResult<Self> {
        let base = BPBase::new(pcm.as_array(), prior.as_array())?;

        Ok(Self {
            base,
            gamma: gamma.as_array().to_owned(),
            norm: norm.unwrap_or(1.0),
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
        let synd = syndrome.as_array();
        let mut buf = BPBuffer::new(&self.base);
        let (ehat, converged, num_iter) = self.run(&mut buf, synd);
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
    m.add_class::<DMemBPDecoderRust>()?;
    Ok(())
}
