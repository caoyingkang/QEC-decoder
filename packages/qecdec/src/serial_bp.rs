use crate::bp_base::{init_v2c_msg, BPBase, BPBuffer};
use crate::bp_like::{BPLike, DetBPLike};
use crate::serial_bp_core::run_serial_bp_one_iteration;
use crate::utils::is_all_zeros;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Belief Propagation decoder with serial message passing schedule and min-sum CN update rule.
#[pyclass]
pub struct SerialBPDecoderRust {
    /// Base struct for BP-based decoders, which stores parity-check matrix and prior error probabilities.
    base: BPBase,
    /// Permutation of variable nodes.
    vn_order: Vec<usize>,
    /// Maximum number of iterations (one iteration = one full pass over `vn_order`).
    max_iter: usize,
}

impl BPLike for SerialBPDecoderRust {
    fn base(&self) -> &BPBase {
        &self.base
    }
}

impl DetBPLike for SerialBPDecoderRust {
    fn run(&self, buffer: &mut BPBuffer, synd: ArrayView1<u8>) -> (Array1<u8>, bool, usize) {
        let base = &self.base;

        // Return immediately if syndrome is all zeros.
        if is_all_zeros(synd) {
            return (Array1::zeros(base.num_vars), true, 0);
        }

        init_v2c_msg(base, buffer);
        let mut ehat = vec![0_u8; base.num_vars];
        let mut llr = vec![0.0; base.num_vars];

        let mut num_iter = 0;
        let mut converged = false;
        while num_iter < self.max_iter {
            num_iter += 1;
            converged = run_serial_bp_one_iteration(
                base,
                &self.vn_order,
                buffer,
                &mut llr,
                &mut ehat,
                synd,
            );
            if converged {
                break;
            }
        }

        (Array1::from_vec(ehat), converged, num_iter)
    }
}

#[pymethods]
impl SerialBPDecoderRust {
    /// Create a serial-schedule BP decoder.
    ///
    /// Parameters:
    /// - `pcm`: Parity-check matrix. Each row has ≥2 nonzeros; each column has ≥1 nonzero.
    /// - `prior`: Prior error probabilities.
    /// - `vn_order`: Permutation of variable nodes.
    /// - `max_iter`: Maximum number of iterations (one iteration = one full pass over `vn_order`).
    #[new]
    #[pyo3(signature = (pcm, prior, *, vn_order, max_iter))]
    pub fn new(
        pcm: PyReadonlyArray2<'_, u8>,
        prior: PyReadonlyArray1<'_, f64>,
        vn_order: PyReadonlyArray1<'_, i64>,
        max_iter: usize,
    ) -> PyResult<Self> {
        let base = BPBase::new(pcm.as_array(), prior.as_array())?;

        Ok(Self {
            base,
            vn_order: vn_order.as_array().iter().map(|&x| x as usize).collect(),
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
    /// - `num_iter`: The number of iterations actually run.
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
    /// - `decoding_iters`: Number of iterations actually run in each shot.
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
    m.add_class::<SerialBPDecoderRust>()?;
    Ok(())
}
