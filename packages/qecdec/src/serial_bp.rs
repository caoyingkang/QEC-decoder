use crate::bp_base::{alloc_msg_buffers, init_v2c_msg, BPBase};
use crate::serial_bp_core::run_serial_bp_one_iteration;
use crate::utils::is_all_zeros;
use numpy::ndarray::{Array1, Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Run serial-schedule min-sum BP decoding algorithm.
/// Return `(ehat, converged, num_iter)`.
///
/// Convergence is checked once per iteration (iteration = one full pass
/// over `vn_order`). Update `chk_inmsg` and `var_inmsg` in place.
/// `chk_inmsg` and `var_inmsg` only need to be sized correctly; their
/// initial values will be overwritten.
fn run_serial_bp(
    base: &BPBase,
    vn_order: &[usize],
    max_iter: usize,
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    synd: ArrayView1<u8>,
) -> (Array1<u8>, bool, usize) {
    // Return immediately if syndrome is all zeros.
    if is_all_zeros(synd) {
        return (Array1::zeros(base.num_vars), true, 0);
    }

    init_v2c_msg(base, chk_inmsg);
    let mut ehat = vec![0_u8; base.num_vars];
    let mut llr = vec![0.0; base.num_vars];

    // Main BP iteration loop.
    let mut num_iter = 0;
    let mut converged = false;
    while num_iter < max_iter {
        num_iter += 1;
        converged = run_serial_bp_one_iteration(
            base, vn_order, chk_inmsg, var_inmsg, &mut llr, &mut ehat, synd,
        );
        if converged {
            break;
        }
    }

    (Array1::from_vec(ehat), converged, num_iter)
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
        let pcm = pcm.as_array();
        let prior = prior.as_array();
        let base = BPBase::new(pcm, prior)?;
        let (chk_inmsg, var_inmsg) = alloc_msg_buffers(&base);

        Ok(Self {
            base,
            vn_order: vn_order.as_array().iter().map(|&x| x as usize).collect(),
            max_iter,
            chk_inmsg,
            var_inmsg,
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
        &mut self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, bool, usize)> {
        let (ehat, converged, num_iter) = run_serial_bp(
            &self.base,
            &self.vn_order,
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
    /// - `decoding_iters`: Number of iterations actually run in each shot.
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
                        let (ehat, converged, num_iter) = run_serial_bp(
                            base,
                            vn_order,
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
                let (ehat, converged, num_iter) = run_serial_bp(
                    &self.base,
                    &self.vn_order,
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
    m.add_class::<SerialBPDecoderRust>()?;
    Ok(())
}
