use crate::utils::prob_to_llr;
use numpy::ndarray::{Array1, ArrayView1, ArrayView2};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// Base struct for BP-based decoders.
pub(crate) struct BPBase {
    /// Log-likelihood ratios of the prior error probabilities.
    pub(crate) prior_llr: Array1<f64>,
    /// Number of check nodes (= number of rows of pcm).
    pub(crate) num_chks: usize,
    /// Number of variable nodes (= number of columns of pcm).
    pub(crate) num_vars: usize,
    /// `chk_nbrs[i]` is the list of VNs connected to CN `i` (ordered by VN indices).
    pub(crate) chk_nbrs: Vec<Vec<usize>>,
    /// `var_nbrs[j]` is the list of CNs connected to VN `j` (ordered by CN indices).
    pub(crate) var_nbrs: Vec<Vec<usize>>,
    /// `chk_nbr_pos[i][k]` is the relative position of CN `i` in the list of neighbors of the VN `chk_nbrs[i][k]`.
    /// I.e., if `chk_nbrs[i][k] == j`, then `var_nbrs[j][chk_nbr_pos[i][k]] == i`.
    pub(crate) chk_nbr_pos: Vec<Vec<usize>>,
    /// `var_nbr_pos[j][k]` is the relative position of VN `j` in the list of neighbors of the CN `var_nbrs[j][k]`.
    /// I.e., if `var_nbrs[j][k] == i`, then `chk_nbrs[i][var_nbr_pos[j][k]] == j`.
    pub(crate) var_nbr_pos: Vec<Vec<usize>>,
}

impl BPBase {
    pub(crate) fn new(pcm: ArrayView2<u8>, prior: ArrayView1<f64>) -> PyResult<Self> {
        let num_chks = pcm.nrows();
        let num_vars = pcm.ncols();

        if prior.len() != num_vars {
            return Err(PyValueError::new_err(format!(
                "prior length ({}) must equal number of variable nodes ({})",
                prior.len(),
                num_vars
            )));
        }

        let mut chk_nbrs = vec![Vec::new(); num_chks];
        let mut var_nbrs = vec![Vec::new(); num_vars];
        let mut chk_nbr_pos = vec![Vec::new(); num_chks];
        let mut var_nbr_pos = vec![Vec::new(); num_vars];
        for i in 0..num_chks {
            for j in 0..num_vars {
                if pcm[[i, j]] != 0 {
                    chk_nbr_pos[i].push(var_nbrs[j].len());
                    var_nbr_pos[j].push(chk_nbrs[i].len());
                    chk_nbrs[i].push(j);
                    var_nbrs[j].push(i);
                }
            }
        }
        for (i, nbrs) in chk_nbrs.iter().enumerate() {
            if nbrs.len() < 2 {
                return Err(PyValueError::new_err(format!(
                    "CN {} has less than 2 neighbors",
                    i
                )));
            }
        }
        for (j, nbrs) in var_nbrs.iter().enumerate() {
            if nbrs.is_empty() {
                return Err(PyValueError::new_err(format!("VN {} has zero neighbor", j)));
            }
        }

        Ok(Self {
            prior_llr: prior.mapv(prob_to_llr),
            num_chks,
            num_vars,
            chk_nbrs,
            var_nbrs,
            chk_nbr_pos,
            var_nbr_pos,
        })
    }

    /// Check whether the candidate error pattern `ehat` produces the syndrome `synd`.
    pub(crate) fn syndrome_satisfied(&self, ehat: &[u8], synd: ArrayView1<u8>) -> bool {
        for i in 0..self.num_chks {
            let mut parity = 0_u8;
            for &j in self.chk_nbrs[i].iter() {
                parity ^= ehat[j];
            }
            if parity != synd[i] {
                return false;
            }
        }
        true
    }
}

/// Message buffer for a BP decoder run.
#[derive(Clone)]
pub(crate) struct BPBuffer {
    /// `chk_inmsg[i]` stores the incoming messages at CN `i` from its neighboring VNs.
    pub(crate) chk_inmsg: Vec<Vec<f64>>,
    /// `var_inmsg[j]` stores the incoming messages at VN `j` from its neighboring CNs.
    pub(crate) var_inmsg: Vec<Vec<f64>>,
}

impl BPBuffer {
    /// Allocate fresh per-node message buffers sized to the Tanner graph degrees.
    pub(crate) fn new(base: &BPBase) -> Self {
        let mut chk_inmsg = Vec::with_capacity(base.num_chks);
        for i in 0..base.num_chks {
            chk_inmsg.push(vec![0.0; base.chk_nbrs[i].len()]);
        }
        let mut var_inmsg = Vec::with_capacity(base.num_vars);
        for j in 0..base.num_vars {
            var_inmsg.push(vec![0.0; base.var_nbrs[j].len()]);
        }
        Self {
            chk_inmsg,
            var_inmsg,
        }
    }
}

/// Initialize VN-to-CN messages from prior LLRs.
pub(crate) fn init_v2c_msg(base: &BPBase, buffer: &mut BPBuffer) {
    for (j, &value) in base.prior_llr.iter().enumerate() {
        for (k, &i) in base.var_nbrs[j].iter().enumerate() {
            buffer.chk_inmsg[i][base.var_nbr_pos[j][k]] = value;
        }
    }
}
