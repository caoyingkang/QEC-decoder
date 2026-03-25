use numpy::ndarray::{Array1, ArrayView1, ArrayView2};

/// Given a probability `p`, return the log-likelihood ratio `ln((1-p)/p)`.
fn prob_to_llr(p: f64) -> f64 {
    // Clamp the probability to [EPS, 1-EPS] to avoid numerical instability.
    const EPS: f64 = 1e-10;
    let pp = if p < EPS {
        EPS
    } else if p > 1.0 - EPS {
        1.0 - EPS
    } else {
        p
    };
    ((1.0 - pp) / pp).ln()
}

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
    pub(crate) fn new(pcm: ArrayView2<u8>, prior: ArrayView1<f64>) -> Self {
        let num_chks = pcm.nrows();
        let num_vars = pcm.ncols();

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
        for i in 0..num_chks {
            assert!(chk_nbrs[i].len() >= 2, "CN {} has less than 2 neighbors", i);
        }
        for j in 0..num_vars {
            assert!(var_nbrs[j].len() >= 1, "VN {} has less than 1 neighbor", j);
        }

        Self {
            prior_llr: prior.mapv(prob_to_llr),
            num_chks: num_chks,
            num_vars: num_vars,
            chk_nbrs: chk_nbrs,
            var_nbrs: var_nbrs,
            chk_nbr_pos: chk_nbr_pos,
            var_nbr_pos: var_nbr_pos,
        }
    }
}
