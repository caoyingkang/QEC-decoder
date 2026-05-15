use crate::bp_base::{init_v2c_msg, BPBase};
use crate::utils::{sign_parities, two_smallest_abs};
use numpy::ndarray::ArrayView1;

/// Subroutine of `run_dmembp` and `run_relaybp`. Return `(converged, num_iter)`.
///
/// Update `chk_inmsg`, `var_inmsg`, `llr`, and `ehat` in place. The caller is
/// responsible for initializing `llr`: for standalone DMemBP and the first stage
/// of RelayBP, `llr` should be initialized to `base.prior_llr`; for subsequent
/// stages of RelayBP, `llr` should be passed from the previous stage. `chk_inmsg`,
/// `var_inmsg`, and `ehat` only need to be sized correctly; their initial values
/// will be overwritten. It is assumed that `synd` is not all-zeros.
pub(crate) fn run_dmembp_in_relay(
    base: &BPBase,
    gamma: ArrayView1<f64>,
    norm: f64,
    max_iter: usize,
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    llr: &mut [f64],
    ehat: &mut [u8],
    synd: ArrayView1<u8>,
) -> (bool, usize) {
    init_v2c_msg(base, chk_inmsg);

    // Main BP iteration loop.
    let mut num_iter = 0;
    let mut converged = false;
    while num_iter < max_iter {
        num_iter += 1;

        // Message processing at CNs.
        for i in 0..base.num_chks {
            let inmsg = &chk_inmsg[i];
            let (inmsg_sgnpar, total_sgnpar) = sign_parities(inmsg);
            let (minabs1, minabs2, minidx) = two_smallest_abs(inmsg);
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
        if base.syndrome_satisfied(ehat, synd) {
            converged = true;
            break;
        }
    }

    (converged, num_iter)
}
