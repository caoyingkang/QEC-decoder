use crate::bp_base::{init_v2c_msg, BPBase, BPBuffer};
use crate::utils::{sign_parities, two_smallest_abs};
use numpy::ndarray::ArrayView1;

/// Subroutine of `run_dmembp` and `run_relaybp`. Return `(converged, num_iter)`.
///
/// Update `buffer`, `llr`, and `ehat` in place. The caller is responsible for
/// initializing `llr`: for standalone DMemBP and the first stage of RelayBP,
/// `llr` should be initialized to `base.prior_llr`; for subsequent stages of
/// RelayBP, `llr` should be passed from the previous stage. The initial contents
/// of `buffer` and `ehat` are overwritten. It is assumed that `synd` is not
/// all-zeros.
pub(crate) fn run_dmembp_in_relay(
    base: &BPBase,
    gamma: ArrayView1<f64>,
    norm: f64,
    max_iter: usize,
    buffer: &mut BPBuffer,
    llr: &mut [f64],
    ehat: &mut [u8],
    synd: ArrayView1<u8>,
) -> (bool, usize) {
    init_v2c_msg(base, buffer);

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
            let nbrs = &base.chk_nbrs[i];
            let pos = &base.chk_nbr_pos[i];
            for (k, (&j, &p)) in nbrs.iter().zip(pos).enumerate() {
                let msg_sgnpar = synd[i] ^ total_sgnpar ^ inmsg_sgnpar[k];
                let msg_abs = if k == minidx { minabs2 } else { minabs1 };
                let msg = if msg_sgnpar == 0 { msg_abs } else { -msg_abs };
                buffer.var_inmsg[j][p] = norm * msg;
            }
        }

        // Message processing at VNs.
        for j in 0..base.num_vars {
            let inmsg = &buffer.var_inmsg[j];
            llr[j] = (1.0 - gamma[j]) * base.prior_llr[j]
                + gamma[j] * llr[j]
                + inmsg.iter().sum::<f64>();
            ehat[j] = if llr[j] < 0.0 { 1 } else { 0 };
            let nbrs = &base.var_nbrs[j];
            let pos = &base.var_nbr_pos[j];
            for ((&i, &p), &m) in nbrs.iter().zip(pos).zip(inmsg) {
                buffer.chk_inmsg[i][p] = llr[j] - m;
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
