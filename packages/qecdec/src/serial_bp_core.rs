//! One iteration of the serial-schedule min-sum BP decoder.

use crate::bp_base::{BPBase, BPBuffer};
use numpy::ndarray::ArrayView1;

/// Run one full pass of serial-schedule min-sum BP over `vn_order`, then check
/// whether the current `ehat` satisfies the syndrome.
///
/// Updates `buffer`, `llr`, and `ehat` in place. The caller is responsible for
/// initializing `buffer.chk_inmsg` before the first call. Any initial values of
/// `buffer.var_inmsg`, `llr`, and `ehat` will be overwritten.
///
/// Return `true` iff the syndrome is satisfied after this iteration.
pub(crate) fn run_serial_bp_one_iteration(
    base: &BPBase,
    vn_order: &[usize],
    buffer: &mut BPBuffer,
    llr: &mut [f64],
    ehat: &mut [u8],
    synd: ArrayView1<u8>,
) -> bool {
    for &v in vn_order.iter() {
        // Update c->v message for all neighbor c of v.
        for (k, &c) in base.var_nbrs[v].iter().enumerate() {
            let v_pos = base.var_nbr_pos[v][k];
            let inmsg = &buffer.chk_inmsg[c];
            let mut sgnpar = synd[c];
            let mut minabs = f64::MAX;
            for (kk, &val) in inmsg.iter().enumerate() {
                if kk == v_pos {
                    continue;
                }
                sgnpar ^= if val < 0.0 { 1 } else { 0 };
                let val_abs = val.abs();
                if val_abs < minabs {
                    minabs = val_abs;
                }
            }
            let msg = if sgnpar == 0 { minabs } else { -minabs };
            buffer.var_inmsg[v][k] = msg;
        }

        // Update posterior LLR and hard decision at VN v.
        llr[v] = base.prior_llr[v] + buffer.var_inmsg[v].iter().sum::<f64>();
        ehat[v] = if llr[v] < 0.0 { 1 } else { 0 };

        // Update v->c message for all neighbor c of v.
        for (k, &c) in base.var_nbrs[v].iter().enumerate() {
            let v_pos = base.var_nbr_pos[v][k];
            buffer.chk_inmsg[c][v_pos] = llr[v] - buffer.var_inmsg[v][k];
        }
    }

    // Check if the syndrome is satisfied.
    base.syndrome_satisfied(ehat, synd)
}
