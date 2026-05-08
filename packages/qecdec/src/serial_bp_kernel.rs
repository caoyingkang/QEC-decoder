//! One-iteration kernel for the serial-schedule min-sum BP decoder.

use crate::bp_base::BPBase;
use numpy::ndarray::ArrayView1;

/// Run one full pass of serial-schedule min-sum BP over `vn_order`, then check
/// whether the current `ehat` satisfies the syndrome.
///
/// Updates `chk_inmsg`, `var_inmsg`, `llr`, and `ehat` in place. The caller is
/// responsible for initializing `chk_inmsg` via `BPBase::init_messages` before
/// the first call. `llr` and `ehat` only need to be sized correctly; their
/// initial values are overwritten as each VN in `vn_order` is visited.
///
/// Returns `true` iff the parity of `ehat` matches `synd` after this iteration.
pub(crate) fn run_serial_bp_iteration(
    base: &BPBase,
    vn_order: &[usize],
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    llr: &mut [f64],
    ehat: &mut [u8],
    synd: ArrayView1<u8>,
) -> bool {
    for &v in vn_order.iter() {
        // Update c->v message for all neighbor c of v.
        for (k, &c) in base.var_nbrs[v].iter().enumerate() {
            let v_pos = base.var_nbr_pos[v][k];
            let inmsg = &chk_inmsg[c];
            let mut sgnpar = synd[c];
            let mut minabs = f64::MAX;
            for (kk, &val) in inmsg.iter().enumerate() {
                if kk == v_pos {
                    continue;
                }
                let s: u8 = if val < 0.0 { 1 } else { 0 };
                sgnpar ^= s;
                let val_abs = val.abs();
                if val_abs < minabs {
                    minabs = val_abs;
                }
            }
            let msg = if sgnpar == 0 { minabs } else { -minabs };
            var_inmsg[v][k] = msg;
        }

        // Update posterior LLR and hard decision at VN v.
        llr[v] = base.prior_llr[v] + var_inmsg[v].iter().sum::<f64>();
        ehat[v] = if llr[v] < 0.0 { 1 } else { 0 };

        // Update v->c message for all neighbor c of v.
        for (k, &c) in base.var_nbrs[v].iter().enumerate() {
            let v_pos = base.var_nbr_pos[v][k];
            chk_inmsg[c][v_pos] = llr[v] - var_inmsg[v][k];
        }
    }

    // Check if the syndrome is satisfied.
    for c in 0..base.num_chks {
        let mut parity = 0_u8;
        for &v in base.chk_nbrs[c].iter() {
            parity ^= ehat[v];
        }
        if parity != synd[c] {
            return false;
        }
    }
    true
}
