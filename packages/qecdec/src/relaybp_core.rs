use crate::bp_base::BPBase;
use crate::dmembp_core::run_dmembp_in_relay;
use crate::utils::{is_all_zeros, make_pcg64_rng, pick_most_likely, sample_vec_uniform};
use numpy::ndarray::{Array1, ArrayView1};
use rand::distr::Uniform;

/// Run RelayBP decoding algorithm. Return `(ehat, converged, num_iter)`.
///
/// Update `chk_inmsg` and `var_inmsg` in place. `chk_inmsg` and `var_inmsg`
/// only need to be sized correctly; their initial values will be overwritten.
pub(crate) fn run_relaybp(
    base: &BPBase,
    gamma0: ArrayView1<f64>,
    gamma_dist: &Uniform<f64>,
    num_relays: usize,
    pre_iter: usize,
    max_iter_per_relay: usize,
    stop_nconv: usize,
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    synd: ArrayView1<u8>,
    seed: Option<u64>,
) -> (Array1<u8>, bool, usize) {
    // Return immediately if syndrome is all zeros.
    if is_all_zeros(synd) {
        return (Array1::zeros(base.num_vars), true, 0);
    }

    let mut llr = base.prior_llr.to_vec();
    let mut ehat = vec![0; base.num_vars];
    let mut num_iters = 0;
    let mut candidates = Vec::with_capacity(stop_nconv);

    // Stage 0: user-provided gamma0.
    let (conv, it) = run_dmembp_in_relay(
        base, gamma0, 1.0, pre_iter, chk_inmsg, var_inmsg, &mut llr, &mut ehat, synd,
    );
    num_iters += it;
    if conv {
        candidates.push(ehat.clone());
    }

    // Stages 1, 2, ..., num_relays: random gamma.
    if candidates.len() < stop_nconv {
        let mut rng = make_pcg64_rng(seed);
        let mut gamma = Array1::zeros(base.num_vars);
        for _ in 0..num_relays {
            sample_vec_uniform(gamma.view_mut(), gamma_dist, &mut rng);
            let (conv, it) = run_dmembp_in_relay(
                base,
                gamma.view(),
                1.0,
                max_iter_per_relay,
                chk_inmsg,
                var_inmsg,
                &mut llr,
                &mut ehat,
                synd,
            );
            num_iters += it;
            if conv {
                candidates.push(ehat.clone());
                if candidates.len() == stop_nconv {
                    break;
                }
            }
        }
    }

    if candidates.is_empty() {
        (Array1::from_vec(ehat), false, num_iters)
    } else {
        let best_ehat = pick_most_likely(candidates, base.prior_llr.view());
        (Array1::from_vec(best_ehat), true, num_iters)
    }
}
