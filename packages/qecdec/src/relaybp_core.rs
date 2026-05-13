use crate::bp_base::BPBase;
use crate::dmembp_core::run_dmembp_in_relay;
use crate::utils::{make_pcg64_rng, pick_most_likely, sample_vec_uniform};
use numpy::ndarray::{Array1, ArrayView1};
use rand::distr::Uniform;

/// Run the random-gamma relay stages of RelayBP starting from pre-populated state.
/// Used by both RelayBP (after its own stage 0) and MultiRelayBP (where many
/// chains share a single stage 0). Return `(ehat, converged, num_iter)`.
///
/// Update `chk_inmsg`, `var_inmsg`, `llr`, and `ehat` in place. The caller is
/// responsible for making sure that `llr` is passed from the initial stage of RelayBP.
/// `candidates` may already contain an entry from a prior stage (e.g., a converged
/// stage 0 result); `num_iters_init` is the iteration count accumulated before this
/// call.
pub(crate) fn run_random_relays(
    base: &BPBase,
    gamma_dist: &Uniform<f64>,
    num_relays: usize,
    max_iter_per_relay: usize,
    stop_nconv: usize,
    chk_inmsg: &mut [Vec<f64>],
    var_inmsg: &mut [Vec<f64>],
    llr: &mut [f64],
    ehat: &mut [u8],
    synd: ArrayView1<u8>,
    mut candidates: Vec<Vec<u8>>,
    num_iters_init: usize,
    seed: Option<u64>,
) -> (Array1<u8>, bool, usize) {
    let mut num_iters = num_iters_init;
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
                llr,
                ehat,
                synd,
            );
            num_iters += it;
            if conv {
                candidates.push(ehat.to_vec());
                if candidates.len() == stop_nconv {
                    break;
                }
            }
        }
    }

    if candidates.is_empty() {
        (Array1::from_vec(ehat.to_vec()), false, num_iters)
    } else {
        let best_ehat = pick_most_likely(candidates, base.prior_llr.view());
        (Array1::from_vec(best_ehat), true, num_iters)
    }
}
