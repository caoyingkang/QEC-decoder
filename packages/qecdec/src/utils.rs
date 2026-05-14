use numpy::ndarray::{ArrayView1, ArrayViewMut1};
use rand::distr::{Distribution, Uniform};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Given a probability `p`, return the log-likelihood ratio `ln((1-p)/p)`.
pub(crate) fn prob_to_llr(p: f64) -> f64 {
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

/// Check if all elements of `arr` are zero.
pub(crate) fn is_all_zeros(arr: ArrayView1<u8>) -> bool {
    arr.iter().all(|&x| x == 0)
}

/// LLR-weight of an error pattern: Σ_j llr\[j\] * ehat\[j\].
/// Smaller weight ↔ more likely error pattern.
pub(crate) fn llr_weight(llr: ArrayView1<f64>, ehat: &[u8]) -> f64 {
    let mut w = 0.0;
    for (j, &e) in ehat.iter().enumerate() {
        if e != 0 {
            w += llr[j];
        }
    }
    w
}

/// Pick the most likely error vector (i.e., the one with the smallest LLR-weight)
/// among the nonempty `candidates` list. Ties broken by encounter order.
///
/// Panic if `candidates` is empty.
pub(crate) fn pick_most_likely(candidates: Vec<Vec<u8>>, prior_llr: ArrayView1<f64>) -> Vec<u8> {
    let cost_fn = |e: &[u8]| -> f64 { llr_weight(prior_llr, e) };
    candidates
        .into_iter()
        .min_by(|e1, e2| cost_fn(e1).partial_cmp(&cost_fn(e2)).unwrap())
        .expect("candidates must be non-empty")
}

/// Build a Pcg64 RNG from a u64 `seed`. If `seed` is `None`, use OS entropy.
pub(crate) fn make_pcg64_rng(seed: Option<u64>) -> Pcg64 {
    match seed {
        Some(s) => Pcg64::seed_from_u64(s),
        None => rand::make_rng(),
    }
}

/// Spawn `n` random seeds used for downstream applications from a `master_seed`.
/// If `master_seed` is `None`, then the output is a vector of `None`'s.
pub(crate) fn spawn_seeds(master_seed: Option<u64>, n: usize) -> Vec<Option<u64>> {
    match master_seed {
        Some(s) => {
            let mut rng = Pcg64::seed_from_u64(s);
            (0..n).map(|_| Some(rng.next_u64())).collect()
        }
        None => vec![None; n],
    }
}

/// Sample vector components i.i.d. uniformly from `dist` into `out`.
pub(crate) fn sample_vec_uniform(
    mut out: ArrayViewMut1<f64>,
    dist: &Uniform<f64>,
    rng: &mut impl Rng,
) {
    for el in out.iter_mut() {
        *el = dist.sample(rng);
    }
}
