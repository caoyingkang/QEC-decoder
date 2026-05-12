use numpy::ndarray::ArrayView1;

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
