//! Shared trait machinery for BP-like decoders.

use crate::bp_base::{BPBase, BPBuffer};
use crate::utils::spawn_seeds;
use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// General BP-like decoders.
pub(crate) trait BPLike: Sync {
    fn base(&self) -> &BPBase;
}

/// Deterministic BP-like decoders.
pub(crate) trait DetBPLike: BPLike {
    /// Run decoder on a single syndrome vector.
    /// Return `(ehat, converged, num_iter)`.
    ///
    /// Update `buffer` in place. The initial contents of `buffer` are overwritten.
    fn run(&self, buffer: &mut BPBuffer, syndrome: ArrayView1<u8>) -> (Array1<u8>, bool, usize);

    /// Run decoder on a batch of syndrome vectors.
    /// Return `(ehat_batch, converged_mask, decoding_iters)`.
    fn run_batch(
        &self,
        syndrome_batch: ArrayView2<u8>,
        parallel: bool,
    ) -> (Array2<u8>, Array1<bool>, Array1<i64>) {
        let batch_size = syndrome_batch.nrows();
        let base = self.base();

        let results: Vec<(Array1<u8>, bool, usize)> = if parallel {
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let mut buf = BPBuffer::new(base);
                    self.run(&mut buf, syndrome_batch.row(i))
                })
                .collect()
        } else {
            let mut buf = BPBuffer::new(base);
            (0..batch_size)
                .map(|i| self.run(&mut buf, syndrome_batch.row(i)))
                .collect()
        };

        let mut ehat_batch = Array2::zeros((batch_size, base.num_vars));
        let mut converged_mask = Array1::default(batch_size);
        let mut decoding_iters = Array1::zeros(batch_size);

        for (i, (ehat, converged, num_iter)) in results.into_iter().enumerate() {
            ehat_batch.row_mut(i).assign(&ehat);
            converged_mask[i] = converged;
            decoding_iters[i] = num_iter as i64;
        }

        (ehat_batch, converged_mask, decoding_iters)
    }
}

/// Randomized BP-like decoders.
pub(crate) trait RandBPLike: BPLike {
    /// Run decoder on a single syndrome vector.
    /// Return `(ehat, converged, num_iter)`.
    ///
    /// Update `buffer` in place. The initial contents of `buffer` are overwritten.
    fn run(
        &self,
        buffer: &mut BPBuffer,
        syndrome: ArrayView1<u8>,
        seed: Option<u64>,
    ) -> (Array1<u8>, bool, usize);

    /// Run decoder on a batch of syndrome vectors.
    /// Return `(ehat_batch, converged_mask, decoding_iters)`.
    fn run_batch(
        &self,
        syndrome_batch: ArrayView2<u8>,
        parallel: bool,
        seed: Option<u64>,
    ) -> (Array2<u8>, Array1<bool>, Array1<i64>) {
        let batch_size = syndrome_batch.nrows();
        let base = self.base();

        let child_seeds = spawn_seeds(seed, batch_size);

        let results: Vec<(Array1<u8>, bool, usize)> = if parallel {
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let mut buf = BPBuffer::new(base);
                    self.run(&mut buf, syndrome_batch.row(i), child_seeds[i])
                })
                .collect()
        } else {
            let mut buf = BPBuffer::new(base);
            (0..batch_size)
                .map(|i| self.run(&mut buf, syndrome_batch.row(i), child_seeds[i]))
                .collect()
        };

        let mut ehat_batch = Array2::zeros((batch_size, base.num_vars));
        let mut converged_mask = Array1::default(batch_size);
        let mut decoding_iters = Array1::zeros(batch_size);

        for (i, (ehat, converged, num_iter)) in results.into_iter().enumerate() {
            ehat_batch.row_mut(i).assign(&ehat);
            converged_mask[i] = converged;
            decoding_iters[i] = num_iter as i64;
        }

        (ehat_batch, converged_mask, decoding_iters)
    }
}
