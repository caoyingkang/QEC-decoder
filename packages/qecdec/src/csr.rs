//! Compressed-sparse-row-style flat storage for variable-length rows.

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use std::ops::{Index, IndexMut};

/// Sequence of variable-length rows stored in one contiguous allocation.
#[derive(Clone, Debug)]
pub(crate) struct Csr<T> {
    data: Vec<T>,
    /// Length `num_rows + 1`. `offsets[0] == 0`, `offsets[num_rows] == data.len()`.
    offsets: Vec<usize>,
}

impl<T> Csr<T> {
    /// Length of the `i`-th row.
    pub(crate) fn row_len(&self, i: usize) -> usize {
        self.offsets[i + 1] - self.offsets[i]
    }

    /// Build from a ragged `Vec<Vec<T>>`, concatenating into one allocation.
    pub(crate) fn from_rows(rows: Vec<Vec<T>>) -> Self {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0);
        let total = rows.iter().map(|r| r.len()).sum();
        let mut data = Vec::with_capacity(total);
        for mut row in rows {
            data.append(&mut row);
            offsets.push(data.len());
        }
        Self { data, offsets }
    }

    /// Build from `Vec<Vec<T>>` and require each row's length to match `expected_lens`.
    /// `name` is used in the error message to identify the offending parameter.
    pub(crate) fn from_rows_with_lens(
        rows: Vec<Vec<T>>,
        expected_lens: &[usize],
        name: &str,
    ) -> PyResult<Self> {
        if rows.len() != expected_lens.len() {
            return Err(PyValueError::new_err(format!(
                "`{}` has {} rows, expected {}",
                name,
                rows.len(),
                expected_lens.len()
            )));
        }
        for (i, (row, &l)) in rows.iter().zip(expected_lens).enumerate() {
            if row.len() != l {
                return Err(PyValueError::new_err(format!(
                    "`{}` row {} has length {}, expected {}",
                    name,
                    i,
                    row.len(),
                    l
                )));
            }
        }
        Ok(Self::from_rows(rows))
    }
}

impl<T: Clone + Default> Csr<T> {
    /// Allocate a zero-initialized CSR with the given per-row lengths.
    pub(crate) fn zeros(row_lens: &[usize]) -> Self {
        let mut offsets = Vec::with_capacity(row_lens.len() + 1);
        offsets.push(0);
        let mut acc = 0;
        for &len in row_lens {
            acc += len;
            offsets.push(acc);
        }
        Self {
            data: vec![T::default(); acc],
            offsets,
        }
    }
}

impl<T> Index<usize> for Csr<T> {
    type Output = [T];

    fn index(&self, i: usize) -> &[T] {
        &self.data[self.offsets[i]..self.offsets[i + 1]]
    }
}

impl<T> IndexMut<usize> for Csr<T> {
    fn index_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.data[self.offsets[i]..self.offsets[i + 1]]
    }
}
