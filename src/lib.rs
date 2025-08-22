use std::f64;

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// Given a probability p, return the log-likelihood ratio ln((1-p)/p).
fn prob_to_llr(p: f64) -> f64 {
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

struct DecoderBase {
    // parity-check matrix
    pcm: Array2<u8>,
    // prior probabilities of errors
    prior: Array1<f64>,
    // number of rows of pcm (equivalently, number of CNs in Tanner graph)
    m: usize,
    // number of columns of pcm (equivalently, number of VNs in Tanner graph)
    n: usize,
    // chk_neighbors[i] = list of VNs connected to CN i (ordered by VN indices)
    chk_neighbors: Vec<Vec<usize>>,
    // var_neighbors[j] = list of CNs connected to VN j (ordered by CN indices)
    var_neighbors: Vec<Vec<usize>>,
    // chk_neighbor_pos[i][k] = position of CN i in the list of neighbors of the VN chk_neighbors[i][k].
    // i.e., if chk_neighbors[i][k] = j, then var_neighbors[j][chk_neighbor_pos[i][k]] = i.
    chk_neighbor_pos: Vec<Vec<usize>>,
    // var_neighbor_pos[j][k] = position of VN j in the list of neighbors of the CN var_neighbors[j][k].
    // i.e., if var_neighbors[j][k] = i, then chk_neighbors[i][var_neighbor_pos[j][k]] = j.
    var_neighbor_pos: Vec<Vec<usize>>,
}

#[pyclass]
pub struct BPDecoder {
    // base struct for storing parity-check matrix and prior probabilities of errors
    base: DecoderBase,
    // maximum number of iterations
    max_iter: usize,
    // chk_incoming_msgs[i] = list of incoming messages at CN i
    chk_incoming_msgs: Vec<Vec<f64>>,
    // var_incoming_msgs[j] = list of incoming messages at VN j
    var_incoming_msgs: Vec<Vec<f64>>,
    // prior LLR values
    prior_llr: Vec<f64>,
    // posterior LLR values
    llr: Vec<f64>,
    // estimated error vector
    ehat: Array1<u8>,
    // syndrome vector
    syndrome: Array1<u8>,
}

#[pymethods]
impl BPDecoder {
    #[new]
    pub fn new(
        pcm: PyReadonlyArray2<'_, u8>, // parity-check matrix (m x n, np.uint8)
        prior: PyReadonlyArray1<'_, f64>, // prior probabilities of errors (n, np.float64)
        max_iter: usize,               // maximum number of BP iterations
    ) -> PyResult<Self> {
        let pcm_arr = pcm.as_array();
        let m: usize = pcm_arr.shape()[0];
        let n: usize = pcm_arr.shape()[1];

        let prior_arr = prior.as_array();

        let mut chk_neighbors: Vec<Vec<usize>> = vec![Vec::new(); m];
        let mut var_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut chk_neighbor_pos: Vec<Vec<usize>> = vec![Vec::new(); m];
        let mut var_neighbor_pos: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..m {
            for j in 0..n {
                if pcm_arr[(i, j)] != 0 {
                    chk_neighbor_pos[i].push(var_neighbors[j].len());
                    var_neighbor_pos[j].push(chk_neighbors[i].len());
                    chk_neighbors[i].push(j);
                    var_neighbors[j].push(i);
                }
            }
        }
        for i in 0..m {
            assert!(
                chk_neighbors[i].len() >= 2,
                "CN {} has less than 2 neighbors",
                i
            );
        }
        for j in 0..n {
            assert!(
                var_neighbors[j].len() >= 1,
                "VN {} has less than 1 neighbor",
                j
            );
        }

        let mut var_incoming_msgs: Vec<Vec<f64>> = Vec::new();
        for j in 0..n {
            var_incoming_msgs.push(vec![0.0; var_neighbors[j].len()]);
        }

        let mut chk_incoming_msgs: Vec<Vec<f64>> = Vec::new();
        for i in 0..m {
            chk_incoming_msgs.push(vec![0.0; chk_neighbors[i].len()]);
        }

        Ok(Self {
            base: DecoderBase {
                pcm: pcm_arr.to_owned(),
                prior: prior_arr.to_owned(),
                m: m,
                n: n,
                chk_neighbors: chk_neighbors,
                var_neighbors: var_neighbors,
                chk_neighbor_pos: chk_neighbor_pos,
                var_neighbor_pos: var_neighbor_pos,
            },
            max_iter: max_iter,
            chk_incoming_msgs: chk_incoming_msgs,
            var_incoming_msgs: var_incoming_msgs,
            prior_llr: prior_arr.iter().map(|&p| prob_to_llr(p)).collect(),
            llr: vec![0.0; n],
            ehat: Array1::zeros((n,)),
            syndrome: Array1::zeros((m,)),
        })
    }

    pub fn get_num_checks(&self) -> usize {
        self.base.m
    }

    pub fn get_num_variables(&self) -> usize {
        self.base.n
    }

    pub fn get_chk_neighbors(&self) -> Vec<Vec<usize>> {
        self.base.chk_neighbors.clone()
    }

    pub fn get_var_neighbors(&self) -> Vec<Vec<usize>> {
        self.base.var_neighbors.clone()
    }

    pub fn get_chk_neighbor_pos(&self) -> Vec<Vec<usize>> {
        self.base.chk_neighbor_pos.clone()
    }

    pub fn get_var_neighbor_pos(&self) -> Vec<Vec<usize>> {
        self.base.var_neighbor_pos.clone()
    }

    pub fn get_pcm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        PyArray2::from_array(py, &self.base.pcm)
    }

    pub fn get_prior<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_array(py, &self.base.prior)
    }

    fn _decode(&mut self) {
        let syndrome_sgn: Vec<f64> = self
            .syndrome
            .iter()
            .map(|&x| if x == 0 { 1.0 } else { -1.0 })
            .collect();

        // Initialize messages from VNs to CNs
        for j in 0..self.base.n {
            let msg = self.prior_llr[j];
            for (k, &i) in self.base.var_neighbors[j].iter().enumerate() {
                self.chk_incoming_msgs[i][self.base.var_neighbor_pos[j][k]] = msg;
            }
        }

        // Main BP iteration loop
        for _iter in 0..self.max_iter {
            // Message processing at CNs
            for i in 0..self.base.m {
                let incoming_msgs = &self.chk_incoming_msgs[i];
                let mut sgnprod = 1.0; // product of signs of the incoming messages
                let mut minabs1 = f64::MAX; // minimum absolute value of the incoming messages
                let mut minabs2 = f64::MAX; // second minimum absolute value of the incoming messages
                let mut minidx = 0; // index of the incoming message with minimum absolute value
                for (k, &val) in incoming_msgs.iter().enumerate() {
                    sgnprod *= val.signum();
                    let val_abs = val.abs();
                    if val_abs < minabs1 {
                        minabs2 = minabs1;
                        minabs1 = val_abs;
                        minidx = k;
                    } else if val_abs < minabs2 {
                        minabs2 = val_abs;
                    }
                }
                for (k, &j) in self.base.chk_neighbors[i].iter().enumerate() {
                    let msg_sgn = sgnprod * incoming_msgs[k].signum() * syndrome_sgn[i];
                    let msg_abs = if k == minidx { minabs2 } else { minabs1 };
                    self.var_incoming_msgs[j][self.base.chk_neighbor_pos[i][k]] = msg_sgn * msg_abs;
                }
            }

            // Message processing at VNs
            for j in 0..self.base.n {
                let incoming_msgs = &self.var_incoming_msgs[j];
                let marginal = self.prior_llr[j] + incoming_msgs.iter().sum::<f64>();
                for (k, &i) in self.base.var_neighbors[j].iter().enumerate() {
                    self.chk_incoming_msgs[i][self.base.var_neighbor_pos[j][k]] =
                        marginal - incoming_msgs[k];
                }
                self.llr[j] = marginal;
            }

            // Hard decision
            for j in 0..self.base.n {
                self.ehat[j] = if self.llr[j] < 0.0 { 1 } else { 0 };
            }

            // Check if the syndrome is satisfied
            let mut satisfied: bool = true;
            for i in 0..self.base.m {
                if self.base.pcm.row(i).dot(&self.ehat) % 2 != self.syndrome[i] {
                    satisfied = false;
                    break;
                }
            }
            if satisfied {
                // early stopping
                break;
            }
        }
    }

    pub fn decode<'py>(
        &mut self,
        syndrome: PyReadonlyArray1<'_, u8>,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<u8>> {
        self.syndrome = syndrome.as_array().to_owned();
        self._decode();
        PyArray1::from_array(py, &self.ehat)
    }

    pub fn decode_batch<'py>(
        &mut self,
        syndrome_batch: PyReadonlyArray2<'_, u8>,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<u8>> {
        let syndrome_batch_arr = syndrome_batch.as_array();
        let batch_size: usize = syndrome_batch_arr.shape()[0];
        let mut ehat_batch: Array2<u8> = Array2::zeros((batch_size, self.base.n));

        for i in 0..batch_size {
            self.syndrome.assign(&syndrome_batch_arr.row(i));
            self._decode();
            ehat_batch.row_mut(i).assign(&self.ehat);
        }

        PyArray2::from_owned_array(py, ehat_batch)
    }
}

#[pymodule]
fn qecdec(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BPDecoder>()?;
    Ok(())
}
