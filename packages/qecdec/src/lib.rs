mod bp;
mod bp_base;
mod bp_like;
mod csr;
mod dmembp;
mod dmembp_core;
mod dmemoffsetbp;
mod ens_serial_bp;
mod multi_relaybp;
mod relaybp;
mod relaybp_core;
mod serial_bp;
mod serial_bp_core;
mod uf;
mod utils;
use pyo3::prelude::*;

#[pymodule]
fn qecdec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bp::register(m)?;
    dmembp::register(m)?;
    dmemoffsetbp::register(m)?;
    ens_serial_bp::register(m)?;
    multi_relaybp::register(m)?;
    relaybp::register(m)?;
    serial_bp::register(m)?;
    uf::register(m)?;
    Ok(())
}
