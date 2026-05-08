mod bp;
mod bp_base;
mod dmembp;
mod dmemoffsetbp;
mod ens_serial_bp;
mod serial_bp;
mod serial_bp_kernel;
mod uf;
use pyo3::prelude::*;

#[pymodule]
fn qecdec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bp::register(m)?;
    dmembp::register(m)?;
    dmemoffsetbp::register(m)?;
    ens_serial_bp::register(m)?;
    serial_bp::register(m)?;
    uf::register(m)?;
    Ok(())
}
