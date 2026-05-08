mod bp_base;
mod bp;
mod dmembp;
mod dmemoffsetbp;
mod serialbp;
mod uf;
use pyo3::prelude::*;

#[pymodule]
fn qecdec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bp::register(m)?;
    dmembp::register(m)?;
    dmemoffsetbp::register(m)?;
    serialbp::register(m)?;
    uf::register(m)?;
    Ok(())
}
