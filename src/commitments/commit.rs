use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};

use crate::{gadgets::gadget::GadgetConfig, layers::layer::CellRc};

pub trait Commit<F: PrimeField> {
  fn commit(
    &self,
    layouter: impl Layouter<F>,
    gadget_config: Rc<GadgetConfig>,
    constants: &HashMap<i64, CellRc<F>>,
    values: &Vec<CellRc<F>>,
    blinding: CellRc<F>,
  ) -> Result<Vec<CellRc<F>>, Error>;
}
