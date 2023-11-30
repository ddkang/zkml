use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error},
};

use super::gadget::{Gadget, GadgetConfig, GadgetType};

pub struct InputLookupChip<F: PrimeField> {
  config: Rc<GadgetConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> InputLookupChip<F> {
  pub fn construct(config: Rc<GadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let lookup = meta.lookup_table_column();
    let mut tables = gadget_config.tables;
    tables.insert(GadgetType::InputLookup, vec![lookup]);

    GadgetConfig {
      tables,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for InputLookupChip<F> {
  fn load_lookups(&self, mut layouter: impl Layouter<F>) -> Result<(), Error> {
    let lookup = self.config.tables[&GadgetType::InputLookup][0];

    layouter
      .assign_table(
        || "input lookup",
        |mut table| {
          for i in 0..self.config.num_rows as i64 {
            table
              .assign_cell(
                || "mod lookup",
                lookup,
                i as usize,
                || Value::known(F::from(i as u64)),
              )
              .unwrap();
          }
          Ok(())
        },
      )
      .unwrap();

    Ok(())
  }

  fn name(&self) -> String {
    panic!("InputLookupChip should not be called directly")
  }

  fn num_cols_per_op(&self) -> usize {
    panic!("InputLookupChip should not be called directly")
  }

  fn num_inputs_per_row(&self) -> usize {
    panic!("InputLookupChip should not be called directly")
  }

  fn num_outputs_per_row(&self) -> usize {
    panic!("InputLookupChip should not be called directly")
  }

  fn op_row_region(
    &self,
    _region: &mut Region<F>,
    _row_offset: usize,
    _vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    _single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    panic!("InputLookupChip should not be called directly")
  }
}
