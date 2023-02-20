use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Region},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error},
};

use super::{
  super::gadget::{Gadget, GadgetConfig, GadgetType},
  non_linearity::NonLinearGadget,
};

type ExpGadgetConfig = GadgetConfig;

const NUM_COLS_PER_OP: usize = 2;

// IMPORTANT: this return exp(x) * SF
pub struct ExpChip<F: FieldExt> {
  config: Rc<ExpGadgetConfig>,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> ExpChip<F> {
  pub fn construct(config: Rc<ExpGadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    <ExpChip<F> as NonLinearGadget<F>>::configure(meta, gadget_config, GadgetType::Exp)
  }
}

impl<F: FieldExt> NonLinearGadget<F> for ExpChip<F> {
  fn generate_map(scale_factor: u64, min_val: i64, max_val: i64) -> HashMap<i64, i64> {
    let range = max_val - min_val;

    let mut map = HashMap::new();
    for i in 0..range {
      let shifted = i + min_val;
      let x = (shifted as f64) / (scale_factor as f64);
      let exp = x.exp();
      let exp = (exp * ((scale_factor * scale_factor) as f64)).round() as i64;
      map.insert(i as i64, exp);
    }
    map
  }

  fn get_map(&self) -> &HashMap<i64, i64> {
    &self.config.maps.get(&GadgetType::Exp).unwrap()[0]
  }

  fn get_selector(&self) -> halo2_proofs::plonk::Selector {
    self.config.selectors.get(&GadgetType::Exp).unwrap()[0]
  }
}

impl<F: FieldExt> Gadget<F> for ExpChip<F> {
  fn name(&self) -> String {
    "Exp".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    NUM_COLS_PER_OP
  }

  fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() / NUM_COLS_PER_OP
  }

  fn num_outputs_per_row(&self) -> usize {
    self.config.columns.len() / NUM_COLS_PER_OP
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    NonLinearGadget::op_row_region(
      self,
      region,
      row_offset,
      vec_inputs,
      single_inputs,
      self.config.clone(),
    )
  }

  fn forward(
    &self,
    layouter: impl halo2_proofs::circuit::Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    NonLinearGadget::forward(self, layouter, vec_inputs, single_inputs)
  }
}
