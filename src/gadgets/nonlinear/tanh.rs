use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error},
};

use super::{
  super::gadget::{Gadget, GadgetConfig, GadgetType},
  non_linearity::NonLinearGadget,
};

pub struct TanhGadgetChip<F: FieldExt> {
  config: Rc<GadgetConfig>,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> TanhGadgetChip<F> {
  pub fn construct(config: Rc<GadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    <TanhGadgetChip<F> as NonLinearGadget<F>>::configure(meta, gadget_config, GadgetType::Tanh)
  }
}

impl<F: FieldExt> NonLinearGadget<F> for TanhGadgetChip<F> {
  fn generate_map(scale_factor: u64, min_val: i64, num_rows: i64) -> HashMap<i64, i64> {
    let scale_factor = scale_factor as f64;

    let mut map = HashMap::new();
    for i in 0..num_rows {
      let shifted = i + min_val;
      let x = (shifted as f64) / scale_factor;
      let y = x.tanh();
      let y = (y * scale_factor).round() as i64;
      map.insert(i as i64, y);
    }

    map
  }

  fn get_map(&self) -> &HashMap<i64, i64> {
    &self.config.maps.get(&GadgetType::Tanh).unwrap()[0]
  }

  fn get_selector(&self) -> halo2_proofs::plonk::Selector {
    self.config.selectors.get(&GadgetType::Tanh).unwrap()[0]
  }
}

impl<F: FieldExt> Gadget<F> for TanhGadgetChip<F> {
  fn name(&self) -> String {
    "TanhGadgetChip".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    <TanhGadgetChip<F> as NonLinearGadget<F>>::num_cols_per_op()
  }

  fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() / self.num_cols_per_op()
  }

  fn num_outputs_per_row(&self) -> usize {
    self.config.columns.len() / self.num_cols_per_op()
  }

  fn load_lookups(&self, layouter: impl Layouter<F>) -> Result<(), Error> {
    NonLinearGadget::load_lookups(self, layouter, self.config.clone(), GadgetType::Tanh)?;
    Ok(())
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
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
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    NonLinearGadget::forward(self, layouter, vec_inputs, single_inputs)
  }
}
