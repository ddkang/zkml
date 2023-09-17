use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error},
  poly::Rotation,
};

use super::gadget::{Gadget, GadgetConfig, GadgetType};

pub struct SquareGadgetChip<F: PrimeField> {
  config: Rc<GadgetConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> SquareGadgetChip<F> {
  pub fn construct(config: Rc<GadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  // TODO: it would be more efficient to do the division here directly
  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.selector();
    let columns = gadget_config.columns;

    meta.create_gate("square gate", |meta| {
      let s = meta.query_selector(selector);
      let gate_inp = meta.query_advice(columns[0], Rotation::cur());
      let gate_output = meta.query_advice(columns[1], Rotation::cur());

      let res = gate_inp.clone() * gate_inp;

      vec![s * (res - gate_output)]
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Square, vec![selector]);

    GadgetConfig {
      columns,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for SquareGadgetChip<F> {
  fn name(&self) -> String {
    "SquareChip".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    2
  }

  fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() / self.num_cols_per_op()
  }

  fn num_outputs_per_row(&self) -> usize {
    self.num_inputs_per_row()
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    _single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    assert_eq!(vec_inputs.len(), 1);

    if self.config.use_selectors {
      let selector = self.config.selectors.get(&GadgetType::Square).unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    let inps = &vec_inputs[0];
    let mut outp = vec![];
    for (i, inp) in inps.iter().enumerate() {
      let offset = i * self.num_cols_per_op();
      inp.0.copy_advice(|| "", region, self.config.columns[offset], row_offset)?;
      let outp_val = inp.1 * inp.1;
      let outp_cell = region.assign_advice(
        || "square output",
        self.config.columns[offset + 1],
        row_offset,
        || Value::known(outp_val),
      )?;
      outp.push((outp_cell, outp_val));
    }

    Ok(outp)
  }

  fn forward(
    &self,
    mut layouter: impl halo2_proofs::circuit::Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let zero = &single_inputs[0];

    let mut inp = vec_inputs[0].clone();
    let initial_len = inp.len();
    while inp.len() % self.num_inputs_per_row() != 0 {
      inp.push(*zero);
    }

    let vec_inputs = vec![inp];
    let res = self.op_aligned_rows(
      layouter.namespace(|| format!("forward row {}", self.name())),
      &vec_inputs,
      single_inputs,
    )?;
    Ok(res[0..initial_len].to_vec())
  }
}
