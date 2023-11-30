use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error},
  poly::Rotation,
};

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type MulPairsConfig = GadgetConfig;

pub struct MulPairsChip<F: PrimeField> {
  config: Rc<MulPairsConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> MulPairsChip<F> {
  pub fn construct(config: Rc<MulPairsConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn num_cols_per_op() -> usize {
    3
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.selector();
    let columns = gadget_config.columns;

    meta.create_gate("mul pair", |meta| {
      let s = meta.query_selector(selector);
      let mut constraints = vec![];
      for i in 0..columns.len() / Self::num_cols_per_op() {
        let offset = i * Self::num_cols_per_op();
        let inp1 = meta.query_advice(columns[offset + 0], Rotation::cur());
        let inp2 = meta.query_advice(columns[offset + 1], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 2], Rotation::cur());

        let res = inp1 * inp2;
        constraints.append(&mut vec![s.clone() * (res - outp)])
      }

      constraints
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::MulPairs, vec![selector]);

    GadgetConfig {
      columns,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for MulPairsChip<F> {
  fn name(&self) -> String {
    "MulPairs".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    Self::num_cols_per_op()
  }

  fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() / self.num_cols_per_op()
  }

  fn num_outputs_per_row(&self) -> usize {
    self.config.columns.len() / self.num_cols_per_op()
  }

  // TODO: This + below is basically copied from add pairs - make arithmetic generic
  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    _single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let inp1 = &vec_inputs[0];
    let inp2 = &vec_inputs[1];
    assert_eq!(inp1.len(), inp2.len());

    let columns = &self.config.columns;

    if self.config.use_selectors {
      let selector = self.config.selectors.get(&GadgetType::MulPairs).unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    let mut outps = vec![];
    for i in 0..inp1.len() {
      let offset = i * self.num_cols_per_op();
      inp1[i].0.copy_advice(|| "", region, columns[offset + 0], row_offset)?;
      inp2[i].0.copy_advice(|| "", region, columns[offset + 1], row_offset)?;
      let outp = inp1[i].1 * inp2[i].1;

      let outpc = region.assign_advice(|| "", columns[offset + 2], row_offset, || Value::known(outp))?;
      outps.push((outpc, outp));
    }
    Ok(outps)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let zero = &single_inputs[0];

    let mut inp1 = vec_inputs[0].clone();
    let mut inp2 = vec_inputs[1].clone();
    let initial_len = inp1.len();
    while inp1.len() % self.num_inputs_per_row() != 0 {
      inp1.push(*zero);
      inp2.push(*zero);
    }

    let vec_inputs = vec![inp1, inp2];

    let res = self.op_aligned_rows(
      layouter.namespace(|| format!("forward row {}", self.name())),
      &vec_inputs,
      single_inputs,
    )?;
    Ok(res[0..initial_len].to_vec())
  }
}
