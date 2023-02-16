use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error},
  poly::Rotation,
};

use crate::gadgets::gadget::USE_SELECTORS;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type AddPairsConfig = GadgetConfig;

const NUM_COLS_PER_OP: usize = 3;

pub struct AddPairsChip<F: FieldExt> {
  config: Rc<AddPairsConfig>,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> AddPairsChip<F> {
  pub fn construct(config: Rc<AddPairsConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.selector();
    let columns = gadget_config.columns;

    meta.create_gate("add pair", |meta| {
      let s = meta.query_selector(selector);
      let mut constraints = vec![];
      for i in 0..columns.len() / NUM_COLS_PER_OP {
        let offset = i * NUM_COLS_PER_OP;
        let inp1 = meta.query_advice(columns[offset + 0], Rotation::cur());
        let inp2 = meta.query_advice(columns[offset + 1], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 2], Rotation::cur());

        let res = inp1 + inp2;
        constraints.append(&mut vec![s.clone() * (res - outp)])
      }

      constraints
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::AddPairs, vec![selector]);

    GadgetConfig {
      columns,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: FieldExt> Gadget<F> for AddPairsChip<F> {
  fn name(&self) -> String {
    "add pairs chip".to_string()
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
    _single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let inp1 = &vec_inputs[0];
    let inp2 = &vec_inputs[1];
    assert_eq!(inp1.len(), inp2.len());

    let selector = self.config.selectors.get(&GadgetType::AddPairs).unwrap()[0];
    let columns = &self.config.columns;

    if USE_SELECTORS {
      selector.enable(region, row_offset)?;
    }

    let mut outps = vec![];
    for i in 0..inp1.len() {
      let offset = i * NUM_COLS_PER_OP;
      let inp1 = inp1[i].copy_advice(|| "", region, columns[offset + 0], row_offset)?;
      let inp2 = inp2[i].copy_advice(|| "", region, columns[offset + 1], row_offset)?;
      let outp = inp1.value().map(|x: &F| x.to_owned()) + inp2.value().map(|x: &F| x.to_owned());

      let outp = region.assign_advice(|| "", columns[offset + 2], row_offset, || outp)?;
      outps.push(outp);
    }
    Ok(outps)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let zero = &single_inputs[0];

    let mut inp1 = vec_inputs[0].clone();
    let mut inp2 = vec_inputs[1].clone();
    let initial_len = inp1.len();
    while inp1.len() % self.num_cols_per_op() != 0 {
      inp1.push(zero);
      inp2.push(zero);
    }

    let vec_inputs = vec![inp1, inp2];

    let res = self.op_aligned_rows(
      layouter.namespace(|| format!("forward row {}", self.name())),
      &vec_inputs,
      &single_inputs,
    )?;
    Ok(res[0..initial_len].to_vec())
  }
}
