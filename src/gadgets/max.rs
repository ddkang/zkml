use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error},
  poly::Rotation,
};

use crate::gadgets::gadget::convert_to_u64;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

pub struct MaxChip<F: PrimeField> {
  config: Rc<GadgetConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> MaxChip<F> {
  pub fn construct(config: Rc<GadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn num_cols_per_op() -> usize {
    3
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.complex_selector();
    let columns = gadget_config.columns;
    let tables = gadget_config.tables;

    let inp_lookup = tables.get(&GadgetType::InputLookup).unwrap()[0];

    meta.create_gate("max arithmetic", |meta| {
      let s = meta.query_selector(selector);
      let mut constraints = vec![];
      for i in 0..columns.len() / Self::num_cols_per_op() {
        let offset = i * Self::num_cols_per_op();
        let inp1 = meta.query_advice(columns[offset + 0], Rotation::cur());
        let inp2 = meta.query_advice(columns[offset + 1], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 2], Rotation::cur());

        constraints.push(s.clone() * (inp1 - outp.clone()) * (inp2 - outp))
      }
      constraints
    });

    for idx in 0..columns.len() / Self::num_cols_per_op() {
      meta.lookup("max inp1", |meta| {
        let s = meta.query_selector(selector);
        let offset = idx * Self::num_cols_per_op();
        let inp1 = meta.query_advice(columns[offset + 0], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 2], Rotation::cur());

        vec![(s * (outp - inp1), inp_lookup)]
      });
      meta.lookup("max inp2", |meta| {
        let s = meta.query_selector(selector);
        let offset = idx * Self::num_cols_per_op();
        let inp2 = meta.query_advice(columns[offset + 1], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 2], Rotation::cur());

        vec![(s * (outp - inp2), inp_lookup)]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Max, vec![selector]);

    GadgetConfig {
      columns,
      selectors,
      tables,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for MaxChip<F> {
  fn name(&self) -> String {
    "max".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    3
  }

  fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() / self.num_cols_per_op() * 2
  }

  fn num_outputs_per_row(&self) -> usize {
    self.config.columns.len() / self.num_cols_per_op()
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    _single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    assert_eq!(vec_inputs.len(), 1);
    let inp = &vec_inputs[0];

    if self.config.use_selectors {
      let selector = self.config.selectors.get(&GadgetType::Max).unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    let min_val_pos = F::from((-self.config.shift_min_val) as u64);

    let mut outp = vec![];

    let chunks: Vec<&[(&AssignedCell<F, F>, F)]> = inp.chunks(self.num_outputs_per_row()).collect();
    let i1 = chunks[0];
    let i2 = chunks[1];
    for (idx, (inp1, inp2)) in i1.iter().zip(i2.iter()).enumerate() {
      let offset = idx * self.num_cols_per_op();
      inp1.0
        .copy_advice(|| "", region, self.config.columns[offset + 0], row_offset)
        .unwrap();
      inp2.0
        .copy_advice(|| "", region, self.config.columns[offset + 1], row_offset)
        .unwrap();

      let max = {
        let a = convert_to_u64(&(inp1.1 + min_val_pos));
        let b = convert_to_u64(&(inp2.1 + min_val_pos));
        let max = a.max(b);
        let max = F::from(max) - min_val_pos;
        max
      };

      let res = region
        .assign_advice(|| "", self.config.columns[offset + 2], row_offset, || Value::known(max))
        .unwrap();
      outp.push((res, max));
    }

    Ok(outp)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let mut inputs = vec_inputs[0].clone();
    let first = inputs[0];

    while inputs.len() % self.num_inputs_per_row() != 0 {
      inputs.push(first);
    }

    // TODO: pretty sure this is correct but check
    let num_iters = inputs.len().div_ceil(self.num_inputs_per_row()) + self.num_inputs_per_row();

    let mut outputs = self.op_aligned_rows(
      layouter.namespace(|| "max forward"),
      &vec![inputs],
      single_inputs,
    )?;
    for _ in 0..num_iters {
      while outputs.len() % self.num_inputs_per_row() != 0 {
        outputs.push((first.0.clone(), first.1));
      }
      let tmp = outputs.iter().map(|x| (&x.0, x.1)).collect::<Vec<_>>();
      outputs = self.op_aligned_rows(
        layouter.namespace(|| "max forward"),
        &vec![tmp],
        single_inputs,
      )?;
    }

    outputs = vec![outputs.into_iter().next().unwrap()];

    Ok(outputs)
  }
}
