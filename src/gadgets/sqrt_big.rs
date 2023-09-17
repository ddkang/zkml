use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use crate::gadgets::gadget::convert_to_u64;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type SqrtBigConfig = GadgetConfig;

pub struct SqrtBigChip<F: PrimeField> {
  config: Rc<SqrtBigConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> SqrtBigChip<F> {
  pub fn construct(config: Rc<SqrtBigConfig>) -> Self {
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
    let two = Expression::Constant(F::from(2));
    let columns = gadget_config.columns;

    let tables = gadget_config.tables;

    let inp_lookup = tables.get(&GadgetType::InputLookup).unwrap()[0];

    // TODO: prove that these constraints work
    meta.create_gate("sqrt_big arithm", |meta| {
      let s = meta.query_selector(selector);

      let mut constraints = vec![];
      for op_idx in 0..columns.len() / Self::num_cols_per_op() {
        let offset = op_idx * Self::num_cols_per_op();
        let inp = meta.query_advice(columns[offset + 0], Rotation::cur());
        let sqrt = meta.query_advice(columns[offset + 1], Rotation::cur());
        let rem = meta.query_advice(columns[offset + 2], Rotation::cur());

        let lhs = inp.clone();
        let rhs = sqrt.clone() * sqrt.clone() + rem.clone();
        constraints.push(s.clone() * (lhs - rhs));
      }
      constraints
    });

    for op_idx in 0..columns.len() / Self::num_cols_per_op() {
      let offset = op_idx * Self::num_cols_per_op();
      meta.lookup("sqrt_big sqrt lookup", |meta| {
        let s = meta.query_selector(selector);
        let sqrt = meta.query_advice(columns[offset + 1], Rotation::cur());

        vec![(s.clone() * sqrt, inp_lookup)]
      });

      meta.lookup("sqrt_big rem lookup", |meta| {
        let s = meta.query_selector(selector);
        let sqrt = meta.query_advice(columns[offset + 1], Rotation::cur());
        let rem = meta.query_advice(columns[offset + 2], Rotation::cur());

        vec![(s.clone() * (rem + sqrt), inp_lookup)]
      });

      meta.lookup("sqrt_big sqrt - rem lookup", |meta| {
        let s = meta.query_selector(selector);
        let sqrt = meta.query_advice(columns[offset + 1], Rotation::cur());
        let rem = meta.query_advice(columns[offset + 2], Rotation::cur());

        vec![(s.clone() * (two.clone() * sqrt - rem), inp_lookup)]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::SqrtBig, vec![selector]);

    GadgetConfig {
      columns,
      tables,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for SqrtBigChip<F> {
  fn name(&self) -> String {
    "sqrt_big".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    Self::num_cols_per_op()
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
    let inps = &vec_inputs[0];

    if self.config.use_selectors {
      let selector = self.config.selectors.get(&GadgetType::SqrtBig).unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    let mut outp_cells = vec![];
    for (i, inp) in inps.iter().enumerate() {
      let offset = i * self.num_cols_per_op();
      inp.0.copy_advice(
        || "sqrt_big",
        region,
        self.config.columns[offset],
        row_offset,
      )?;

      let outp = {
        let inp_val = convert_to_u64(&inp.1) as i64;
        let fsqrt = (inp_val as f64).sqrt();
        let sqrt = fsqrt.round() as i64;
        let rem = inp_val - sqrt * sqrt;
        (sqrt, rem)
      };

      let sqrt_cell = region.assign_advice(
        || "sqrt_big",
        self.config.columns[offset + 1],
        row_offset,
        || Value::known(F::from(outp.0 as u64)),
      )?;

      let _rem_cell = region.assign_advice(
        || "sqrt_big",
        self.config.columns[offset + 2],
        row_offset,
        || {
          let tmp = {
            let rem_pos = outp.1 + outp.0;
            F::from(rem_pos as u64) - F::from(outp.0 as u64)
          };
          Value::known(tmp)
        },
      )?;
      outp_cells.push((sqrt_cell, F::from(outp.0 as u64)));
    }

    Ok(outp_cells)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let zero = &single_inputs[0];

    let mut inp = vec_inputs[0].clone();
    let inp_len = inp.len();
    while inp.len() % self.num_inputs_per_row() != 0 {
      inp.push(*zero);
    }

    let vec_inputs = vec![inp];
    let outp = self.op_aligned_rows(
      layouter.namespace(|| format!("forward row {}", self.name())),
      &vec_inputs,
      single_inputs,
    )?;

    Ok(outp[0..inp_len].to_vec())
  }
}
