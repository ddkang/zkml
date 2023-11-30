use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};
use rounded_div::RoundedDiv;

use super::gadget::{convert_to_u128, Gadget, GadgetConfig, GadgetType};

pub struct VarDivRoundBig3Chip<F: PrimeField> {
  config: Rc<GadgetConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> VarDivRoundBig3Chip<F> {
  pub fn construct(config: Rc<GadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn num_cols_per_op() -> usize {
    9
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let columns = gadget_config.columns;
    let selector = meta.complex_selector();
    let two = Expression::Constant(F::from(2));
    let range = Expression::Constant(F::from(gadget_config.num_rows as u64));
    let range_sq = range.clone() * range.clone();

    let tables = gadget_config.tables;
    let lookup = tables.get(&GadgetType::InputLookup).unwrap()[0];

    // a | c | r | (2b - r)_2 | (2 b - r)_1 | (2 b - r)_0 | r_2 | r_1 | r_0 | ... | b
    // a / b = c
    meta.create_gate("var_div_big3_arithm", |meta| {
      let s = meta.query_selector(selector);
      let mut constraints = vec![];

      let b = meta.query_advice(columns[columns.len() - 1], Rotation::cur());
      for i in 0..(columns.len() - 1) / Self::num_cols_per_op() {
        let offset = i * Self::num_cols_per_op();
        // Constrain that (2 * a + b) = (2 * b) * c + r
        let a = meta.query_advice(columns[offset], Rotation::cur());
        let c = meta.query_advice(columns[offset + 1], Rotation::cur());
        let r = meta.query_advice(columns[offset + 2], Rotation::cur());

        let lhs = a.clone() * two.clone() + b.clone();
        let rhs = b.clone() * two.clone() * c + r.clone();
        constraints.push(s.clone() * (lhs - rhs));

        // Constrain that (2 * b - r) = br1 * max_val + br0
        let br2 = meta.query_advice(columns[offset + 3], Rotation::cur());
        let br1 = meta.query_advice(columns[offset + 4], Rotation::cur());
        let br0 = meta.query_advice(columns[offset + 5], Rotation::cur());
        let lhs = b.clone() * two.clone() - r.clone();
        let rhs = br2 * range_sq.clone() + br1 * range.clone() + br0;
        constraints.push(s.clone() * (lhs - rhs));

        // Constrains that r = r1 * max_val + r0
        let r2 = meta.query_advice(columns[offset + 6], Rotation::cur());
        let r1 = meta.query_advice(columns[offset + 7], Rotation::cur());
        let r0 = meta.query_advice(columns[offset + 8], Rotation::cur());
        let lhs = r.clone();
        let rhs = r2 * range_sq.clone() + r1 * range.clone() + r0;
        constraints.push(s.clone() * (lhs - rhs));
      }

      constraints
    });

    // For var div big, we assume that a, b > 0 and are outputs of the previous layer
    // r must be constrained to be in [0, b)
    for i in 0..(columns.len() - 1) / Self::num_cols_per_op() {
      let offset = i * Self::num_cols_per_op();

      // (2 * b - r)_{1, 0} \in [0, 2^N)
      meta.lookup("var div big br2", |meta| {
        let s = meta.query_selector(selector);
        let br2 = meta.query_advice(columns[offset + 3], Rotation::cur());
        vec![(s * br2, lookup)]
      });
      meta.lookup("var div big br1", |meta| {
        let s = meta.query_selector(selector);
        let br1 = meta.query_advice(columns[offset + 4], Rotation::cur());
        vec![(s * br1, lookup)]
      });
      meta.lookup("var div big br0", |meta| {
        let s = meta.query_selector(selector);
        let br0 = meta.query_advice(columns[offset + 5], Rotation::cur());
        vec![(s * br0, lookup)]
      });
      // r_{1, 0} \in [0, 2^N)
      meta.lookup("var div big r2", |meta| {
        let s = meta.query_selector(selector);
        let r2 = meta.query_advice(columns[offset + 6], Rotation::cur());
        vec![(s * r2, lookup)]
      });
      meta.lookup("var div big r1", |meta| {
        let s = meta.query_selector(selector);
        let r1 = meta.query_advice(columns[offset + 7], Rotation::cur());
        vec![(s * r1, lookup)]
      });
      meta.lookup("var div big r0", |meta| {
        let s = meta.query_selector(selector);
        let r0 = meta.query_advice(columns[offset + 8], Rotation::cur());
        vec![(s * r0, lookup)]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::VarDivRoundBig3, vec![selector]);

    GadgetConfig {
      columns,
      tables,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for VarDivRoundBig3Chip<F> {
  fn name(&self) -> String {
    "VarDivBig3RoundChip".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    Self::num_cols_per_op()
  }

  fn num_inputs_per_row(&self) -> usize {
    (self.config.columns.len() - 1) / self.num_cols_per_op()
  }

  fn num_outputs_per_row(&self) -> usize {
    self.num_inputs_per_row()
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let a_vec = &vec_inputs[0];
    // let zero = single_inputs[0].clone();
    let b = &single_inputs[1];

    let c_shift_base = (-(1_i64 << 62)) as i128;
    let num_rows = self.config.num_rows as i128;

    if self.config.use_selectors {
      let selector = self
        .config
        .selectors
        .get(&GadgetType::VarDivRoundBig3)
        .unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    b.0.copy_advice(
      || "",
      region,
      self.config.columns[self.config.columns.len() - 1],
      row_offset,
    )?;

    let mut div_out = vec![];
    for (i, a) in a_vec.iter().enumerate() {
      let offset = i * self.num_cols_per_op();
      a.0.copy_advice(|| "", region, self.config.columns[offset], row_offset)
        .unwrap();

      let div_mod = {
        let b = convert_to_u128(&b.1);
        let c_shift = (-c_shift_base) as u128 / b * b;
        let div_inp_min_val_pos = F::from(c_shift as u64);

        let a_pos = a.1 + div_inp_min_val_pos;
        let a = convert_to_u128(&a_pos);
        // c = (2 * a + b) / (2 * b)
        let c_pos = a.rounded_div(b);
        let c = c_pos as i128 - (c_shift / b) as i128;

        // r = (2 * a + b) % (2 * b)
        let rem_floor = (a as i128) - (c_pos * b) as i128;
        let r = 2 * rem_floor + (b as i128);
        (c, r)
      };

      let br_split = {
        let b = convert_to_u128(&b.1) as i128;
        let val = 2 * b - div_mod.1;
        let p2 = val / (num_rows * num_rows);
        let p1 = (val % (num_rows * num_rows)) / num_rows;
        let p0 = val % num_rows;
        // val = p2 * max_val^2 + p1 * max_val + p0
        (p2, p1, p0)
      };

      let r_split = {
        let p2 = div_mod.1 / (num_rows * num_rows);
        let p1 = (div_mod.1 % (num_rows * num_rows)) / num_rows;
        let p0 = div_mod.1 % num_rows;
        // val = p1 * max_val + p0
        (p2, p1, p0)
      };

      let div_val = {
        let offset = F::from(-c_shift_base as u64);
        let c = F::from((div_mod.0 - c_shift_base) as u64);
        c - offset
      };
      let div_cell = region.assign_advice(
        || "",
        self.config.columns[offset + 1],
        row_offset,
        || Value::known(div_val),
      )?;

      let _mod_cell = region.assign_advice(
        || "",
        self.config.columns[offset + 2],
        row_offset,
        || Value::known(F::from(div_mod.1 as u64)),
      )?;
      // Assign 2 * b - r to the next 3 columns
      let _br_split_cell_2 = region.assign_advice(
        || "",
        self.config.columns[offset + 3],
        row_offset,
        || Value::known(F::from(br_split.0 as u64)),
      )?;
      let _br_split_cell_1 = region.assign_advice(
        || "",
        self.config.columns[offset + 4],
        row_offset,
        || Value::known(F::from(br_split.1 as u64)),
      )?;
      let _br_split_cell_0 = region.assign_advice(
        || "",
        self.config.columns[offset + 5],
        row_offset,
        || Value::known(F::from(br_split.2 as u64)),
      )?;
      // Assign r to the next 3 columns
      let _r_split_cell_2 = region.assign_advice(
        || "",
        self.config.columns[offset + 6],
        row_offset,
        || Value::known(F::from(r_split.0 as u64)),
      )?;
      let _r_split_cell_1 = region.assign_advice(
        || "",
        self.config.columns[offset + 7],
        row_offset,
        || Value::known(F::from(r_split.1 as u64)),
      )?;
      let _r_split_cell_0 = region.assign_advice(
        || "",
        self.config.columns[offset + 8],
        row_offset,
        || Value::known(F::from(r_split.2 as u64)),
      )?;

      div_out.push((div_cell, div_val));
    }

    Ok(div_out)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let mut inps = vec_inputs[0].clone();
    let initial_len = inps.len();

    // Needed to pad
    let default = &single_inputs[0];
    while inps.len() % self.num_inputs_per_row() != 0 {
      inps.push(*default);
    }

    let res = self.op_aligned_rows(
      layouter.namespace(|| "var_div_big3"),
      &vec![inps],
      single_inputs,
    )?;
    Ok(res[..initial_len].to_vec())
  }
}
