use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};
use rounded_div::RoundedDiv;

use super::gadget::{convert_to_u128, Gadget, GadgetConfig, GadgetType};

pub struct VarDivRoundBigChip<F: FieldExt> {
  config: Rc<GadgetConfig>,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> VarDivRoundBigChip<F> {
  pub fn construct(config: Rc<GadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn num_cols_per_op() -> usize {
    5
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let columns = gadget_config.columns;
    let selector = meta.complex_selector();
    let two = Expression::Constant(F::from(2));
    let range = gadget_config.max_val - gadget_config.min_val;
    let range = Expression::Constant(F::from(range as u64));

    let tables = gadget_config.tables;
    let lookup = if tables.contains_key(&GadgetType::BiasDivRoundRelu6) {
      tables.get(&GadgetType::BiasDivRoundRelu6).unwrap()[0]
    } else {
      panic!()
    };

    // a | c | r | (2 b - r)_1 | (2 b - r)_0 | ... | b
    // a / b = c
    meta.create_gate("var_div_arithm", |meta| {
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
        let br1 = meta.query_advice(columns[offset + 3], Rotation::cur());
        let br0 = meta.query_advice(columns[offset + 4], Rotation::cur());
        let lhs = b.clone() * two.clone() - r.clone();
        let rhs = br1 * range.clone() + br0;
        constraints.push(s.clone() * (lhs - rhs));
      }

      constraints
    });

    // For var div big, we assume that a, b > 0 and are outputs of the previous layer
    // c and r must be constrained
    // However, constraining r properly also constrains c
    for i in 0..(columns.len() - 1) / Self::num_cols_per_op() {
      let offset = i * Self::num_cols_per_op();

      // (2 * b - r)_{1, 0} \in [0, 2^N)
      meta.lookup("var div big br1", |meta| {
        let s = meta.query_selector(selector);
        let br1 = meta.query_advice(columns[offset + 3], Rotation::cur());
        vec![(s * br1, lookup)]
      });
      meta.lookup("var div big br0", |meta| {
        let s = meta.query_selector(selector);
        let br0 = meta.query_advice(columns[offset + 4], Rotation::cur());
        vec![(s * br0, lookup)]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::VarDivRoundBig, vec![selector]);

    GadgetConfig {
      columns,
      tables,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: FieldExt> Gadget<F> for VarDivRoundBigChip<F> {
  fn name(&self) -> String {
    "VarDivRoundChip".to_string()
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
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let a_vec = &vec_inputs[0];
    // let zero = single_inputs[0].clone();
    let b = &single_inputs[1];

    let div_outp_min_val_i64 = self.config.div_outp_min_val;
    let div_inp_min_val_pos_i64 = -self.config.shift_min_val;
    let range = self.config.max_val - self.config.min_val;

    if self.config.use_selectors {
      let selector = self
        .config
        .selectors
        .get(&GadgetType::VarDivRoundBig)
        .unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    b.copy_advice(
      || "",
      region,
      self.config.columns[self.config.columns.len() - 1],
      row_offset,
    )?;

    let mut div_out = vec![];
    for (i, a) in a_vec.iter().enumerate() {
      let offset = i * self.num_cols_per_op();
      a.copy_advice(|| "", region, self.config.columns[offset], row_offset)
        .unwrap();

      let div_mod = a.value().zip(b.value()).map(|(a, b)| {
        let b = convert_to_u128(b);
        // Needs to be divisible by b
        let div_inp_min_val_pos_i64 = div_inp_min_val_pos_i64 / (b as i64) * (b as i64);
        let div_inp_min_val_pos = F::from(div_inp_min_val_pos_i64 as u64);

        let a_pos = *a + div_inp_min_val_pos;
        let a = convert_to_u128(&a_pos);
        // c = (2 * a + b) / (2 * b)
        let c_pos = a.rounded_div(b);
        let c = (c_pos as i128 - (div_inp_min_val_pos_i64 as u128 / b) as i128) as i64;

        // r = (2 * a + b) % (2 * b)
        let rem_floor = (a as i128) - (c_pos * b) as i128;
        let r = 2 * rem_floor + (b as i128);
        let r = r as i64;
        (c, r)
      });

      let br_split = div_mod.zip(b.value()).map(|((_, r), b)| {
        let b = convert_to_u128(b) as i64;
        let val = 2 * b - r;
        let p1 = val / range;
        let p0 = val % range;
        // val = p1 * max_val + p0
        (p1, p0)
      });

      let div_cell = region.assign_advice(
        || "",
        self.config.columns[offset + 1],
        row_offset,
        || {
          div_mod.map(|(c, _)| {
            let offset = F::from(-div_outp_min_val_i64 as u64);
            let c = F::from((c - div_outp_min_val_i64) as u64);
            c - offset
          })
        },
      )?;
      let _mod_cell = region.assign_advice(
        || "",
        self.config.columns[offset + 2],
        row_offset,
        || div_mod.map(|(_, r)| F::from(r as u64)),
      )?;
      // Assign 2 * b - r to the next 2 columns
      let _br_split_cell_1 = region.assign_advice(
        || "",
        self.config.columns[offset + 3],
        row_offset,
        || br_split.map(|(p1, _)| F::from(p1 as u64)),
      )?;
      let _br_split_cell_2 = region.assign_advice(
        || "",
        self.config.columns[offset + 4],
        row_offset,
        || br_split.map(|(_, p0)| F::from(p0 as u64)),
      )?;

      div_out.push(div_cell);
    }

    Ok(div_out)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let mut inps = vec_inputs[0].clone();
    let initial_len = inps.len();

    // Needed to pad
    let default = &single_inputs[0];
    while inps.len() % self.num_inputs_per_row() != 0 {
      inps.push(&default);
    }

    let res = self.op_aligned_rows(
      layouter.namespace(|| "var_div_big"),
      &vec![inps],
      single_inputs,
    )?;
    Ok(res[..initial_len].to_vec())
  }
}
