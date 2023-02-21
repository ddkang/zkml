use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};
use rounded_div::RoundedDiv;

use super::gadget::{convert_to_u128, Gadget, GadgetConfig, GadgetType, USE_SELECTORS};

type VarDivRoundConfig = GadgetConfig;

const NUM_COLS_PER_OP: usize = 3;

pub struct VarDivRoundChip<F: FieldExt> {
  config: Rc<VarDivRoundConfig>,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> VarDivRoundChip<F> {
  pub fn construct(config: Rc<VarDivRoundConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let columns = gadget_config.columns;
    let selector = meta.complex_selector();
    let two = Expression::Constant(F::from(2));

    let tables = gadget_config.tables;
    let lookup = if tables.contains_key(&GadgetType::BiasDivRoundRelu6) {
      tables.get(&GadgetType::BiasDivRoundRelu6).unwrap()[0]
    } else {
      meta.lookup_table_column()
    };

    // a | c | r | ... | b
    // (2 * a + b) = (2 * b) * c + r
    // b - r \in [0, 2^N) <-- forces b > r
    meta.create_gate("var_div_arithm", |meta| {
      let s = meta.query_selector(selector);
      let mut constraints = vec![];

      let b = meta.query_advice(columns[columns.len() - 1], Rotation::cur());
      for i in 0..(columns.len() - 1) / NUM_COLS_PER_OP {
        let offset = i * NUM_COLS_PER_OP;
        let a = meta.query_advice(columns[offset], Rotation::cur());
        let c = meta.query_advice(columns[offset + 1], Rotation::cur());
        let r = meta.query_advice(columns[offset + 2], Rotation::cur());

        let lhs = a.clone() * two.clone() + b.clone();
        let rhs = b.clone() * two.clone() * c + r;
        constraints.push(s.clone() * (lhs - rhs));
      }

      constraints
    });

    for i in 0..(columns.len() - 1) / NUM_COLS_PER_OP {
      let offset = i * NUM_COLS_PER_OP;
      // r \in [0, 2^N)
      meta.lookup("var div range checks r", |meta| {
        let s = meta.query_selector(selector);
        let r = meta.query_advice(columns[offset + 2], Rotation::cur());

        vec![(s.clone() * r, lookup)]
      });

      // 2 * b - r \in [0, 2^N)
      meta.lookup("var div range checks 2b-r", |meta| {
        let s = meta.query_selector(selector);
        let b = meta.query_advice(columns[columns.len() - 1], Rotation::cur());
        let r = meta.query_advice(columns[offset + 2], Rotation::cur());

        vec![(s.clone() * (two.clone() * b - r), lookup)]
      });
    }
    // b \in [0, 2^N)
    meta.lookup("var div range checks b", |meta| {
      let s = meta.query_selector(selector);
      let b = meta.query_advice(columns[columns.len() - 1], Rotation::cur());

      vec![(s.clone() * b, lookup)]
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::VarDivRound, vec![selector]);

    // FIXME: tables needs to be updated

    GadgetConfig {
      columns,
      tables,
      selectors,
      ..gadget_config
    }
  }
}

// FIXME: needs load lookups
impl<F: FieldExt> Gadget<F> for VarDivRoundChip<F> {
  fn name(&self) -> String {
    "VarDivRoundChip".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    NUM_COLS_PER_OP
  }

  fn num_inputs_per_row(&self) -> usize {
    (self.config.columns.len() - 1) / NUM_COLS_PER_OP
  }

  fn num_outputs_per_row(&self) -> usize {
    self.num_inputs_per_row()
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let a_vec = &vec_inputs[0];
    // let zero = single_inputs[0].clone();
    let b = &single_inputs[1];

    let div_outp_min_val_i64 = self.config.div_outp_min_val;
    let div_inp_min_val_pos_i64 = -self.config.shift_min_val;

    let selector = self.config.selectors.get(&GadgetType::VarDivRound).unwrap()[0];

    if USE_SELECTORS {
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
      let offset = i * NUM_COLS_PER_OP;
      a.copy_advice(|| "", region, self.config.columns[offset], row_offset)?;

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
      div_out.push(div_cell);
    }

    Ok(div_out)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let mut inps = vec_inputs[0].clone();
    let initial_len = inps.len();

    // Needed to pad: bias - bias = 0
    let default = &single_inputs[0];
    while inps.len() % self.num_inputs_per_row() != 0 {
      inps.push(&default);
    }

    let res = self.op_aligned_rows(layouter.namespace(|| "var_div"), &vec![inps], single_inputs)?;
    Ok(res[..initial_len].to_vec())
  }
}
