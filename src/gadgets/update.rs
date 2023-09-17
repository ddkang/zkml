use std::marker::PhantomData;

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use crate::gadgets::gadget::{convert_to_u64, GadgetConfig};

use super::gadget::{Gadget, GadgetType};

type UpdateConfig = GadgetConfig;

#[derive(Clone, Debug)]
pub struct UpdateGadgetChip<F: PrimeField> {
  config: UpdateConfig,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> UpdateGadgetChip<F> {
  pub fn construct(config: UpdateConfig) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn num_cols_per_op() -> usize {
    4
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> UpdateConfig {
    let tables = &gadget_config.tables;
    let mod_lookup = tables.get(&GadgetType::InputLookup).unwrap()[0];

    let columns = gadget_config.columns;
    let selector = meta.complex_selector();

    let div_val = gadget_config.scale_factor;
    let eta: u64 = (gadget_config.scale_factor as f64 * gadget_config.eta) as u64;

    meta.create_gate("updater_arith", |meta| {
      let s = meta.query_selector(selector);

      let sf = Expression::Constant(F::from(div_val as u64));
      let eta = Expression::Constant(F::from(eta as u64));

      let mut constraints = vec![];
      for op_idx in 0..columns.len() / Self::num_cols_per_op() {
        let offset = op_idx * Self::num_cols_per_op();
        let w = meta.query_advice(columns[offset], Rotation::cur());
        let dw = meta.query_advice(columns[offset + 1], Rotation::cur());
        let div = meta.query_advice(columns[offset + 2], Rotation::cur());
        let mod_res = meta.query_advice(columns[offset + 3], Rotation::cur());

        let expr = (w * sf.clone() - dw * eta.clone()) - (div * sf.clone() + mod_res);
        constraints.push(s.clone() * expr);
      }
      constraints
    });

    for op_idx in 0..columns.len() / Self::num_cols_per_op() {
      let offset = op_idx * Self::num_cols_per_op();

      // Check that mod is smaller than SF
      meta.lookup("max inp1", |meta| {
        let s = meta.query_selector(selector);
        let mod_res = meta.query_advice(columns[offset + 3], Rotation::cur());

        // Constrains that the modulus \in [0, DIV_VAL)
        vec![(s.clone() * mod_res.clone(), mod_lookup)]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Update, vec![selector]);

    UpdateConfig {
      columns,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: PrimeField + Ord> Gadget<F> for UpdateGadgetChip<F> {
  fn name(&self) -> String {
    "updater chip".to_string()
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

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    _single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let div_val = self.config.scale_factor as i64;
    let div_val_f = F::from(div_val as u64);
    let eta = div_val / 1000;
    let eta = F::from(eta as u64);

    let div_outp_min_val = self.config.div_outp_min_val;
    let div_inp_min_val_pos_i64 = -self.config.shift_min_val;
    let div_inp_min_val_pos = F::from(div_inp_min_val_pos_i64 as u64);

    let columns = &self.config.columns;

    if self.config.use_selectors {
      let selector = self.config.selectors.get(&GadgetType::Update).unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    let w = &vec_inputs[0];
    let dw = &vec_inputs[1];

    let mut output_cells = vec![];

    for i in 0..w.len() {
      let offset = i * self.num_cols_per_op();
      let _w_cell = w[i].0.copy_advice(|| "", region, columns[offset + 0], row_offset)?;
      let _dw_cell = dw[i].0.copy_advice(|| "", region, columns[offset + 1], row_offset)?;

      let w_val = w[i].1;
      let dw_val = dw[i].1;
      let out_scaled = w_val * div_val_f - dw_val * eta;

      let div_mod =  {
        let x_pos = out_scaled + div_inp_min_val_pos;
        let x_pos = if x_pos > F::ZERO {
          x_pos
        } else {
          x_pos + div_val_f
        };
        let inp = convert_to_u64(&x_pos);

        let div_res = inp as i64 / div_val - (div_inp_min_val_pos_i64 as i64 / div_val);
        let mod_res = inp as i64 % div_val;
        (div_res, mod_res)
      };

      let div_res_cell = region
        .assign_advice(
          || "div_res",
          self.config.columns[offset + 2],
          row_offset,
          || {
              Value::known(
                F::from((div_mod.0 - div_outp_min_val as i64) as u64) - F::from(-div_outp_min_val as u64)
              )
          },
        )
        .unwrap();

      let _mod_res_cell = region
        .assign_advice(
          || "mod_res",
          self.config.columns[offset + 3],
          row_offset,
          || Value::known(F::from(div_mod.1 as u64)),
        )
        .unwrap();

      output_cells.push((
        div_res_cell,
        F::from((div_mod.0 - div_outp_min_val as i64) as u64) - F::from(-div_outp_min_val as u64)
      ));
    }
    Ok(output_cells)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let zero = &single_inputs[0];
    let mut w = vec_inputs[0].clone();
    let mut dw = vec_inputs[1].clone();

    let initial_len = w.len();
    while !w.len() % self.num_cols_per_op() == 0 {
      w.push(*zero);
    }
    while !dw.len() % self.num_cols_per_op() == 0 {
      dw.push(*zero);
    }

    let res = self.op_aligned_rows(
      layouter.namespace(|| format!("forward row {}", self.name())),
      &vec![w, dw],
      single_inputs,
    )?;

    Ok(res[0..initial_len].to_vec())
  }
}
