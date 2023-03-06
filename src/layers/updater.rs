use std::marker::PhantomData;

use halo2_proofs::{
  arithmetic::FieldExt,
  circuit::{AssignedCell, Layouter},
  plonk::{Advice, Column, ConstraintSystem, Error, Expression, Selector, TableColumn},
  poly::Rotation,
};

use crate::chips::{DIV_INP_MIN_VAL_POS, ETA_TOP};
use crate::gadgets::{bias_div_round_relu6::SHIFT_MIN_VAL};

use super::{convert_to_u64, DIV_OUTP_MIN_VAL, DIV_VAL};

pub const NUM_COLS_PER_OP: usize = 4;

#[derive(Clone, Debug)]
pub struct UpdaterConfig {
  pub shared_columns: Vec<Column<Advice>>,
  pub mod_lookup: TableColumn,
  pub selector: Selector,
}

#[derive(Clone, Debug)]
pub struct UpdaterChip<F: FieldExt> {
  config: UpdaterConfig,
  _marker: PhantomData<F>,
}

// Contains code that updates the weights of a feedforward convolution layer
impl<F: FieldExt> UpdaterChip<F> {
  pub fn construct(config: UpdaterConfig) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(
    meta: &mut ConstraintSystem<F>,
    shared_columns: Vec<Column<Advice>>,
    mod_lookup: TableColumn,
  ) -> UpdaterConfig {
    let selector = meta.complex_selector();

    let div_val = DIV_VAL.load(std::sync::atomic::Ordering::Relaxed);
    let eta = ETA_TOP.load(std::sync::atomic::Ordering::Relaxed);

    meta.create_gate("updater_arith", |meta| {
      let s = meta.query_selector(selector);
      let sf = Expression::Constant(F::from(div_val as u64));
      let eta = Expression::Constant(F::from(eta as u64));

      let mut constraints = vec![];
      for op_idx in 0..shared_columns.len() / NUM_COLS_PER_OP {
        let offset = op_idx * NUM_COLS_PER_OP;
        let w = meta.query_advice(shared_columns[offset], Rotation::cur());
        let dw = meta.query_advice(shared_columns[offset + 1], Rotation::cur());
        let div = meta.query_advice(shared_columns[offset + 2], Rotation::cur());
        let mod_res = meta.query_advice(shared_columns[offset + 3], Rotation::cur());

        let expr = (w * sf.clone() - dw * eta.clone()) - (div * sf.clone() + mod_res);
        constraints.push(s.clone() * expr);
      }
      constraints
    });

    for op_idx in 0..shared_columns.len() / NUM_COLS_PER_OP {
      let offset = op_idx * NUM_COLS_PER_OP;
      meta.lookup("updater", |meta| {
        let s = meta.query_selector(selector);
        let mod_res = meta.query_advice(shared_columns[offset + 3], Rotation::cur());

        // Constrains that the modulus \in [0, DIV_VAL)
        vec![(s.clone() * mod_res.clone(), mod_lookup)]
      });
    }

    UpdaterConfig {
      shared_columns,
      mod_lookup,
      selector,
    }
  }

  pub fn ops_per_row(&self) -> usize {
    self.config.shared_columns.len() / NUM_COLS_PER_OP
  }

  pub fn update_row(
    &self,
    mut layouter: impl Layouter<F>,
    w: &Vec<AssignedCell<F, F>>,
    dw: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert!(w.len() == dw.len());
    assert!(w.len() % self.ops_per_row() == 0);

    let div_val = DIV_VAL.load(std::sync::atomic::Ordering::Relaxed) as i64;
    let div_val_f = F::from(div_val as u64);
    // Eta
    let eta = ETA_TOP.load(std::sync::atomic::Ordering::Relaxed);
    let eta = F::from(eta as u64);
    // div_outp_min_val
    let div_outp_min_val = DIV_OUTP_MIN_VAL.load(std::sync::atomic::Ordering::Relaxed) as i64;
    let div_inp_min_val_pos = F::from(DIV_INP_MIN_VAL_POS as u64);
    
    let div_inp_min_val_pos_i64 = -SHIFT_MIN_VAL;
    let div_inp_min_val_pos = F::from(div_inp_min_val_pos_i64 as u64);

    let outp_cell = layouter.assign_region(
      || "",
      |mut region| {
        let mut outp_cells = vec![];
        for (i, (w, dw)) in w.iter().zip(dw.iter()).enumerate() {
          let offset = i * NUM_COLS_PER_OP;

          let w_val = w.value().map(|x: &F| x.to_owned());
          let dw_val = dw.value().map(|x: &F| x.to_owned());
          let out_scaled = w_val.zip(dw_val).map(|(w, dw)| w * div_val_f - dw * eta);

          let div_mod = out_scaled.map(|x| {
            // add this in order to make sure we can divide
            let x_pos = x + div_inp_min_val_pos;
            let x_pos = if x_pos > F::zero() {
              x_pos
            } else {
              x_pos + div_val_f
            };
            // convert to u64
            let inp = convert_to_u64(&x_pos);
            // Divide
            let div_res = inp as i64 / div_val - (DIV_INP_MIN_VAL_POS as i64 / div_val);
            let mod_res = inp as i64 % div_val;
            (div_res, mod_res)
          });

          w.copy_advice(|| "", &mut region, self.config.shared_columns[offset], 0)?;
          dw.copy_advice(
            || "",
            &mut region,
            self.config.shared_columns[offset + 1],
            0,
          )?;
          let div_res_cell = region
            .assign_advice(
              || "div_res",
              self.config.shared_columns[offset + 2],
              0,
              || {
                div_mod.map(|(x, _): (i64, i64)| {
                  F::from((x - div_outp_min_val as i64) as u64) - F::from(-div_outp_min_val as u64)
                })
              },
            )
            .unwrap();
          let _mod_res_cell = region
            .assign_advice(
              || "mod_res",
              self.config.shared_columns[offset + 3],
              0,
              || div_mod.map(|(_, x): (i64, i64)| F::from(x as u64)),
            )
            .unwrap();

          outp_cells.push(div_res_cell);
        }
        Ok(outp_cells)
      },
    );

    outp_cell
  }

  pub fn update(
    &self,
    mut layouter: impl Layouter<F>,
    w: &Vec<AssignedCell<F, F>>,
    dw: &Vec<AssignedCell<F, F>>,
    zero: &AssignedCell<F, F>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let mut w = w.clone();
    let mut dw = dw.clone();
    let out_len = w.len();
    while w.len() % self.ops_per_row() != 0 {
      w.push(zero.clone());
      dw.push(zero.clone());
    }

    let mut updated = vec![];
    for i in 0..w.len() / self.ops_per_row() {
      let offset = i * self.ops_per_row();
      let w_row = w[offset..offset + self.ops_per_row()].to_vec();
      let dw_row = dw[offset..offset + self.ops_per_row()].to_vec();
      let tmp = self
        .update_row(layouter.namespace(|| ""), &w_row, &dw_row)
        .unwrap();
      updated.extend(tmp);
    }
    Ok(updated[0..out_len].to_vec())
  }
}
