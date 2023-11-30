use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error, Expression, Selector},
  poly::Rotation,
};

use crate::gadgets::gadget::convert_to_u128;

use super::super::gadget::Gadget;
use super::super::gadget::{GadgetConfig, GadgetType};

const NUM_COLS_PER_OP: usize = 2;

pub trait NonLinearGadget<F: PrimeField>: Gadget<F> {
  fn generate_map(scale_factor: u64, min_val: i64, num_rows: i64) -> HashMap<i64, i64>;

  fn get_map(&self) -> &HashMap<i64, i64>;

  fn get_selector(&self) -> Selector;

  fn num_cols_per_op() -> usize {
    NUM_COLS_PER_OP
  }

  fn configure(
    meta: &mut ConstraintSystem<F>,
    gadget_config: GadgetConfig,
    gadget_type: GadgetType,
  ) -> GadgetConfig {
    let selector = meta.complex_selector();
    let columns = gadget_config.columns;

    let mut tables = gadget_config.tables;
    let inp_lookup = tables.get(&GadgetType::InputLookup).unwrap()[0];
    let outp_lookup = meta.lookup_table_column();

    for op_idx in 0..columns.len() / NUM_COLS_PER_OP {
      let offset = op_idx * NUM_COLS_PER_OP;
      meta.lookup("non-linear lookup", |meta| {
        let s = meta.query_selector(selector);
        let inp = meta.query_advice(columns[offset + 0], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 1], Rotation::cur());
        let shift_val = gadget_config.min_val;
        let shift_val_pos = Expression::Constant(F::from((-shift_val) as u64));

        vec![
          (s.clone() * (inp + shift_val_pos), inp_lookup),
          (s.clone() * outp, outp_lookup),
        ]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(gadget_type, vec![selector]);

    tables.insert(gadget_type, vec![inp_lookup, outp_lookup]);

    let mut maps = gadget_config.maps;
    let non_linear_map = Self::generate_map(
      gadget_config.scale_factor,
      gadget_config.min_val,
      gadget_config.num_rows as i64,
    );
    maps.insert(gadget_type, vec![non_linear_map]);

    GadgetConfig {
      columns,
      selectors,
      tables,
      maps,
      ..gadget_config
    }
  }

  fn load_lookups(
    &self,
    mut layouter: impl Layouter<F>,
    config: Rc<GadgetConfig>,
    gadget_type: GadgetType,
  ) -> Result<(), Error> {
    let map = self.get_map();
    let table_col = config.tables.get(&gadget_type).unwrap()[1];

    let shift_pos_i64 = -config.shift_min_val;
    let shift_pos = F::from(shift_pos_i64 as u64);
    layouter.assign_table(
      || "non linear table",
      |mut table| {
        for i in 0..config.num_rows {
          let i = i as i64;
          // FIXME: refactor this
          let tmp = *map.get(&i).unwrap();
          let val = if i == 0 {
            F::ZERO
          } else {
            if tmp >= 0 {
              F::from(tmp as u64)
            } else {
              let tmp = tmp + shift_pos_i64;
              F::from(tmp as u64) - shift_pos
            }
          };
          table.assign_cell(
            || "non linear cell",
            table_col,
            i as usize,
            || Value::known(val),
          )?;
        }
        Ok(())
      },
    )?;
    Ok(())
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    _single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let columns = &gadget_config.columns;
    let inp = &vec_inputs[0];
    let map = self.get_map();
    let shift_val_pos_i64 = -gadget_config.shift_min_val;
    let shift_val_pos = F::from(shift_val_pos_i64 as u64);
    let min_val = gadget_config.min_val;

    if gadget_config.use_selectors {
      let selector = self.get_selector();
      selector.enable(region, row_offset)?;
    }

    let mut outps = vec![];
    for i in 0..inp.len() {
      let offset = i * 2;
      inp[i].0.copy_advice(|| "", region, columns[offset + 0], row_offset)?;
      let outp = {
        let pos = convert_to_u128(&(inp[i].1 + shift_val_pos)) as i128 - shift_val_pos_i64 as i128;
        let x = pos as i64 - min_val;
        // if ((*map).get(&x)).is_none() {
        //   println!("x: {}", x);
        // }
        let val = *map.get(&x).unwrap();
        if x == 0 {
          F::ZERO
        } else {
          if val >= 0 {
            F::from(val as u64)
          } else {
            let val_pos = val + shift_val_pos_i64;
            F::from(val_pos as u64) - F::from(shift_val_pos_i64 as u64)
          }
        }
      };

      let outpc =
        region.assign_advice(|| "nonlinearity", columns[offset + 1], row_offset, || Value::known(outp))?;
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
    let inp_len = vec_inputs[0].len();
    let mut inp = vec_inputs[0].clone();

    while inp.len() % self.num_inputs_per_row() != 0 {
      inp.push(*zero);
    }

    let vec_inputs = vec![inp];
    let outp = self.op_aligned_rows(
      layouter.namespace(|| format!("forward row {}", self.name())),
      &vec_inputs,
      &single_inputs,
    )?;

    Ok(outp[0..inp_len].to_vec())
  }
}
