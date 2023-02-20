use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error, Expression, Selector},
  poly::Rotation,
};

use super::gadget::{convert_to_u64, GadgetConfig, GadgetType};
use super::gadget::{Gadget, USE_SELECTORS};

const NUM_COLS_PER_OP: usize = 2;

// TODO: load lookups
pub trait NonLinearGadget<F: FieldExt>: Gadget<F> {
  fn generate_map(scale_factor: u64, min_val: i64, max_val: i64) -> HashMap<i64, i64>;

  fn get_map(&self) -> &HashMap<i64, i64>;

  fn get_selector(&self) -> Selector;

  fn configure(
    meta: &mut ConstraintSystem<F>,
    gadget_config: GadgetConfig,
    gadget_type: GadgetType,
  ) -> GadgetConfig {
    let selector = meta.complex_selector();
    let columns = gadget_config.columns;

    let mut tables = gadget_config.tables;
    let inp_lookup = if tables.contains_key(&GadgetType::BiasDivRoundRelu6) {
      tables.get(&GadgetType::BiasDivRoundRelu6).unwrap()[0]
    } else {
      meta.lookup_table_column()
    };
    let outp_lookup = meta.lookup_table_column();

    for op_idx in 0..columns.len() / NUM_COLS_PER_OP {
      let offset = op_idx * NUM_COLS_PER_OP;
      meta.lookup("non-linear lookup", |meta| {
        let s = meta.query_selector(selector);
        let inp = meta.query_advice(columns[offset + 0], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 1], Rotation::cur());
        let shift_val = gadget_config.shift_min_val;
        let shift_val = Expression::Constant(F::from((-shift_val) as u64));

        vec![
          (s.clone() * (inp + shift_val), inp_lookup),
          (s.clone() * outp, outp_lookup),
        ]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(gadget_type, vec![selector]);

    tables.insert(gadget_type, vec![inp_lookup, outp_lookup]);

    let mut maps = gadget_config.maps;
    let relu_map = Self::generate_map(
      gadget_config.scale_factor,
      gadget_config.min_val,
      gadget_config.max_val,
    );
    maps.insert(gadget_type, vec![relu_map]);

    GadgetConfig {
      columns,
      selectors,
      tables,
      maps,
      ..gadget_config
    }
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    _single_inputs: &Vec<AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let columns = &gadget_config.columns;
    let inp = &vec_inputs[0];
    let map = self.get_map();
    let shift_val_pos_i64 = -gadget_config.shift_min_val;
    let shift_val_pos = F::from(shift_val_pos_i64 as u64);
    let min_val = gadget_config.min_val;

    if USE_SELECTORS {
      let selector = self.get_selector();
      selector.enable(region, row_offset)?;
    }

    let mut outps = vec![];
    for i in 0..inp.len() {
      let offset = i * 2;
      inp[i].copy_advice(|| "", region, columns[offset + 0], row_offset)?;
      let outp = inp[i].value().map(|x: &F| {
        let pos = convert_to_u64(&(*x + shift_val_pos)) as i64 - shift_val_pos_i64;
        let x = pos - min_val;
        let val = map.get(&x).unwrap();
        F::from(*val as u64)
      });

      let outp =
        region.assign_advice(|| "nonlinearity", columns[offset + 1], row_offset, || outp)?;
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
    let inp_len = vec_inputs[0].len();
    let mut inp = vec_inputs[0].clone();

    while inp.len() % self.num_inputs_per_row() != 0 {
      inp.push(zero);
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
