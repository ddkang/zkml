use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::FieldExt,
  plonk::{Error, Selector},
};

use super::gadget::{convert_to_u64, GadgetConfig};
use super::gadget::{Gadget, USE_SELECTORS};

pub trait NonLinearGadget<F: FieldExt>: Gadget<F> {
  fn get_map(&self) -> HashMap<i64, i64>;

  fn get_selector(&self) -> Selector;

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
