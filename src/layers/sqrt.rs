use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig, GadgetType},
  nonlinear::sqrt::SqrtGadgetChip,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct SqrtChip {}

impl<F: PrimeField> Layer<F> for SqrtChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
    let mut inp_vec = vec![];

    let mask = &layer_config.mask;
    let mut mask_map = HashMap::new();
    for i in 0..mask.len() / 2 {
      mask_map.insert(mask[2 * i], mask[2 * i + 1]);
    }

    let min_val_cell = constants.get(&gadget_config.min_val).unwrap().as_ref();
    let min_val = {
      let shift_val_i64 = -gadget_config.min_val * 2;
      let shift_val_f = F::from(shift_val_i64 as u64);
      F::from((gadget_config.min_val + shift_val_i64) as u64) - shift_val_f
    };
    let max_val_cell = constants.get(&gadget_config.max_val).unwrap().as_ref();
    let max_val = {
      let shift_val_i64 = -gadget_config.min_val * 2;
      let shift_val_f = F::from(shift_val_i64 as u64);
      F::from((gadget_config.max_val + shift_val_i64) as u64) - shift_val_f
    };

    for (i, val) in inp.iter().enumerate() {
      let i = i as i64;
      if mask_map.contains_key(&i) {
        let mask_val = *mask_map.get(&i).unwrap();
        if mask_val == 1 {
          inp_vec.push(
            (max_val_cell, max_val)
          );
        } else if mask_val == -1 {
          inp_vec.push(
            (min_val_cell, min_val)
          );
        } else {
          panic!();
        }
      } else {
        inp_vec.push((val.0.as_ref(), val.1));
      }
    }

    let zero = constants.get(&0).unwrap().as_ref();
    let sqrt_chip = SqrtGadgetChip::<F>::construct(gadget_config.clone());
    let vec_inps = vec![inp_vec];

    let constants = vec![
      (zero, F::ZERO), 
      (min_val_cell, min_val), 
      (max_val_cell, max_val)
    ];
    let out = sqrt_chip.forward(layouter.namespace(|| "sqrt chip"), &vec_inps, &constants)?;

    let out = out.into_iter().map(|x| (Rc::new(x.0), x.1)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(inp.shape()), out).unwrap();

    Ok(vec![out])
  }
}

impl GadgetConsumer for SqrtChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![GadgetType::Sqrt, GadgetType::InputLookup]
  }
}
