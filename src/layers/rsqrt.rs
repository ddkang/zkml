use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig},
  nonlinear::rsqrt::RsqrtGadgetChip,
};

use super::layer::{Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct RsqrtChip {}

impl<F: FieldExt> Layer<F> for RsqrtChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    let inp = &tensors[0];
    let mut inp_vec = vec![];

    let mask = &layer_config.mask;
    let mut mask_map = HashMap::new();
    for i in 0..mask.len() / 2 {
      mask_map.insert(mask[2 * i], mask[2 * i + 1]);
    }

    let min_val = gadget_config.min_val;
    let min_val = constants.get(&min_val).unwrap().clone();
    let max_val = gadget_config.max_val;
    let max_val = constants.get(&max_val).unwrap().clone();
    for (i, val) in inp.iter().enumerate() {
      let i = i as i64;
      if mask_map.contains_key(&i) {
        let mask_val = *mask_map.get(&i).unwrap();
        if mask_val == 1 {
          inp_vec.push(max_val.clone());
        } else if mask_val == -1 {
          inp_vec.push(min_val.clone());
        } else {
          panic!();
        }
      } else {
        inp_vec.push(val.clone());
      }
    }
    let inp_vec = inp_vec.iter().map(|x| x).collect();

    let zero = constants.get(&0).unwrap().clone();
    let rsqrt_chip = RsqrtGadgetChip::<F>::construct(gadget_config.clone());
    let vec_inps = vec![inp_vec];
    let constants = vec![zero, min_val, max_val];
    let out = rsqrt_chip.forward(layouter.namespace(|| "rsqrt chip"), &vec_inps, &constants)?;

    let out = Array::from_shape_vec(IxDyn(inp.shape()), out).unwrap();

    Ok(vec![out])
  }
}
