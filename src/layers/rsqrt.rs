use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{circuit::Layouter, halo2curves::FieldExt, plonk::Error};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig},
  nonlinear::rsqrt::RsqrtGadgetChip,
};

use super::layer::{AssignedTensor, CellRc, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct RsqrtChip {}

impl<F: FieldExt> Layer<F> for RsqrtChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
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

    let min_val = gadget_config.min_val;
    let min_val = constants.get(&min_val).unwrap();
    let max_val = gadget_config.max_val;
    let max_val = constants.get(&max_val).unwrap();
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
    let inp_vec = inp_vec.iter().map(|x| x.as_ref()).collect();

    let zero = constants.get(&0).unwrap();
    let rsqrt_chip = RsqrtGadgetChip::<F>::construct(gadget_config.clone());
    let vec_inps = vec![inp_vec];
    let constants = vec![(**zero).clone(), (**min_val).clone(), (**max_val).clone()];
    let out = rsqrt_chip.forward(layouter.namespace(|| "rsqrt chip"), &vec_inps, &constants)?;

    let out = out.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(inp.shape()), out).unwrap();

    Ok(vec![out])
  }
}
