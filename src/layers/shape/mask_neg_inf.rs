use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::FieldExt, plonk::Error};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::super::layer::{Layer, LayerConfig};

pub struct MaskNegInfChip {}

impl<F: FieldExt> Layer<F> for MaskNegInfChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
    let mask_ndim = layer_config.layer_params[0] as usize;
    let mask_shape = layer_config.layer_params[1..mask_ndim + 1]
      .iter()
      .map(|x| *x as usize)
      .collect::<Vec<_>>();

    let mask_vec = layer_config.layer_params[mask_ndim + 1..].to_vec();
    let mask = Array::from_shape_vec(IxDyn(&mask_shape), mask_vec).unwrap();
    let mask = mask.broadcast(inp.raw_dim()).unwrap();

    let min_val = gadget_config.min_val;
    let min_val = constants.get(&min_val).unwrap().clone();
    let mut out_vec = vec![];
    for (val, to_mask) in inp.iter().zip(mask.iter()) {
      if *to_mask == 0 {
        out_vec.push(val.clone());
      } else {
        out_vec.push(min_val.clone());
      }
    }

    let outp = Array::from_shape_vec(inp.raw_dim(), out_vec).unwrap();
    Ok(vec![outp])
  }
}

impl GadgetConsumer for MaskNegInfChip {
  fn used_gadgets(&self) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
