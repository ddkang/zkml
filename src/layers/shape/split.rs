use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{Axis, Slice};

use crate::{
  gadgets::gadget::{GadgetConfig, GadgetType},
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::super::layer::{Layer, LayerConfig};

pub struct SplitChip {}

impl<F: PrimeField> Layer<F> for SplitChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let axis = layer_config.layer_params[0] as usize;
    let num_splits = layer_config.layer_params[1] as usize;
    let inp = &tensors[1];

    let mut out = vec![];
    let split_len = inp.shape()[axis] / num_splits;
    for i in 0..num_splits {
      let slice = inp
        .slice_axis(
          Axis(axis),
          Slice::from((i * split_len)..((i + 1) * split_len)),
        )
        .to_owned();
      out.push(slice.to_owned());
    }
    Ok(out)
  }
}

impl GadgetConsumer for SplitChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<GadgetType> {
    vec![]
  }
}
