use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{concatenate, Axis};

use crate::{
  gadgets::gadget::{GadgetConfig, GadgetType},
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::super::layer::{Layer, LayerConfig};

pub struct ConcatenationChip {}

impl<F: PrimeField> Layer<F> for ConcatenationChip {
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
    let views = tensors.iter().map(|x| x.view()).collect::<Vec<_>>();
    // TODO: this is a bit of a hack
    let out = concatenate(Axis(axis), views.as_slice()).unwrap_or(tensors[0].clone());

    Ok(vec![out])
  }
}

impl GadgetConsumer for ConcatenationChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<GadgetType> {
    vec![]
  }
}
