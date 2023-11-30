use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::super::layer::{Layer, LayerConfig};

pub struct TransposeChip {}

impl<F: PrimeField> Layer<F> for TransposeChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    assert_eq!(layer_config.layer_params.len() % 2, 0);
    let ndim = layer_config.layer_params.len() / 2;
    let inp_shape = layer_config.layer_params[0..ndim]
      .to_vec()
      .iter()
      .map(|x| *x as usize)
      .collect::<Vec<_>>();
    let permutation = layer_config.layer_params[ndim..]
      .to_vec()
      .iter()
      .map(|x| *x as usize)
      .collect::<Vec<_>>();

    let inp = &tensors[0];
    // Required because of memory layout issues
    let inp_flat = inp.iter().cloned().collect::<Vec<_>>();
    let inp = Array::from_shape_vec(IxDyn(&inp_shape), inp_flat).unwrap();

    let inp = inp.permuted_axes(IxDyn(&permutation));

    Ok(vec![inp])
  }
}

impl GadgetConsumer for TransposeChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
