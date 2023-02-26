use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::FieldExt, plonk::Error};
use ndarray::{IxDyn, stack, Axis};

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::layer::{AssignedTensor, CellRc},
};

use super::super::layer::{Layer, LayerConfig};

pub struct PackChip {}

impl<F: FieldExt> Layer<F> for PackChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _constants: &HashMap<i64, CellRc<F>>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    // Ensure that all of the tensors have the same dimensions
    let tensor_shape = tensors[0].shape();
    for tensor in tensors {
      assert!(tensor_shape == tensor.shape());
    }
    let axis = layer_config.layer_params[0];
    let mut tensor_views = vec![];

    for tensor in tensors.iter() {
      tensor_views.push(tensor.view());
    }
    let stacked_tensor = stack(Axis(axis as usize), &tensor_views).unwrap();
    
    Ok(vec![stacked_tensor])
  }
}
