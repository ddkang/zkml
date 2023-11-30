//
// Broadcast is used as a temporary measure to represent a the backprop
// of a full-kernel AvgPool2D
//

use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::Array;

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::super::layer::{Layer, LayerConfig};

pub struct BroadcastChip {}

// TODO: Fix this after demo
impl<F: PrimeField> Layer<F> for BroadcastChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
    let shape = inp.shape();
    let output_shape = layer_config.out_shapes[0].clone();

    // Check that we only broadcast dimensions with shape 1
    assert!(shape.len() == output_shape.len());
    assert!(shape.len() == 4);

    for (inp, outp) in shape.iter().zip(output_shape.iter()) {
      if *inp != *outp && !(*inp == 1) {
        panic!();
      }
    }

    let mut output_flat = vec![];

    for i in 0..output_shape[0] {
      for j in 0..output_shape[1] {
        for k in 0..output_shape[2] {
          for l in 0..output_shape[3] {
            let indexes = [i, j, k, l]
              .iter()
              .enumerate()
              .map(|(idx, x)| if shape[idx] == 1 { 0 } else { *x })
              .collect::<Vec<_>>();
            output_flat.push(inp[[indexes[0], indexes[1], indexes[2], indexes[3]]].clone());
          }
        }
      }
    }

    // println!("Broadcast : {:?} -> {:?}", inp.shape(), output_shape);
    let out = Array::from_shape_vec(output_shape, output_flat).unwrap();
    Ok(vec![out])
  }
}

impl GadgetConsumer for BroadcastChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
