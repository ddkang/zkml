<<<<<<< HEAD
// TODO: The implementation is not ideal.

=======
>>>>>>> e10397f (save backwards)
use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::FieldExt, plonk::Error};
use ndarray::Array;

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::super::layer::{Layer, LayerConfig};

<<<<<<< HEAD
pub struct RotateChip {}

impl<F: FieldExt> Layer<F> for RotateChip {
=======
pub struct ReshapeChip {}

impl<F: FieldExt> Layer<F> for ReshapeChip {
>>>>>>> e10397f (save backwards)
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _constants: &HashMap<i64, CellRc<F>>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
<<<<<<< HEAD
    let params = &layer_config.layer_params;

    assert!(inp.shape().len() == 4);
    
    let mut flip = vec![false; 4];
    for p in params {
      flip[*p as usize] = true;
    }
    // See which layers to 'flip'
    let shape = inp.shape();

    println!("Rotate: {:?} -> {:?}", inp.shape(), shape);

    let mut out = inp.clone();

    for i in 0..shape[0] {
      for j in 0..shape[1] {
        for k in 0..shape[2] {
          for l in 0..shape[3] {
            let [ix, jx, kx, lx]: [usize; 4] = [i, j, k, l].iter().enumerate().map(|(idx, x)| {
              if flip[idx] {
                shape[idx] - 1 - *x
              } else {
                *x
              }
            }).collect::<Vec<_>>().try_into().unwrap();
            out[[ix, jx, kx, lx]] = inp[[i, j, k, l]].clone();
          }
        }
      }
    }

=======
    let shape = layer_config.out_shapes[0].clone();

    println!("Reshape: {:?} -> {:?}", inp.shape(), shape);
>>>>>>> e10397f (save backwards)
    let flat = inp.iter().map(|x| x.clone()).collect();
    let out = Array::from_shape_vec(shape, flat).unwrap();
    Ok(vec![out])
  }
}

<<<<<<< HEAD
impl GadgetConsumer for RotateChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
=======
impl GadgetConsumer for ReshapeChip {
  fn used_gadgets(&self) -> Vec<crate::gadgets::gadget::GadgetType> {
>>>>>>> e10397f (save backwards)
    vec![]
  }
}
