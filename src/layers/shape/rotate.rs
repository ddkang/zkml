// TODO: The implementation is not ideal.

use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::super::layer::{Layer, LayerConfig};

pub struct RotateChip {}

// Example:
// input:
// [1 2 3 4]
// [5 6 7 8]
//
// params: [1] -- flip axis 1 only
// output:
// [4 3 2 1]
// [8 7 6 5]
impl<F: PrimeField> Layer<F> for RotateChip {
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
    let params = &layer_config.layer_params;

    assert!(inp.shape().len() == 4);

    let mut flip = vec![false; 4];
    for p in params {
      flip[*p as usize] = true;
    }
    let shape = inp.shape();

    // println!("Rotate: {:?} -> {:?}", inp.shape(), shape);

    let mut out = inp.clone();

    for i in 0..shape[0] {
      for j in 0..shape[1] {
        for k in 0..shape[2] {
          for l in 0..shape[3] {
            let [ix, jx, kx, lx]: [usize; 4] = [i, j, k, l]
              .iter()
              .enumerate()
              .map(|(idx, x)| if flip[idx] { shape[idx] - 1 - *x } else { *x })
              .collect::<Vec<_>>()
              .try_into()
              .unwrap();
            out[[ix, jx, kx, lx]] = inp[[i, j, k, l]].clone();
          }
        }
      }
    }

    Ok(vec![out])
  }
}

impl GadgetConsumer for RotateChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
