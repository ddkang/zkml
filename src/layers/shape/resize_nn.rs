use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::super::layer::{Layer, LayerConfig};

pub struct ResizeNNChip {}

// TODO: this does not work in general
impl<F: PrimeField> Layer<F> for ResizeNNChip {
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
    let output_shape = layer_config.out_shapes[0].clone();

    assert_eq!(inp.ndim(), 4);
    assert_eq!(inp.shape()[0], 1);
    assert_eq!(inp.shape()[3], output_shape[3]);

    let mut flat = vec![];
    // Do nearest neighbor interpolation over batch, h, w, c
    // The interpolation is over h and w
    for b in 0..inp.shape()[0] {
      for h in 0..output_shape[1] {
        let h_in = (h as f64 * (inp.shape()[1] as f64 / output_shape[1] as f64)) as usize;
        for w in 0..output_shape[2] {
          let w_in = (w as f64 * (inp.shape()[2] as f64 / output_shape[2] as f64)) as usize;
          for c in 0..inp.shape()[3] {
            flat.push(inp[[b, h_in, w_in, c]].clone());
          }
        }
      }
    }

    let outp = Array::from_shape_vec(IxDyn(&output_shape), flat).unwrap();
    Ok(vec![outp])
  }
}

impl GadgetConsumer for ResizeNNChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
