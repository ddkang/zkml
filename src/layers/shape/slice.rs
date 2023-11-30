use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::Slice;

use crate::{
  gadgets::gadget::{GadgetConfig, GadgetType},
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::super::layer::{Layer, LayerConfig};

pub struct SliceChip {}

impl<F: PrimeField> Layer<F> for SliceChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let params = &layer_config.layer_params;
    assert_eq!(params.len() % 2, 0);
    let num_axes = params.len() / 2;
    let starts = &params[0..num_axes];
    let sizes = &params[num_axes..];

    let inp = &tensors[0];
    let outp = inp.slice_each_axis(|ax| {
      let start = starts[ax.axis.0] as usize;
      let size = sizes[ax.axis.0];
      if size == -1 {
        Slice::from(start..)
      } else {
        Slice::from(start..(start + size as usize))
      }
    });
    Ok(vec![outp.to_owned()])
  }
}

impl GadgetConsumer for SliceChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<GadgetType> {
    vec![]
  }
}
