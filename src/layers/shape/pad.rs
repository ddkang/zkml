use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::ff::PrimeField,
  plonk::Error,
};
use ndarray::{Array, Axis, IxDyn, Slice};

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::layer::{AssignedTensor, GadgetConsumer, CellRc},
};

use super::super::layer::{Layer, LayerConfig};

// TODO: figure out where to put this
pub fn pad<G: Clone, F: PrimeField>(
  input: &Array<(Rc<G>, F), IxDyn>,
  padding: Vec<[usize; 2]>,
  pad_val: &(Rc<G>, F),
) -> Array<(Rc<G>, F), IxDyn> {
  let tmp = input.iter().collect();
  let input = Array::from_shape_vec(input.raw_dim(), tmp).unwrap();
  assert_eq!(input.ndim(), padding.len());
  let mut padded_shape = input.raw_dim();
  for (ax, (&ax_len, &[pad_lo, pad_hi])) in input.shape().iter().zip(&padding).enumerate() {
    padded_shape[ax] = ax_len + pad_lo + pad_hi;
  }

  let mut padded = Array::from_elem(padded_shape, pad_val);
  let padded_dim = padded.raw_dim();
  {
    // Select portion of padded array that needs to be copied from the
    // original array.
    let mut orig_portion = padded.view_mut();
    for (ax, &[pad_lo, pad_hi]) in padding.iter().enumerate() {
      orig_portion.slice_axis_inplace(
        Axis(ax),
        Slice::from(pad_lo as isize..padded_dim[ax] as isize - (pad_hi as isize)),
      );
    }
    // Copy the data from the original array.
    orig_portion.assign(&input.view());
  }

  let dim = padded.raw_dim();
  let tmp = padded.into_iter().map(|x| x.clone()).collect();
  let padded = Array::from_shape_vec(dim, tmp).unwrap();

  padded
}

pub struct PadChip {}

pub struct PadConfig {
  pub padding: Vec<[usize; 2]>,
}

impl PadChip {
  pub fn param_vec_to_config(layer_params: Vec<i64>) -> PadConfig {
    assert!(layer_params.len() % 2 == 0);

    let padding = layer_params
      .chunks(2)
      .map(|chunk| [chunk[0] as usize, chunk[1] as usize])
      .collect();
    PadConfig { padding }
  }
}

impl<F: PrimeField> Layer<F> for PadChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, Rc<AssignedCell<F, F>>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    // FIXME: the pad from tflite is actually two, but mine is one
    // assert_eq!(tensors.len(), 1);
    let input = &tensors[0];

    let zero = constants.get(&0).unwrap().clone();
    let padding = PadChip::param_vec_to_config(layer_config.layer_params.clone());
    let padded = pad(input, padding.padding, &(zero, F::ZERO));

    Ok(vec![padded])
  }
}

impl GadgetConsumer for PadChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
