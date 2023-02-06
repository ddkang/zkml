use std::{collections::HashMap, marker::PhantomData};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, Axis, IxDyn, Slice};

use crate::gadgets::gadget::GadgetConfig;

use super::layer::{Layer, LayerConfig};

// TODO: figure out where to put this
pub fn pad<G: Clone>(
  input: &Array<G, IxDyn>,
  padding: Vec<[usize; 2]>,
  pad_val: G,
) -> Array<G, IxDyn> {
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
    orig_portion.assign(input);
  }

  padded
}

pub struct PadChip<F: FieldExt> {
  config: LayerConfig,
  _marker: PhantomData<F>,
}

pub struct PadConfig {
  pub padding: Vec<[usize; 2]>,
}

impl<F: FieldExt> PadChip<F> {
  pub fn construct(config: LayerConfig) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn param_vec_to_config(layer_params: Vec<i64>) -> PadConfig {
    assert!(layer_params.len() % 2 == 0);

    let padding = layer_params
      .chunks(2)
      .map(|chunk| [chunk[0] as usize, chunk[1] as usize])
      .collect();
    PadConfig { padding }
  }
}

impl<F: FieldExt> Layer<F> for PadChip<F> {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    _gadget_config: &GadgetConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    assert_eq!(tensors.len(), 1);
    let input = &tensors[0];

    let zero = constants.get(&0).unwrap().clone();
    let padding = PadChip::<F>::param_vec_to_config(self.config.layer_params.clone());
    let padded = pad(input, padding.padding, zero);

    Ok(vec![padded])
  }
}
