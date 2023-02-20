use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::gadget::GadgetConfig;

use super::super::layer::{Layer, LayerConfig};

pub struct TransposeChip {}

impl<F: FieldExt> Layer<F> for TransposeChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    _constants: &HashMap<i64, AssignedCell<F, F>>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
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

    let inp = tensors[0].to_owned();
    let inp = inp.into_shape(IxDyn(&inp_shape)).unwrap();

    let inp = inp.permuted_axes(IxDyn(&permutation));

    Ok(vec![inp])
  }
}
