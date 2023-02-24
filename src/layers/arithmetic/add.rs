use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    add_pairs::AddPairsChip,
    gadget::{Gadget, GadgetConfig},
  },
  layers::layer::{AssignedTensor, CellRc},
};

use super::{
  super::layer::{Layer, LayerConfig},
  Arithmetic,
};

#[derive(Clone, Debug)]
pub struct AddChip {}

impl<F: FieldExt> Arithmetic<F> for AddChip {
  fn gadget_forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    constants: &Vec<AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let add_pairs_chip = AddPairsChip::<F>::construct(gadget_config);
    let out = add_pairs_chip.forward(layouter.namespace(|| "add chip"), &vec_inputs, &constants)?;
    Ok(out)
  }
}

impl<F: FieldExt> Layer<F> for AddChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let (out, out_shape) = self.arithmetic_forward(
      layouter.namespace(|| ""),
      tensors,
      constants,
      gadget_config.clone(),
    )?;
    let out = Array::from_shape_vec(IxDyn(out_shape.as_slice()), out).unwrap();

    Ok(vec![out])
  }
}
