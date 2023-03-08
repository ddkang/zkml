use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    gadget::{Gadget, GadgetConfig, GadgetType},
    mul_pairs::MulPairsChip,
    var_div::VarDivRoundChip,
  },
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::{
  super::layer::{Layer, LayerConfig},
  Arithmetic,
};

#[derive(Clone, Debug)]
pub struct MulChip {}

impl<F: FieldExt> Arithmetic<F> for MulChip {
  fn gadget_forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    constants: &Vec<&AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let mul_pairs_chip = MulPairsChip::<F>::construct(gadget_config.clone());

    let out = mul_pairs_chip.forward(
      layouter.namespace(|| "mul pairs chip"),
      &vec_inputs,
      constants,
    )?;
    Ok(out)
  }
}

// FIXME: move this + add to an arithmetic layer
impl<F: FieldExt> Layer<F> for MulChip {
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

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
    let div = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .as_ref();
    let zero = constants.get(&0).unwrap().as_ref();
    let single_inputs = vec![zero, div];
    let out = out.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let out = var_div_chip.forward(
      layouter.namespace(|| "average div"),
      &vec![out],
      &single_inputs,
    )?;

    let out = out.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(out_shape.as_slice()), out).unwrap();
    Ok(vec![out])
  }
}

impl GadgetConsumer for MulChip {
  fn used_gadgets(&self) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::MulPairs,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ]
  }
}
