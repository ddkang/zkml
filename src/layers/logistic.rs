use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig},
  nonlinear::logistic::LogisticGadgetChip,
};

use super::layer::{Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct LogisticChip {}

impl<F: FieldExt> Layer<F> for LogisticChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    let inp = &tensors[0];
    let inp_vec = inp.iter().collect::<Vec<_>>();
    let zero = constants.get(&0).unwrap().clone();

    let logistic_chip = LogisticGadgetChip::<F>::construct(gadget_config.clone());
    let vec_inps = vec![inp_vec];
    let constants = vec![zero.clone()];
    let out = logistic_chip.forward(
      layouter.namespace(|| "logistic chip"),
      &vec_inps,
      &constants,
    )?;

    let out = Array::from_shape_vec(IxDyn(inp.shape()), out).unwrap();

    Ok(vec![out])
  }
}
