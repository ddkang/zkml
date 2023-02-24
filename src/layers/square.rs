use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig},
  square::SquareGadgetChip,
  var_div::VarDivRoundChip,
};

use super::layer::{Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct SquareChip {}

impl<F: FieldExt> Layer<F> for SquareChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    assert_eq!(tensors.len(), 1);

    let inp = &tensors[0];

    let square_chip = SquareGadgetChip::<F>::construct(gadget_config.clone());
    let inp_vec = inp.iter().collect::<Vec<_>>();
    let vec_inputs = vec![inp_vec];
    let single_inps = vec![constants.get(&0).unwrap().clone()];
    let out = square_chip.forward(
      layouter.namespace(|| "square chip"),
      &vec_inputs,
      &single_inps,
    )?;

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
    let div = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .clone();
    let single_inps = vec![constants.get(&0).unwrap().clone(), div];
    let out = out.iter().collect::<Vec<_>>();
    let vec_inputs = vec![out];
    let out = var_div_chip.forward(
      layouter.namespace(|| "var div chip"),
      &vec_inputs,
      &single_inps,
    )?;

    let out = Array::from_shape_vec(IxDyn(inp.shape()), out).unwrap();
    Ok(vec![out])
  }
}
