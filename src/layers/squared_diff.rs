use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{circuit::Layouter, halo2curves::FieldExt, plonk::Error};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    gadget::{Gadget, GadgetConfig},
    squared_diff::SquaredDiffGadgetChip,
    var_div::VarDivRoundChip,
  },
  utils::helpers::broadcast,
};

use super::layer::{AssignedTensor, CellRc, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct SquaredDiffChip {}

impl<F: FieldExt> Layer<F> for SquaredDiffChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    assert_eq!(tensors.len(), 2);
    let inp1 = &tensors[0];
    let inp2 = &tensors[1];
    // Broadcoasting allowed... can't check shapes easily
    let (inp1, inp2) = broadcast(inp1, inp2);

    let zero = constants.get(&0).unwrap();

    let sq_diff_chip = SquaredDiffGadgetChip::<F>::construct(gadget_config.clone());
    let inp1_vec = inp1.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let inp2_vec = inp2.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let vec_inputs = vec![inp1_vec, inp2_vec];
    let tmp_constants = vec![(**zero).clone()];
    let out = sq_diff_chip.forward(
      layouter.namespace(|| "sq diff chip"),
      &vec_inputs,
      &tmp_constants,
    )?;

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
    let div = constants.get(&(gadget_config.scale_factor as i64)).unwrap();

    let single_inputs = vec![(**zero).clone(), (**div).clone()];
    let out = out.iter().map(|x| x).collect::<Vec<_>>();
    let out = var_div_chip.forward(
      layouter.namespace(|| "sq diff div"),
      &vec![out],
      &single_inputs,
    )?;

    let out = out.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(inp1.shape()), out).unwrap();

    Ok(vec![out])
  }
}
