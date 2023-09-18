use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::Layouter, 
  halo2curves::ff::PrimeField, 
  plonk::Error
};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig, GadgetType},
  square::SquareGadgetChip,
  var_div::VarDivRoundChip,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct SquareChip {}

impl<F: PrimeField> Layer<F> for SquareChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    assert_eq!(tensors.len(), 1);

    let inp = &tensors[0];
    let zero = constants.get(&0).unwrap().as_ref();

    let square_chip = SquareGadgetChip::<F>::construct(gadget_config.clone());
    let inp_vec = inp.iter().map(|x| (x.0.as_ref(), x.1)).collect::<Vec<_>>();
    let vec_inputs = vec![inp_vec];
    let single_inps = vec![(zero, F::ZERO)];
    let out = square_chip.forward(
      layouter.namespace(|| "square chip"),
      &vec_inputs,
      &single_inps,
    )?;

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
    let div_cell = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .as_ref();
    let div = {
      let shift_val_i64 = -gadget_config.min_val * 2;
      let shift_val_f = F::from(shift_val_i64 as u64);
      F::from((gadget_config.scale_factor as i64 + shift_val_i64) as u64) - shift_val_f
    };

    let single_inps = vec![
      (zero, F::ZERO), 
      (div_cell, div)
    ];
    let out = out.iter().map(|x| (&x.0, x.1)).collect::<Vec<_>>();
    let vec_inputs = vec![out];
    let out = var_div_chip.forward(
      layouter.namespace(|| "var div chip"),
      &vec_inputs,
      &single_inps,
    )?;

    let out = out.into_iter().map(|x| (Rc::new(x.0), x.1)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(inp.shape()), out).unwrap();
    Ok(vec![out])
  }
}

impl GadgetConsumer for SquareChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::Square,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ]
  }
}
