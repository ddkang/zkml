use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::ff::PrimeField,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    gadget::{Gadget, GadgetConfig, GadgetType},
    mul_pairs::MulPairsChip,
    var_div::VarDivRoundChip,
  },
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer},
};

use super::Arithmetic;

pub struct DivVarChip {}

// TODO: hack. Used for multiplying by the scale factor
impl<F: PrimeField> Arithmetic<F> for DivVarChip {
  fn gadget_forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    constants: &Vec<(&AssignedCell<F, F>, F)>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let mul_pairs_chip = MulPairsChip::<F>::construct(gadget_config.clone());

    let out = mul_pairs_chip.forward(
      layouter.namespace(|| "mul pairs chip"),
      &vec_inputs,
      constants,
    )?;
    Ok(out)
  }
}

impl<F: PrimeField> Layer<F> for DivVarChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &crate::layers::layer::LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    assert_eq!(tensors.len(), 2);
    // TODO: We only support dividing by a single number for now
    assert_eq!(tensors[1].shape().len(), 1);
    assert_eq!(tensors[1].shape()[0], 1);

    let sf_cell = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .as_ref();
    let sf = {
      let shift_val_i64 = -gadget_config.min_val * 2;
      let shift_val_f = F::from(shift_val_i64 as u64);
      F::from((gadget_config.scale_factor as i64 + shift_val_i64) as u64) - shift_val_f
    };

    let sf_tensor = 
      Array::from_shape_vec(IxDyn(&[1]), vec![(Rc::new(sf_cell.clone()), sf)]).unwrap();

    // out = inp * SF
    let (out, out_shape) = self.arithmetic_forward(
      layouter.namespace(|| ""),
      &vec![tensors[0].clone(), sf_tensor],
      constants,
      gadget_config.clone(),
    )?;

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
    let div = tensors[1].iter().next().map(|x| (x.0.as_ref(), x.1)).unwrap();
    let zero = (constants.get(&0).unwrap().as_ref(), F::ZERO);
    let single_inputs = vec![zero, div];
    let out = out.iter().map(|x| (x.0.as_ref(), x.1)).collect::<Vec<_>>();
    let out = var_div_chip.forward(layouter.namespace(|| "mul div"), &vec![out], &single_inputs)?;

    let out = out.into_iter().map(|x| (Rc::new(x.0), x.1)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(out_shape.as_slice()), out).unwrap();
    Ok(vec![out])
  }
}

impl GadgetConsumer for DivVarChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::MulPairs,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ]
  }
}
