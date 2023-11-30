use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::ff::PrimeField,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig, GadgetType},
  var_div::VarDivRoundChip,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct DivFixedChip {}

impl DivFixedChip {
  fn get_div_val<F: PrimeField>(
    &self,
    mut layouter: impl Layouter<F>,
    _tensors: &Vec<AssignedTensor<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<(AssignedCell<F, F>, F), Error> {
    // FIXME: this needs to be revealed
    let div = layer_config.layer_params[0];
    let div = F::from(div as u64);

    let divc = layouter
      .assign_region(
        || "division",
        |mut region| {
          let div = region
            .assign_advice(
              || "avg pool 2d div",
              gadget_config.columns[0],
              0,
              || Value::known(div),
            )
            .unwrap();
          Ok(div)
        },
      )?;

    Ok((divc, div))
  }
}

impl<F: PrimeField> Layer<F> for DivFixedChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
    let inp_flat = inp.iter().map(|x| (x.0.as_ref(), x.1)).collect::<Vec<_>>();

    let zero = (constants.get(&0).unwrap().as_ref(), F::ZERO);
    let shape = inp.shape();

    let div = self.get_div_val(
      layouter.namespace(|| "average div"),
      tensors,
      gadget_config.clone(),
      layer_config,
    )?;
    let div = (&div.0, div.1);

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());

    let dived = var_div_chip.forward(
      layouter.namespace(|| "average div"),
      &vec![inp_flat],
      &vec![zero, div],
    )?;
    let dived = dived.into_iter().map(|x| (Rc::new(x.0), x.1)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(shape), dived).unwrap();

    Ok(vec![out])
  }
}

impl GadgetConsumer for DivFixedChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![GadgetType::VarDivRound]
  }
}
