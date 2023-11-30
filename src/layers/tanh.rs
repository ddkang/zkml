use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::Layouter,
  halo2curves::ff::PrimeField, 
  plonk::Error
};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig, GadgetType},
  nonlinear::tanh::TanhGadgetChip,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct TanhChip {}

impl<F: PrimeField> Layer<F> for TanhChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
    let inp_vec = inp.iter().map(|x| (x.0.as_ref(), x.1)).collect::<Vec<_>>();
    let zero = constants.get(&0).unwrap().as_ref();

    let tanh_chip = TanhGadgetChip::<F>::construct(gadget_config.clone());
    let vec_inps = vec![inp_vec];
    let constants = vec![(zero, F::ZERO)];
    let out = tanh_chip.forward(layouter.namespace(|| "tanh chip"), &vec_inps, &constants)?;

    let out = out.into_iter().map(|x| (Rc::new(x.0), x.1)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(inp.shape()), out).unwrap();

    Ok(vec![out])
  }
}

impl GadgetConsumer for TanhChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![GadgetType::Tanh, GadgetType::InputLookup]
  }
}
