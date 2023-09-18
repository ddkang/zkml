use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::Layouter, 
  halo2curves::ff::PrimeField, 
  plonk::Error
};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig, GadgetType},
  update::UpdateGadgetChip,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

#[derive(Clone, Debug)]
pub struct UpdateChip {}

impl<F: PrimeField + Ord> Layer<F> for UpdateChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let w = &tensors[0];
    let dw = &tensors[1];

    let zero = constants.get(&0).unwrap().as_ref();
    let update_chip = UpdateGadgetChip::<F>::construct((*gadget_config).clone());

    let flattened_w = w.into_iter().map(|x| (x.0.as_ref(), x.1)).collect::<Vec<_>>();
    let flattened_dw = dw.into_iter().map(|x| (x.0.as_ref(), x.1)).collect::<Vec<_>>();
    let flattened_w_ref = flattened_w.iter().map(|x| (x.0, x.1)).collect::<Vec<_>>();
    let flattened_dw_ref = flattened_dw.iter().map(|x| (x.0, x.1)).collect::<Vec<_>>();

    let vec_inps = vec![flattened_w_ref, flattened_dw_ref];
    let constants = vec![(zero, F::ZERO)];
    let out = update_chip.forward(layouter.namespace(|| "update chip"), &vec_inps, &constants)?;

    let out = out.into_iter().map(|x| (Rc::new(x.0), x.1)).collect::<Vec<_>>();
    let out = Array::from_shape_vec(IxDyn(w.shape()), out).unwrap();

    Ok(vec![out])
  }
}

impl GadgetConsumer for UpdateChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![GadgetType::Update]
  }
}
