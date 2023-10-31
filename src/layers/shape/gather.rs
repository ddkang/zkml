use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{circuit::Layouter, halo2curves::ff::PrimeField, plonk::Error};
use ndarray::{Array, Axis};

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::layer::{AssignedTensor, CellRc, GadgetConsumer},
};

use super::super::layer::{Layer, LayerConfig};

pub struct GatherChip {}

impl<F: PrimeField> Layer<F> for GatherChip {
  fn forward(
    &self,
    _layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    _constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    _gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let inp = &tensors[0];
    let view = inp.dim(); // [size,batch]
    let idx = layer_config.layer_params.clone();
    

    let mut tmp = vec![];
    for col in inp.axis_iter(Axis(1)) {
        let flatten = col.iter().cloned().collect::<Vec<_>>();
        
        let _ = idx.iter().for_each(|x| tmp.push(flatten[(*x) as usize].clone()));
    }
    let out = Array::from_shape_vec(vec![view[1], idx.len()], tmp).unwrap().reversed_axes();    
    Ok(vec![out])
  }
}

impl GadgetConsumer for GatherChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
