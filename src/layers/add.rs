use std::{collections::HashMap, marker::PhantomData, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error},
};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  add_pairs::AddPairsChip,
  gadget::{Gadget, GadgetConfig},
};

use super::layer::{Layer, LayerConfig, LayerType};

#[derive(Clone, Debug)]
pub struct AddChip<F: FieldExt> {
  _marker: PhantomData<F>,
}

impl<F: FieldExt> AddChip<F> {
  pub fn construct() -> Self {
    Self {
      _marker: PhantomData,
    }
  }

  pub fn configure(_meta: &mut ConstraintSystem<F>, layer_params: Vec<i64>) -> LayerConfig {
    LayerConfig {
      layer_type: LayerType::Add,
      layer_params,
    }
  }
}

impl<F: FieldExt> Layer<F> for AddChip<F> {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    assert_eq!(tensors.len(), 2);
    let inp1 = &tensors[0];
    // FIXME
    let inp2 = &tensors[1].broadcast(inp1.shape()).unwrap();
    assert_eq!(inp1.shape(), inp2.shape());

    let zero = constants.get(&0).unwrap().clone();

    let add_pairs_chip = AddPairsChip::<F>::construct(gadget_config);
    let inp1_vec = inp1.iter().collect::<Vec<_>>();
    let inp2_vec = inp2.iter().collect::<Vec<_>>();
    let vec_inputs = vec![inp1_vec, inp2_vec];
    let constants = vec![zero.clone()];
    let out = add_pairs_chip.forward(layouter.namespace(|| "add chip"), &vec_inputs, &constants)?;
    let out = Array::from_shape_vec(IxDyn(inp1.shape()), out).unwrap();

    Ok(vec![out])
  }
}
