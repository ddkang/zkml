use std::{collections::HashMap, marker::PhantomData, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error},
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    gadget::{Gadget, GadgetConfig},
    mul_pairs::MulPairsChip,
    var_div::VarDivRoundChip,
  },
  utils::helpers::broadcast,
};

use super::layer::{Layer, LayerConfig, LayerType};

#[derive(Clone, Debug)]
pub struct MulChip<F: FieldExt> {
  _marker: PhantomData<F>,
}

impl<F: FieldExt> MulChip<F> {
  pub fn construct() -> Self {
    Self {
      _marker: PhantomData,
    }
  }

  pub fn configure(_meta: &mut ConstraintSystem<F>, layer_params: Vec<i64>) -> LayerConfig {
    LayerConfig {
      layer_type: LayerType::Mul,
      layer_params,
    }
  }

  fn get_div_val(
    &self,
    mut layouter: impl Layouter<F>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<AssignedCell<F, F>, Error> {
    // FIXME: this needs to be revealed, fixed column
    let div = gadget_config.scale_factor;
    let div = F::from(div as u64);
    let div = layouter.assign_region(
      || "mul div",
      |mut region| {
        let div = region.assign_advice(
          || "mul div",
          gadget_config.columns[0],
          0,
          || Value::known(div),
        )?;
        Ok(div)
      },
    )?;

    Ok(div)
  }
}

// FIXME: move this + add to an arithmetic layer
impl<F: FieldExt> Layer<F> for MulChip<F> {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    assert_eq!(tensors.len(), 2);
    println!("tensors: {:?} {:?}", tensors[0].shape(), tensors[1].shape());
    let (inp1, inp2) = broadcast(&tensors[0], &tensors[1]);
    // let inp1 = &tensors[0];
    // let inp2 = &tensors[1].broadcast(inp1.shape()).unwrap();
    assert_eq!(inp1.shape(), inp2.shape());

    let zero = constants.get(&0).unwrap().clone();

    let mul_pairs_chip = MulPairsChip::<F>::construct(gadget_config.clone());
    let inp1_vec = inp1.iter().collect::<Vec<_>>();
    let inp2_vec = inp2.iter().collect::<Vec<_>>();
    let vec_inputs = vec![inp1_vec, inp2_vec];
    let constants = vec![zero.clone()];
    let out = mul_pairs_chip.forward(
      layouter.namespace(|| "mul pairs chip"),
      &vec_inputs,
      &constants,
    )?;

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
    let div = self.get_div_val(layouter.namespace(|| "get div"), gadget_config.clone())?;
    let single_inputs = vec![zero, div];
    let out = out.iter().map(|x| x).collect::<Vec<_>>();
    let out = var_div_chip.forward(
      layouter.namespace(|| "average div"),
      &vec![out],
      &single_inputs,
    )?;

    let out = Array::from_shape_vec(IxDyn(inp1.shape()), out).unwrap();
    Ok(vec![out])
  }
}
