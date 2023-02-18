use std::{cmp::min, collections::HashMap, marker::PhantomData, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error},
};
use ndarray::{Array, Axis, IxDyn};

use crate::gadgets::{
  gadget::{Gadget, GadgetConfig},
  squared_diff::SquaredDiffGadgetChip,
  var_div::VarDivRoundChip,
};

use super::layer::{Layer, LayerConfig, LayerType};

#[derive(Clone, Debug)]
pub struct SquaredDiffChip<F: FieldExt> {
  _marker: PhantomData<F>,
}

impl<F: FieldExt> SquaredDiffChip<F> {
  pub fn construct() -> Self {
    Self {
      _marker: PhantomData,
    }
  }

  pub fn configure(_meta: &mut ConstraintSystem<F>, layer_params: Vec<i64>) -> LayerConfig {
    LayerConfig {
      layer_type: LayerType::SquaredDifference,
      layer_params,
    }
  }
}

impl<F: FieldExt> Layer<F> for SquaredDiffChip<F> {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    assert_eq!(tensors.len(), 2);
    let mut inp1 = tensors[0].clone();
    let mut inp2 = tensors[1].clone();
    // Broadcoasting allowed...
    for i in 0..min(inp1.shape().len(), inp2.shape().len()) {
      assert_eq!(inp1.shape()[i], inp2.shape()[i]);
    }

    println!("inp1: {:?}", inp1.shape());
    println!("inp2: {:?}", inp2.shape());
    let inp1 = if inp1.ndim() < inp2.ndim() {
      inp1 = loop {
        let axis = Axis(inp1.ndim() - 1);
        inp1 = inp1.insert_axis(axis).clone();
        if inp1.ndim() == inp2.ndim() {
          break inp1;
        }
      };
      inp1.broadcast(inp2.shape()).unwrap().to_owned()
    } else {
      inp1.clone()
    };
    inp2 = if inp2.ndim() < inp1.ndim() {
      inp2 = loop {
        let axis = Axis(inp2.ndim());
        inp2 = inp2.insert_axis(axis).clone();
        if inp2.ndim() == inp1.ndim() {
          break inp2;
        }
      };
      inp2.broadcast(inp1.shape()).unwrap().to_owned()
    } else {
      inp2.clone()
    };
    println!("inp1: {:?}", inp1.shape());
    println!("inp2: {:?}", inp2.shape());

    let zero = constants.get(&0).unwrap().clone();

    let sq_diff_chip = SquaredDiffGadgetChip::<F>::construct(gadget_config.clone());
    let inp1_vec = inp1.iter().collect::<Vec<_>>();
    let inp2_vec = inp2.iter().collect::<Vec<_>>();
    let vec_inputs = vec![inp1_vec, inp2_vec];
    let constants = vec![zero.clone()];
    let out = sq_diff_chip.forward(
      layouter.namespace(|| "sq diff chip"),
      &vec_inputs,
      &constants,
    )?;

    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
    // FIXME: ugh
    let sf = gadget_config.scale_factor;
    let sf = F::from(sf as u64);
    let div = layouter.assign_region(
      || "sq diff sf",
      |mut region| {
        let div = region.assign_advice(
          || "sq diff sf",
          gadget_config.columns[0],
          0,
          || Value::known(sf),
        )?;
        Ok(div)
      },
    )?;

    let single_inputs = vec![zero, div];
    let out = out.iter().map(|x| x).collect::<Vec<_>>();
    let out = var_div_chip.forward(
      layouter.namespace(|| "sq diff div"),
      &vec![out],
      &single_inputs,
    )?;

    let out = Array::from_shape_vec(IxDyn(inp1.shape()), out).unwrap();

    Ok(vec![out])
  }
}
