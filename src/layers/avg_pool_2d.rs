use std::{collections::HashMap, marker::PhantomData};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error},
};
use ndarray::{Array, IxDyn};

use crate::gadgets::{
  adder::AdderChip,
  gadget::{Gadget, GadgetConfig},
};

use super::layer::{Layer, LayerConfig, LayerType};

pub struct AvgPool2DChip<F: FieldExt> {
  _marker: PhantomData<F>,
}

impl<F: FieldExt> AvgPool2DChip<F> {
  pub fn construct() -> Self {
    Self {
      _marker: PhantomData,
    }
  }

  pub fn configure(_meta: ConstraintSystem<F>, layer_params: Vec<i64>) -> LayerConfig {
    LayerConfig {
      layer_type: LayerType::AvgPool2D,
      layer_params,
    }
  }

  pub fn splat<G: Clone>(&self, input: &Array<G, IxDyn>) -> Vec<Vec<G>> {
    assert_eq!(input.shape().len(), 4);
    // Don't support batch size > 1 yet
    assert_eq!(input.shape()[0], 1);

    let mut splat = vec![];
    for k in 0..input.shape()[3] {
      let mut tmp = vec![];
      for i in 0..input.shape()[1] {
        for j in 0..input.shape()[2] {
          tmp.push(input[[0, i, j, k]].clone());
        }
      }
      splat.push(tmp);
    }
    splat
  }
}

impl<F: FieldExt> Layer<F> for AvgPool2DChip<F> {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: &GadgetConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    assert_eq!(tensors.len(), 1);

    let zero = constants.get(&0).unwrap().clone();

    let inp = &tensors[0];
    let splat_inp = self.splat(inp);

    let adder_chip = AdderChip::<F>::construct(gadget_config.clone());
    let single_inputs = vec![zero.clone()];
    let mut added = vec![];
    for i in 0..splat_inp.len() {
      let tmp = adder_chip.forward(
        layouter.namespace(|| format!("avg pool 2d {}", i)),
        &vec![splat_inp[i].clone()],
        &single_inputs,
      )?;
      added.push(tmp[0].clone());
    }

    let mut outp_shape = inp.shape().to_vec();
    outp_shape[1] = 1;
    outp_shape[2] = 1;
    let out = Array::from_shape_vec(IxDyn(&outp_shape), added).unwrap();
    Ok(vec![out])
  }
}
