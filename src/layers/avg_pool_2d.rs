use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::gadget::{GadgetConfig, GadgetType};

use super::{
  averager::Averager,
  layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig},
};

pub struct AvgPool2DChip {}

impl<F: FieldExt> Averager<F> for AvgPool2DChip {
  fn splat<G: Clone>(&self, input: &Array<G, IxDyn>, _layer_config: &LayerConfig) -> Vec<Vec<G>> {
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

  fn get_div_val(
    &self,
    mut layouter: impl Layouter<F>,
    _tensors: &Vec<AssignedTensor<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<AssignedCell<F, F>, Error> {
    // FIXME: this needs to be revealed
    let div = layer_config.layer_params[0] * layer_config.layer_params[1];
    let div = F::from(div as u64);
    let div = layouter
      .assign_region(
        || "avg pool 2d div",
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
      )
      .unwrap();

    Ok(div)
  }
}

impl<F: FieldExt> Layer<F> for AvgPool2DChip {
  fn forward(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let dived = self
      .avg_forward(layouter, tensors, constants, gadget_config, layer_config)
      .unwrap();

    let inp = &tensors[0];
    let mut outp_shape = inp.shape().to_vec();
    outp_shape[1] = 1;
    outp_shape[2] = 1;
    let out = Array::from_shape_vec(IxDyn(&outp_shape), dived).unwrap();
    Ok(vec![out])
  }
}

impl GadgetConsumer for AvgPool2DChip {
  fn used_gadgets(&self) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::Adder,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ]
  }
}
