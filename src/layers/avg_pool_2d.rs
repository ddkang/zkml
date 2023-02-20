use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::gadgets::gadget::GadgetConfig;

use super::{
  averager::Averager,
  layer::{Layer, LayerConfig},
};

pub struct AvgPool2DChip<F: FieldExt> {
  pub _marker: PhantomData<F>,
}

impl<F: FieldExt> Averager<F> for AvgPool2DChip<F> {
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
    _tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<AssignedCell<F, F>, Error> {
    // FIXME: this needs to be revealed
    let div = layer_config.layer_params[0] * layer_config.layer_params[1];
    let div = F::from(div as u64);
    let div = layouter.assign_region(
      || "avg pool 2d div",
      |mut region| {
        let div = region.assign_advice(
          || "avg pool 2d div",
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

impl<F: FieldExt> Layer<F> for AvgPool2DChip<F> {
  fn forward(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    let dived = self.avg_forward(layouter, tensors, constants, gadget_config, layer_config)?;

    let inp = &tensors[0];
    let mut outp_shape = inp.shape().to_vec();
    outp_shape[1] = 1;
    outp_shape[2] = 1;
    let out = Array::from_shape_vec(IxDyn(&outp_shape), dived).unwrap();
    Ok(vec![out])
  }
}
