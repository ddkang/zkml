use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error},
};
use ndarray::{Array, Axis, IxDyn};

use crate::gadgets::gadget::GadgetConfig;

use super::{
  averager::Averager,
  layer::{Layer, LayerConfig, LayerType},
};

pub struct MeanChip<F: FieldExt> {
  config: LayerConfig,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> MeanChip<F> {
  pub fn construct(config: LayerConfig) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(_meta: ConstraintSystem<F>, layer_params: Vec<i64>) -> LayerConfig {
    LayerConfig {
      layer_type: LayerType::Mean,
      layer_params,
    }
  }

  pub fn get_keep_axis(&self) -> usize {
    match self.config.layer_params[0] as usize {
      1 => 2,
      2 => 1,
      _ => panic!("Invalid axis"),
    }
  }
}

impl<F: FieldExt> Averager<F> for MeanChip<F> {
  fn splat<G: Clone>(&self, input: &Array<G, IxDyn>) -> Vec<Vec<G>> {
    // Only support batch size = 1
    assert_eq!(input.shape()[0], 1);
    // Only support batch + 2D, summing over one axis
    assert_eq!(input.shape().len(), 3);
    let keep_axis = self.get_keep_axis();

    let mut splat = vec![];
    for i in 0..input.shape()[keep_axis] {
      let mut tmp = vec![];
      for x in input.index_axis(Axis(keep_axis), i).iter() {
        tmp.push(x.clone());
      }
      splat.push(tmp);
    }

    splat
  }

  fn get_div_val(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<AssignedCell<F, F>, Error> {
    let inp = &tensors[0];
    let keep_axis = self.get_keep_axis();
    let mut div = 1;
    for i in 0..inp.shape().len() {
      if i != keep_axis {
        div *= inp.shape()[i];
      }
    }

    let div = F::from(div as u64);
    // FIXME: put this in the fixed column
    let div = layouter.assign_region(
      || "mean div",
      |mut region| {
        let div = region.assign_advice(
          || "mean div",
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

impl<F: FieldExt> Layer<F> for MeanChip<F> {
  fn forward(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    let dived = self.avg_forward(layouter, tensors, constants, gadget_config)?;

    let inp = &tensors[0];
    let keep_axis = self.get_keep_axis();
    // FIXME: only support batch size = 1
    let outp_shape = match keep_axis {
      1 => vec![1, inp.shape()[keep_axis], 1],
      2 => vec![1, 1, inp.shape()[keep_axis]],
      _ => panic!("Invalid axis"),
    };

    let out = Array::from_shape_vec(IxDyn(&outp_shape), dived).unwrap();
    Ok(vec![out])
  }
}
