use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::ff::PrimeField,
  plonk::Error,
};
use ndarray::{Array, Axis, IxDyn};

use crate::gadgets::gadget::{GadgetConfig, GadgetType};

use super::{
  averager::Averager,
  layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig},
};

pub struct MeanChip {}

impl MeanChip {
  pub fn get_keep_axis(&self, layer_config: &LayerConfig) -> usize {
    let inp_shape = &layer_config.inp_shapes[0];
    let out_shape = &layer_config.out_shapes[0];
    assert_eq!(inp_shape[0], 1);
    assert_eq!(out_shape[0], 1);

    // Skip the batch axis
    let mut keep_axes = (1..inp_shape.len()).collect::<Vec<_>>();
    for mean_axis in layer_config.layer_params.iter() {
      keep_axes.retain(|&x| x != *mean_axis as usize);
    }
    assert_eq!(keep_axes.len(), 1);
    keep_axes[0]

    /*
    let mut num_same = 0;
    let mut keep_axis: i64 = -1;
    for i in 1..inp_shape.len() {
      if inp_shape[i] == out_shape[i] {
        keep_axis = i as i64;
        num_same += 1;
      }
    }

    if keep_axis == -1 {
      panic!("All axes are different");
    }
    if num_same > 1 {
      panic!("More than one axis is the same");
    }
    keep_axis as usize
    */
  }
}

impl<F: PrimeField> Averager<F> for MeanChip {
  fn splat(&self, input: &AssignedTensor<F>, layer_config: &LayerConfig) -> Vec<Vec<(CellRc<F>,F)>> {
    // Only support batch size = 1
    assert_eq!(input.shape()[0], 1);
    // Only support batch + 2D, summing over one axis
    // assert_eq!(input.shape().len(), 3);
    let keep_axis = self.get_keep_axis(layer_config);

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
    tensors: &Vec<AssignedTensor<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<(AssignedCell<F, F>, F), Error> {
    let inp = &tensors[0];
    let keep_axis = self.get_keep_axis(layer_config);
    let mut div = 1;
    for i in 0..inp.shape().len() {
      if i != keep_axis {
        div *= inp.shape()[i];
      }
    }

    let div = F::from(div as u64);
    // FIXME: put this in the fixed column
    let divc = layouter.assign_region(
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

    Ok((divc, div))
  }
}

impl<F: PrimeField> Layer<F> for MeanChip {
  fn forward(
    &self,
    layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    _rand_vector: &HashMap<i64, (CellRc<F>, F)>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let dived = self.avg_forward(layouter, tensors, constants, gadget_config, layer_config)?;

    let out_shape = layer_config.out_shapes[0]
      .iter()
      .map(|x| *x as usize)
      .collect::<Vec<_>>();

    let out = Array::from_shape_vec(IxDyn(&out_shape), dived).unwrap();
    Ok(vec![out])
  }
}

impl GadgetConsumer for MeanChip {
  fn used_gadgets(&self, _layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![
      GadgetType::Adder,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ]
  }
}
