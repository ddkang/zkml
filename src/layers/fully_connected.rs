use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::FieldExt,
  plonk::{Advice, Column, ConstraintSystem, Error},
};
use ndarray::{Array, Axis, IxDyn};

use crate::gadgets::{
  dot_prod::DotProductChip,
  gadget::{Gadget, GadgetConfig},
  var_div::VarDivRoundChip,
};

use super::layer::{Layer, LayerConfig, LayerType};

pub struct FullyConnectedConfig {
  pub activation: i64, // FIXME: to enum
}

pub struct FullyConnectedChip<F: FieldExt> {
  config: LayerConfig,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> FullyConnectedChip<F> {
  pub fn construct(config: LayerConfig) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(_meta: ConstraintSystem<F>, layer_params: Vec<i64>) -> LayerConfig {
    LayerConfig {
      layer_type: LayerType::FullyConnected,
      layer_params,
      inp_shapes: vec![], // FIXME
      out_shapes: vec![],
    }
  }

  pub fn compute_mm(
    input: &Array<AssignedCell<F, F>, IxDyn>,
    weight: &Array<AssignedCell<F, F>, IxDyn>,
  ) -> Array<Value<F>, IxDyn> {
    assert_eq!(input.ndim(), 2);
    assert_eq!(weight.ndim(), 2);
    assert_eq!(input.shape()[1], weight.shape()[0]);

    let mut outp = vec![];
    for i in 0..input.shape()[0] {
      for j in 0..weight.shape()[1] {
        let mut sum = Value::known(F::zero());
        for k in 0..input.shape()[1] {
          sum = sum
            + input[[i, k]].value().map(|x: &F| x.to_owned())
              * weight[[k, j]].value().map(|x: &F| x.to_owned());
        }
        outp.push(sum);
      }
    }

    let out_shape = [input.shape()[0], weight.shape()[1]];
    Array::from_shape_vec(IxDyn(out_shape.as_slice()), outp).unwrap()
  }

  pub fn assign_array(
    columns: &Vec<Column<Advice>>,
    mut layouter: impl Layouter<F>,
    array: &Array<Value<F>, IxDyn>,
  ) -> Result<Array<AssignedCell<F, F>, IxDyn>, Error> {
    assert_eq!(array.ndim(), 2);

    let outp = layouter.assign_region(
      || "assign array",
      |mut region| {
        let mut outp = vec![];
        for (idx, val) in array.iter().enumerate() {
          let row_idx = idx / columns.len();
          let col_idx = idx % columns.len();
          let cell = region.assign_advice(|| "assign array", columns[col_idx], row_idx, || *val)?;
          outp.push(cell);
        }
        Ok(outp)
      },
    )?;

    let out_shape = [array.shape()[0], array.shape()[1]];
    Ok(Array::from_shape_vec(IxDyn(out_shape.as_slice()), outp).unwrap())
  }

  pub fn random_vector(
    columns: &Vec<Column<Advice>>,
    mut layouter: impl Layouter<F>,
    size: usize,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let outp = layouter.assign_region(
      || "assign random",
      |mut region| {
        let mut outp = vec![];
        for idx in 0..size {
          let row_idx = idx / columns.len();
          let col_idx = idx % columns.len();
          let ru64 = rand::random::<u64>(); // FIXME
          let cell = region.assign_advice(
            || "assign array",
            columns[col_idx],
            row_idx,
            || Value::known(F::from(ru64)),
          )?;
          outp.push(cell);
        }
        Ok(outp)
      },
    )?;

    Ok(outp)
  }
}

impl<F: FieldExt> Layer<F> for FullyConnectedChip<F> {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    let input = &tensors[0];
    let ndim = input.ndim();
    let shape = input.shape().clone();
    let input = if ndim == 2 {
      input.clone().into_owned()
    } else {
      input
        .clone()
        .into_shape(IxDyn(&[shape[1], shape[2]]))
        .unwrap()
    };
    let weight = &tensors[1].t().into_owned();
    let zero = constants.get(&0).unwrap().clone();

    // Compute and assign the result
    let mm_result = Self::compute_mm(&input, weight);
    let mm_result = Self::assign_array(
      &gadget_config.columns,
      layouter.namespace(|| ""),
      &mm_result,
    )?;

    // Generate random vectors
    let r1 = Self::random_vector(
      &gadget_config.columns,
      layouter.namespace(|| ""),
      mm_result.shape()[0],
    )?;
    let r2 = Self::random_vector(
      &gadget_config.columns,
      layouter.namespace(|| ""),
      mm_result.shape()[1],
    )?;

    let dot_prod_chip = DotProductChip::<F>::construct(gadget_config.clone());
    let r1_ref = r1.iter().collect::<Vec<_>>();
    let r2_ref = r2.iter().collect::<Vec<_>>();

    // Compute r1 * result
    let mut r1_res = vec![];
    println!("r1_ref: {:?}", r1_ref.len());
    println!("r2_ref: {:?}", r2_ref.len());
    println!("mm_result: {:?}", mm_result.shape());
    for i in 0..mm_result.shape()[1] {
      let tmp = mm_result.index_axis(Axis(1), i).clone();
      let mm_ci = tmp.iter().collect::<Vec<_>>();
      let r1_res_i = dot_prod_chip.forward(
        layouter.namespace(|| format!("r1_res_{}", i)),
        &vec![mm_ci, r1_ref.clone()],
        &vec![zero.clone()],
      )?;
      r1_res.push(r1_res_i[0].clone());
    }

    // Compute r1 * result * r2
    let r1_res_ref = r1_res.iter().collect::<Vec<_>>();
    let r1_res_r2 = dot_prod_chip.forward(
      layouter.namespace(|| "r1_res_r2"),
      &vec![r1_res_ref, r2_ref.clone()],
      &vec![zero.clone()],
    )?;
    let r1_res_r2 = r1_res_r2[0].clone();
    println!("r1_res_r2: {:?}", r1_res_r2);

    // Compute r1 * input
    let mut r1_input = vec![];
    println!("input: {:?}", input.shape());
    println!("r1_ref: {:?}", r1_ref.len());
    for i in 0..input.shape()[1] {
      let tmp = input.index_axis(Axis(1), i).clone();
      let input_ci = tmp.iter().collect::<Vec<_>>();
      let r1_input_i = dot_prod_chip.forward(
        layouter.namespace(|| format!("r1_input_{}", i)),
        &vec![input_ci, r1_ref.clone()],
        &vec![zero.clone()],
      )?;
      r1_input.push(r1_input_i[0].clone());
    }

    // Compute weight * r2
    let mut weight_r2 = vec![];
    for i in 0..weight.shape()[0] {
      let tmp = weight.index_axis(Axis(0), i).clone();
      let weight_ci = tmp.iter().collect::<Vec<_>>();
      let weight_r2_i = dot_prod_chip.forward(
        layouter.namespace(|| format!("weight_r2_{}", i)),
        &vec![weight_ci, r2_ref.clone()],
        &vec![zero.clone()],
      )?;
      weight_r2.push(weight_r2_i[0].clone());
    }

    // Compute (r1 * input) * (weight * r2)
    let r1_input_ref = r1_input.iter().collect::<Vec<_>>();
    let weight_r2_ref = weight_r2.iter().collect::<Vec<_>>();
    let r1_inp_weight_r2 = dot_prod_chip.forward(
      layouter.namespace(|| "r1_inp_weight_r2"),
      &vec![r1_input_ref, weight_r2_ref],
      &vec![zero.clone()],
    )?;

    let r1_inp_weight_r2 = r1_inp_weight_r2[0].clone();
    println!("r1_inp_weight_r2: {:?}", r1_inp_weight_r2);

    layouter.assign_region(
      || "fc equality check",
      |mut region| {
        let t1 = r1_res_r2.copy_advice(|| "", &mut region, gadget_config.columns[0], 0)?;
        let t2 = r1_inp_weight_r2.copy_advice(|| "", &mut region, gadget_config.columns[0], 1)?;

        region.constrain_equal(t1.cell(), t2.cell())?;

        Ok(())
      },
    )?;

    // TODO: bias, division, activation
    let mm_flat = mm_result.iter().collect::<Vec<_>>();
    let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
    let sf = constants
      .get(&(gadget_config.scale_factor as i64))
      .unwrap()
      .clone();
    let mm_div = var_div_chip.forward(
      layouter.namespace(|| "mm_div"),
      &vec![mm_flat],
      &vec![zero.clone(), sf.clone()],
    )?;

    let mm_result = Array::from_shape_vec(IxDyn(mm_result.shape()), mm_div).unwrap();

    Ok(vec![mm_result])
  }
}
