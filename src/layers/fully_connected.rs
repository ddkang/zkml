use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{Advice, Column, Error},
};
use ndarray::{Array, ArrayView, Axis, IxDyn};

use crate::{
  gadgets::{
    add_pairs::AddPairsChip,
    dot_prod::DotProductChip,
    gadget::{Gadget, GadgetConfig, GadgetType},
    nonlinear::relu::ReluChip,
    var_div::VarDivRoundChip,
  },
  layers::layer::ActivationType,
  utils::helpers::RAND_START_IDX,
};

use super::layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig};

pub struct FullyConnectedConfig {
  pub normalize: bool, // Should be true
}

impl FullyConnectedConfig {
  pub fn construct(normalize: bool) -> Self {
    Self { normalize }
  }
}

pub struct FullyConnectedChip<F: PrimeField> {
  pub _marker: PhantomData<F>,
  pub config: FullyConnectedConfig,
}

impl<F: PrimeField> FullyConnectedChip<F> {
  pub fn compute_mm(
    // input: &AssignedTensor<F>,
    input: &ArrayView<CellRc<F>, IxDyn>,
    weight: &AssignedTensor<F>,
  ) -> Array<Value<F>, IxDyn> {
    assert_eq!(input.ndim(), 2);
    assert_eq!(weight.ndim(), 2);
    assert_eq!(input.shape()[1], weight.shape()[0]);

    let mut outp = vec![];
    for i in 0..input.shape()[0] {
      for j in 0..weight.shape()[1] {
        let mut sum = input[[i, 0]].value().map(|x: &F| *x) * weight[[0, j]].value();
        for k in 1..input.shape()[1] {
          sum = sum + input[[i, k]].value().map(|x: &F| *x) * weight[[k, j]].value();
        }
        outp.push(sum);
      }
    }

    let out_shape = [input.shape()[0], weight.shape()[1]];
    Array::from_shape_vec(IxDyn(out_shape.as_slice()), outp).unwrap()
  }

  pub fn assign_array(
    columns: &Vec<Column<Advice>>,
    region: &mut Region<F>,
    array: &Array<Value<F>, IxDyn>,
  ) -> Result<Array<AssignedCell<F, F>, IxDyn>, Error> {
    assert_eq!(array.ndim(), 2);

    let mut outp = vec![];
    for (idx, val) in array.iter().enumerate() {
      let row_idx = idx / columns.len();
      let col_idx = idx % columns.len();
      let cell = region
        .assign_advice(|| "assign array", columns[col_idx], row_idx, || *val)
        .unwrap();
      outp.push(cell);
    }

    let out_shape = [array.shape()[0], array.shape()[1]];
    Ok(Array::from_shape_vec(IxDyn(out_shape.as_slice()), outp).unwrap())
  }

  pub fn random_vector(
    constants: &HashMap<i64, CellRc<F>>,
    size: usize,
  ) -> Result<Vec<CellRc<F>>, Error> {
    let mut outp = vec![];
    for idx in 0..size {
      let idx = RAND_START_IDX + (idx as i64);
      if !constants.contains_key(&idx) {
        println!("Random vector is too small: {:?}", size);
      }
      let cell = constants.get(&idx).unwrap().clone();
      outp.push(cell);
    }

    Ok(outp)
  }

  fn get_activation(&self, layer_params: &Vec<i64>) -> ActivationType {
    let activation = layer_params[0];
    match activation {
      0 => ActivationType::None,
      1 => ActivationType::Relu,
      _ => panic!("Unsupported activation type for fully connected"),
    }
  }
}

impl<F: PrimeField> Layer<F> for FullyConnectedChip<F> {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    assert!(tensors.len() <= 3);
    let activation = self.get_activation(&layer_config.layer_params);

    let input = &tensors[0];
    let ndim = input.ndim();
    let input = if ndim == 2 {
      ArrayView::from(input)
    } else {
      input.index_axis(Axis(0), 0)
    };
    let weight = &tensors[1].t().into_owned();
    let zero = constants.get(&0).unwrap().as_ref();

    // Compute and assign the result
    let mm_result = layouter
      .assign_region(
        || "compute and assign mm",
        |mut region| {
          let mm_result = Self::compute_mm(&input, weight);
          let mm_result =
            Self::assign_array(&gadget_config.columns, &mut region, &mm_result).unwrap();

          Ok(mm_result)
        },
      )
      .unwrap();

    // Generate random vectors
    let r1 = Self::random_vector(constants, mm_result.shape()[0]).unwrap();
    let r2 = Self::random_vector(constants, mm_result.shape()[1]).unwrap();

    let dot_prod_chip = DotProductChip::<F>::construct(gadget_config.clone());
    let r1_ref = r1.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let r2_ref = r2.iter().map(|x| x.as_ref()).collect::<Vec<_>>();

    // Compute r1 * result
    let mut r1_res = vec![];
    // println!("r1_ref: {:?}", r1_ref.len());
    // println!("r2_ref: {:?}", r2_ref.len());
    // println!("mm_result: {:?}", mm_result.shape());
    for i in 0..mm_result.shape()[1] {
      let tmp = mm_result.index_axis(Axis(1), i);
      let mm_ci = tmp.iter().collect::<Vec<_>>();
      let r1_res_i = dot_prod_chip
        .forward(
          layouter.namespace(|| format!("r1_res_{}", i)),
          &vec![mm_ci, r1_ref.clone()],
          &vec![zero],
        )
        .unwrap();
      r1_res.push(r1_res_i[0].clone());
    }

    // Compute r1 * result * r2
    let r1_res_ref = r1_res.iter().collect::<Vec<_>>();
    let r1_res_r2 = dot_prod_chip
      .forward(
        layouter.namespace(|| "r1_res_r2"),
        &vec![r1_res_ref, r2_ref.clone()],
        &vec![zero],
      )
      .unwrap();
    let r1_res_r2 = r1_res_r2[0].clone();
    // println!("r1_res_r2: {:?}", r1_res_r2);

    // Compute r1 * input
    let mut r1_input = vec![];
    // println!("input: {:?}", input.shape());
    // println!("r1_ref: {:?}", r1_ref.len());
    for i in 0..input.shape()[1] {
      let tmp = input.index_axis(Axis(1), i);
      let input_ci = tmp.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
      let r1_input_i = dot_prod_chip
        .forward(
          layouter.namespace(|| format!("r1_input_{}", i)),
          &vec![input_ci, r1_ref.clone()],
          &vec![zero],
        )
        .unwrap();
      r1_input.push(r1_input_i[0].clone());
    }

    // Compute weight * r2
    let mut weight_r2 = vec![];
    for i in 0..weight.shape()[0] {
      let tmp = weight.index_axis(Axis(0), i);
      let weight_ci = tmp.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
      let weight_r2_i = dot_prod_chip
        .forward(
          layouter.namespace(|| format!("weight_r2_{}", i)),
          &vec![weight_ci, r2_ref.clone()],
          &vec![zero],
        )
        .unwrap();
      weight_r2.push(weight_r2_i[0].clone());
    }

    // Compute (r1 * input) * (weight * r2)
    let r1_input_ref = r1_input.iter().collect::<Vec<_>>();
    let weight_r2_ref = weight_r2.iter().collect::<Vec<_>>();
    let r1_inp_weight_r2 = dot_prod_chip
      .forward(
        layouter.namespace(|| "r1_inp_weight_r2"),
        &vec![r1_input_ref, weight_r2_ref],
        &vec![zero],
      )
      .unwrap();

    let r1_inp_weight_r2 = r1_inp_weight_r2[0].clone();
    // println!("r1_inp_weight_r2: {:?}", r1_inp_weight_r2);

    layouter
      .assign_region(
        || "fc equality check",
        |mut region| {
          let t1 = r1_res_r2
            .copy_advice(|| "", &mut region, gadget_config.columns[0], 0)
            .unwrap();
          let t2 = r1_inp_weight_r2
            .copy_advice(|| "", &mut region, gadget_config.columns[0], 1)
            .unwrap();

          region.constrain_equal(t1.cell(), t2.cell()).unwrap();

          Ok(())
        },
      )
      .unwrap();

    let shape = [mm_result.shape()[0], mm_result.shape()[1]];
    let final_result_flat = if self.config.normalize {
      let mm_flat = mm_result.iter().collect::<Vec<_>>();
      let var_div_chip = VarDivRoundChip::<F>::construct(gadget_config.clone());
      let sf = constants
        .get(&(gadget_config.scale_factor as i64))
        .unwrap()
        .as_ref();
      let mm_div = var_div_chip
        .forward(
          layouter.namespace(|| "mm_div"),
          &vec![mm_flat],
          &vec![zero, sf],
        )
        .unwrap();

      let mm_div = if tensors.len() == 3 {
        let bias = tensors[2].broadcast(shape.clone()).unwrap();
        let bias = bias.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
        let mm_div = mm_div.iter().collect::<Vec<_>>();
        let adder_chip = AddPairsChip::<F>::construct(gadget_config.clone());
        let mm_bias = adder_chip
          .forward(
            layouter.namespace(|| "mm_bias"),
            &vec![mm_div, bias],
            &vec![zero],
          )
          .unwrap();
        mm_bias
      } else {
        mm_div
      };

      let mm_div = if activation == ActivationType::Relu {
        let relu_chip = ReluChip::<F>::construct(gadget_config.clone());
        let mm_div = mm_div.iter().collect::<Vec<_>>();
        let vec_inputs = vec![mm_div];
        relu_chip
          .forward(layouter.namespace(|| "relu"), &vec_inputs, &vec![zero])
          .unwrap()
      } else if activation == ActivationType::None {
        mm_div
      } else {
        panic!("Unsupported activation type");
      };

      mm_div.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>()
    } else {
      mm_result
        .into_iter()
        .map(|x| Rc::new(x))
        .collect::<Vec<_>>()
    };
    let final_result = Array::from_shape_vec(IxDyn(&shape), final_result_flat).unwrap();

    Ok(vec![final_result])
  }
}

impl<F: PrimeField> GadgetConsumer for FullyConnectedChip<F> {
  fn used_gadgets(&self, layer_params: Vec<i64>) -> Vec<crate::gadgets::gadget::GadgetType> {
    let activation = self.get_activation(&layer_params);
    let mut outp = vec![
      GadgetType::Adder,
      GadgetType::AddPairs,
      GadgetType::DotProduct,
      GadgetType::VarDivRound,
      GadgetType::InputLookup,
    ];
    match activation {
      ActivationType::Relu => outp.push(GadgetType::Relu),
      ActivationType::None => (),
      _ => panic!("Unsupported activation type"),
    }
    outp
  }
}
