use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{Layouter, Value},
  halo2curves::FieldExt,
  plonk::Error,
};

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::{
    arithmetic::{add::AddChip, mul::MulChip, sub::SubChip},
    batch_mat_mul::BatchMatMulChip,
    fully_connected::{FullyConnectedChip, FullyConnectedConfig},
    logistic::LogisticChip,
    mean::MeanChip,
    noop::NoopChip,
    rsqrt::RsqrtChip,
    shape::{
      mask_neg_inf::MaskNegInfChip, pad::PadChip, reshape::ReshapeChip, transpose::TransposeChip,
    },
    softmax::SoftmaxChip,
    square::SquareChip,
    squared_diff::SquaredDiffChip,
  },
  utils::helpers::{print_assigned_arr, NUM_RANDOMS, RAND_START_IDX},
};

use super::{
  avg_pool_2d::AvgPool2DChip,
  conv2d::Conv2DChip,
  layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig, LayerType},
};

#[derive(Clone, Debug)]
pub struct DAGLayerConfig {
  pub ops: Vec<LayerConfig>,
  pub inp_idxes: Vec<Vec<usize>>,
  pub out_idxes: Vec<Vec<usize>>,
  pub final_out_idxes: Vec<usize>,
}

pub struct DAGLayerChip<F: FieldExt> {
  dag_config: DAGLayerConfig,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> DAGLayerChip<F> {
  pub fn construct(dag_config: DAGLayerConfig) -> Self {
    Self {
      dag_config,
      _marker: PhantomData,
    }
  }

  // TODO: for some horrifying reason, assigning to fixed columns causes everything to blow up
  // Currently get around this by assigning to advice columns
  // This is secure because of the equality checks but EXTREMELY STUPID
  pub fn assign_constants(
    &self,
    mut layouter: impl Layouter<F>,
    gadget_config: Rc<GadgetConfig>,
    fixed_constants: &HashMap<i64, CellRc<F>>,
  ) -> Result<HashMap<i64, CellRc<F>>, Error> {
    let sf = gadget_config.scale_factor;
    let min_val = gadget_config.min_val;
    let max_val = gadget_config.max_val;

    let constants = layouter.assign_region(
      || "constants",
      |mut region| {
        let mut constants: HashMap<i64, CellRc<F>> = HashMap::new();
        let zero = region.assign_advice(
          || "zero",
          gadget_config.columns[0],
          0,
          || Value::known(F::zero()),
        )?;
        let one = region.assign_advice(
          || "one",
          gadget_config.columns[1],
          0,
          || Value::known(F::one()),
        )?;
        let sf_cell = region.assign_advice(
          || "sf",
          gadget_config.columns[2],
          0,
          || Value::known(F::from(sf)),
        )?;
        let min_val_cell = region.assign_advice(
          || "min_val",
          gadget_config.columns[3],
          0,
          || Value::known(F::zero() - F::from((-min_val) as u64)),
        )?;
        // TODO: the table goes from min_val to max_val - 1... fix this
        let max_val_cell = region.assign_advice(
          || "max_val",
          gadget_config.columns[0],
          4,
          || Value::known(F::from((max_val - 1) as u64)),
        )?;

        // TODO: I've made some very bad life decisions
        // TOOD: read this from the config
        let r_base = F::from(0x123456789abcdef);
        let mut r = r_base.clone();
        for i in 0..NUM_RANDOMS {
          let assignment_idx = (5 + i) as usize;
          let row_idx = assignment_idx / gadget_config.columns.len();
          let col_idx = assignment_idx % gadget_config.columns.len();
          let rand = region.assign_advice(
            || format!("rand_{}", i),
            gadget_config.columns[col_idx],
            row_idx,
            || Value::known(r),
          )?;
          r = r * r_base;
          constants.insert(RAND_START_IDX + (i as i64), Rc::new(rand));
        }

        constants.insert(0, Rc::new(zero));
        constants.insert(1, Rc::new(one));
        constants.insert(sf as i64, Rc::new(sf_cell));
        constants.insert(min_val, Rc::new(min_val_cell));
        constants.insert(max_val, Rc::new(max_val_cell));

        for (k, v) in fixed_constants.iter() {
          // This particular equality check fails... figure out why...
          if (*k) == max_val {
            continue;
          }
          let v2 = constants.get(k).unwrap();
          region.constrain_equal(v.cell(), v2.cell()).unwrap();
        }
        Ok(constants)
      },
    )?;
    Ok(constants)
  }
}

// IMPORTANT: Assumes input tensors are in order. Output tensors can be in any order.
impl<F: FieldExt> Layer<F> for DAGLayerChip<F> {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    _layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    // Reveal/commit to weights
    // TODO

    // Reveal/commit to inputs
    // TODO

    // Some halo2 cancer
    let constants_base = self
      .assign_constants(
        layouter.namespace(|| "constants"),
        gadget_config.clone(),
        constants,
      )
      .unwrap();
    let constants = &constants_base;

    // Tensor map
    let mut tensor_map = HashMap::new();
    for (idx, tensor) in tensors.iter().enumerate() {
      tensor_map.insert(idx, tensor.clone());
    }

    // Compute the dag
    for (layer_idx, layer_config) in self.dag_config.ops.iter().enumerate() {
      let layer_type = &layer_config.layer_type;
      let inp_idxes = &self.dag_config.inp_idxes[layer_idx];
      let out_idxes = &self.dag_config.out_idxes[layer_idx];
      println!(
        "Processing layer {}, type: {:?}, inp_idxes: {:?}, out_idxes: {:?}",
        layer_idx, layer_type, inp_idxes, out_idxes
      );
      println!("{:?}", layer_config.layer_params);
      let vec_inps = inp_idxes
        .iter()
        .map(|idx| tensor_map.get(idx).unwrap().clone())
        .collect::<Vec<_>>();

      let out = match layer_type {
        LayerType::Add => {
          let add_chip = AddChip {};
          add_chip.forward(
            layouter.namespace(|| "dag add"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::AvgPool2D => {
          let avg_pool_2d_chip = AvgPool2DChip {};
          avg_pool_2d_chip.forward(
            layouter.namespace(|| "dag avg pool 2d"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::BatchMatMul => {
          let batch_mat_mul_chip = BatchMatMulChip {};
          batch_mat_mul_chip.forward(
            layouter.namespace(|| "dag batch mat mul"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Conv2D => {
          let conv_2d_chip = Conv2DChip {
            config: layer_config.clone(),
            _marker: PhantomData,
          };
          conv_2d_chip.forward(
            layouter.namespace(|| "dag conv 2d"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::FullyConnected => {
          let fc_chip = FullyConnectedChip {
            _marker: PhantomData,
            config: FullyConnectedConfig::construct(true),
          };
          fc_chip.forward(
            layouter.namespace(|| "dag fully connected"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Softmax => {
          let softmax_chip = SoftmaxChip {};
          softmax_chip.forward(
            layouter.namespace(|| "dag softmax"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Mean => {
          let mean_chip = MeanChip {};
          mean_chip.forward(
            layouter.namespace(|| "dag mean"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Pad => {
          let pad_chip = PadChip {};
          pad_chip.forward(
            layouter.namespace(|| "dag pad"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::SquaredDifference => {
          let squared_diff_chip = SquaredDiffChip {};
          squared_diff_chip.forward(
            layouter.namespace(|| "dag squared diff"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Rsqrt => {
          let rsqrt_chip = RsqrtChip {};
          rsqrt_chip.forward(
            layouter.namespace(|| "dag rsqrt"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Logistic => {
          let logistic_chip = LogisticChip {};
          logistic_chip.forward(
            layouter.namespace(|| "dag logistic"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Mul => {
          let mul_chip = MulChip {};
          mul_chip.forward(
            layouter.namespace(|| "dag mul"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Sub => {
          let sub_chip = SubChip {};
          sub_chip.forward(
            layouter.namespace(|| "dag sub"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Noop => {
          let noop_chip = NoopChip {};
          noop_chip.forward(
            layouter.namespace(|| "dag noop"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Transpose => {
          let transpose_chip = TransposeChip {};
          transpose_chip.forward(
            layouter.namespace(|| "dag transpose"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Reshape => {
          let reshape_chip = ReshapeChip {};
          reshape_chip.forward(
            layouter.namespace(|| "dag reshape"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::MaskNegInf => {
          let mask_neg_inf_chip = MaskNegInfChip {};
          mask_neg_inf_chip.forward(
            layouter.namespace(|| "dag mask neg inf"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
        LayerType::Square => {
          let square_chip = SquareChip {};
          square_chip.forward(
            layouter.namespace(|| "dag square"),
            &vec_inps,
            constants,
            gadget_config.clone(),
            &layer_config,
          )?
        }
      };

      for (idx, tensor_idx) in out_idxes.iter().enumerate() {
        println!("{:?}", out[idx].shape());
        tensor_map.insert(*tensor_idx, out[idx].clone());
      }
      println!();
    }

    let mut final_out = vec![];
    for idx in self.dag_config.final_out_idxes.iter() {
      final_out.push(tensor_map.get(idx).unwrap().clone());
    }

    let tmp = final_out[0].iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    print_assigned_arr("final out", &tmp);
    println!("final out idxes: {:?}", self.dag_config.final_out_idxes);

    Ok(final_out)
  }
}

impl<F: FieldExt> GadgetConsumer for DAGLayerChip<F> {
  // Special case: DAG doesn't do anything
  fn used_gadgets(&self) -> Vec<crate::gadgets::gadget::GadgetType> {
    vec![]
  }
}
