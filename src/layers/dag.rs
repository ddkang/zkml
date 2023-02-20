use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::gadget::GadgetConfig,
  layers::{
    arithmetic::add::AddChip,
    arithmetic::{mul::MulChip, sub::SubChip},
    fully_connected::FullyConnectedChip,
    mean::MeanChip,
    noop::NoopChip,
    rsqrt::RsqrtChip,
    squared_diff::SquaredDiffChip,
  },
  utils::helpers::print_assigned_arr,
};

use super::{
  avg_pool_2d::AvgPool2DChip,
  conv2d::Conv2DChip,
  layer::{Layer, LayerConfig, LayerType},
  pad::PadChip,
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
}

// IMPORTANT: Assumes input tensors are in order. Output tensors can be in any order.
impl<F: FieldExt> Layer<F> for DAGLayerChip<F> {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<Array<AssignedCell<F, F>, IxDyn>>,
    constants: &HashMap<i64, AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    // Reveal/commit to weights
    // TODO

    // Reveal/commit to inputs
    // TODO

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
          let add_chip = AddChip::<F>::construct();
          add_chip.forward(
            layouter.namespace(|| "dag add"),
            &vec_inps,
            constants,
            gadget_config.clone(),
          )?
        }
        LayerType::AvgPool2D => {
          let avg_pool_2d_chip = AvgPool2DChip::<F>::construct(layer_config.clone());
          avg_pool_2d_chip.forward(
            layouter.namespace(|| "dag avg pool 2d"),
            &vec_inps,
            constants,
            gadget_config.clone(),
          )?
        }
        LayerType::Conv2D => {
          let conv_2d_chip = Conv2DChip::<F>::construct(layer_config.clone());
          conv_2d_chip.forward(
            layouter.namespace(|| "dag conv 2d"),
            &vec_inps,
            constants,
            gadget_config.clone(),
          )?
        }
        LayerType::FullyConnected => {
          let fc_chip = FullyConnectedChip::<F>::construct(layer_config.clone());
          fc_chip.forward(
            layouter.namespace(|| "dag fully connected"),
            &vec_inps,
            constants,
            gadget_config.clone(),
          )?
        }
        LayerType::Mean => {
          let mean_chip = MeanChip::<F>::construct(layer_config.clone());
          mean_chip.forward(
            layouter.namespace(|| "dag mean"),
            &vec_inps,
            constants,
            gadget_config.clone(),
          )?
        }
        LayerType::Pad => {
          let pad_chip = PadChip::<F>::construct(layer_config.clone());
          pad_chip.forward(
            layouter.namespace(|| "dag pad"),
            &vec_inps,
            constants,
            gadget_config.clone(),
          )?
        }
        LayerType::SquaredDifference => {
          let squared_diff_chip = SquaredDiffChip::<F>::construct();
          squared_diff_chip.forward(
            layouter.namespace(|| "dag squared diff"),
            &vec_inps,
            constants,
            gadget_config.clone(),
          )?
        }
        LayerType::Rsqrt => {
          let rsqrt_chip = RsqrtChip::<F>::construct();
          rsqrt_chip.forward(
            layouter.namespace(|| "dag rsqrt"),
            &vec_inps,
            constants,
            gadget_config.clone(),
          )?
        }
        LayerType::Mul => {
          let mul_chip = MulChip::<F>::construct();
          mul_chip.forward(
            layouter.namespace(|| "dag mul"),
            &vec_inps,
            constants,
            gadget_config.clone(),
          )?
        }
        LayerType::Sub => {
          let sub_chip = SubChip::<F>::construct();
          sub_chip.forward(
            layouter.namespace(|| "dag sub"),
            &vec_inps,
            constants,
            gadget_config.clone(),
          )?
        }
        LayerType::Noop => {
          let noop_chip = NoopChip::<F>::construct(layer_config.clone());
          noop_chip.forward(
            layouter.namespace(|| "dag noop"),
            &vec_inps,
            constants,
            gadget_config.clone(),
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

    let tmp = final_out[0].iter().map(|x| x.clone()).collect::<Vec<_>>();
    print_assigned_arr("final out", &tmp);

    Ok(final_out)
  }
}
