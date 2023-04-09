use std::{
  collections::{BTreeMap, BTreeSet, HashMap},
  marker::PhantomData,
  rc::Rc,
  sync::{Arc, Mutex},
};

use halo2_proofs::{
  circuit::{Layouter, SimpleFloorPlanner, Value},
  halo2curves::ff::PrimeField,
  plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance},
};
use lazy_static::lazy_static;
use ndarray::{Array, IxDyn};

use crate::{
  commitments::{commit::Commit, packer::PackerChip, pit_commit::PITCommitChip},
  gadgets::{
    add_pairs::AddPairsChip,
    adder::AdderChip,
    bias_div_round_relu6::BiasDivRoundRelu6Chip,
    dot_prod::DotProductChip,
    gadget::{Gadget, GadgetConfig, GadgetType},
    input_lookup::InputLookupChip,
    max::MaxChip,
    mul_pairs::MulPairsChip,
    nonlinear::{exp::ExpGadgetChip, pow::PowGadgetChip, relu::ReluChip, tanh::TanhGadgetChip},
    nonlinear::{logistic::LogisticGadgetChip, rsqrt::RsqrtGadgetChip, sqrt::SqrtGadgetChip},
    sqrt_big::SqrtBigChip,
    square::SquareGadgetChip,
    squared_diff::SquaredDiffGadgetChip,
    sub_pairs::SubPairsChip,
    update::UpdateGadgetChip,
    var_div::VarDivRoundChip,
    var_div_big::VarDivRoundBigChip,
    var_div_big3::VarDivRoundBig3Chip,
  },
  layers::{
    arithmetic::{add::AddChip, div_var::DivVarChip, mul::MulChip, sub::SubChip},
    avg_pool_2d::AvgPool2DChip,
    batch_mat_mul::BatchMatMulChip,
    conv2d::Conv2DChip,
    dag::{DAGLayerChip, DAGLayerConfig},
    fully_connected::{FullyConnectedChip, FullyConnectedConfig},
    layer::{AssignedTensor, CellRc, GadgetConsumer, Layer, LayerConfig, LayerType},
    logistic::LogisticChip,
    max_pool_2d::MaxPool2DChip,
    mean::MeanChip,
    noop::NoopChip,
    pow::PowChip,
    rsqrt::RsqrtChip,
    shape::{
      broadcast::BroadcastChip, concatenation::ConcatenationChip, mask_neg_inf::MaskNegInfChip,
      pack::PackChip, pad::PadChip, permute::PermuteChip, reshape::ReshapeChip,
      resize_nn::ResizeNNChip, rotate::RotateChip, slice::SliceChip, split::SplitChip,
      transpose::TransposeChip,
    },
    softmax::SoftmaxChip,
    sqrt::SqrtChip,
    square::SquareChip,
    squared_diff::SquaredDiffChip,
    tanh::TanhChip,
    update::UpdateChip,
  },
  utils::{
    helpers::{convert_pos_int, NUM_RANDOMS, RAND_START_IDX},
    loader::{load_model_msgpack, ModelMsgpack},
  },
};

lazy_static! {
  pub static ref GADGET_CONFIG: Mutex<GadgetConfig> = Mutex::new(GadgetConfig::default());
  pub static ref PUBLIC_VALS: Mutex<Vec<i128>> = Mutex::new(vec![]);
}

#[derive(Clone, Debug, Default)]
pub struct ModelCircuit<F: PrimeField> {
  pub used_gadgets: Arc<BTreeSet<GadgetType>>,
  pub dag_config: DAGLayerConfig,
  pub tensors: BTreeMap<i64, Array<F, IxDyn>>,
  pub commit: bool,
  pub k: usize,
  pub inp_idxes: Vec<i64>,
  pub _marker: PhantomData<F>,
}

#[derive(Clone, Debug)]
pub struct ModelConfig<F: PrimeField> {
  pub gadget_config: Rc<GadgetConfig>,
  pub public_col: Column<Instance>,
  pub _marker: PhantomData<F>,
}

impl<F: PrimeField> ModelCircuit<F> {
  pub fn assign_tensors(
    &self,
    mut layouter: impl Layouter<F>,
    columns: &Vec<Column<Advice>>,
    tensors: &BTreeMap<i64, Array<F, IxDyn>>,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let tensors = layouter.assign_region(
      || "asssignment",
      |mut region| {
        let mut cell_idx = 0;
        let mut assigned_tensors = vec![];

        for (tensor_idx, tensor) in tensors.iter() {
          let tensor_idx = *tensor_idx as usize;
          let mut flat = vec![];
          for val in tensor.iter() {
            let row_idx = cell_idx / columns.len();
            let col_idx = cell_idx % columns.len();
            let cell = region
              .assign_advice(
                || "assignment",
                columns[col_idx],
                row_idx,
                || Value::known(*val),
              )
              .unwrap();
            flat.push(Rc::new(cell));
            cell_idx += 1;
          }
          let tensor = Array::from_shape_vec(tensor.shape(), flat).unwrap();
          // TODO: is there a non-stupid way to do this?
          while assigned_tensors.len() <= tensor_idx {
            assigned_tensors.push(tensor.clone());
          }
          assigned_tensors[tensor_idx] = tensor;
        }

        Ok(assigned_tensors)
      },
    )?;

    Ok(tensors)
  }

  pub fn assign_constants(
    &self,
    mut layouter: impl Layouter<F>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<HashMap<i64, CellRc<F>>, Error> {
    let sf = gadget_config.scale_factor;
    let min_val = gadget_config.min_val;
    let max_val = gadget_config.max_val;

    let constants = layouter.assign_region(
      || "constants",
      |mut region| {
        let mut constants: HashMap<i64, CellRc<F>> = HashMap::new();

        let vals = vec![0 as i64, 1, sf as i64, min_val, max_val];
        let shift_val_i64 = -min_val * 2; // FIXME
        let shift_val_f = F::from(shift_val_i64 as u64);
        for (i, val) in vals.iter().enumerate() {
          let cell = region.assign_fixed(
            || format!("constant_{}", i),
            gadget_config.fixed_columns[0],
            i,
            || Value::known(F::from((val + shift_val_i64) as u64) - shift_val_f),
          )?;
          constants.insert(*val, Rc::new(cell));
        }

        // TODO: I've made some very bad life decisions
        // TOOD: read this from the config
        let r_base = F::from(0x123456789abcdef);
        let mut r = r_base.clone();
        for i in 0..NUM_RANDOMS {
          let rand = region.assign_fixed(
            || format!("rand_{}", i),
            gadget_config.fixed_columns[0],
            constants.len(),
            || Value::known(r),
          )?;
          r = r * r_base;
          constants.insert(RAND_START_IDX + (i as i64), Rc::new(rand));
        }

        Ok(constants)
      },
    )?;
    Ok(constants)
  }

  // TODO: for some horrifying reason, assigning to fixed columns causes everything to blow up
  // Currently get around this by assigning to advice columns
  // This is secure because of the equality checks but EXTREMELY STUPID
  pub fn assign_constants2(
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

        let vals = vec![0 as i64, 1, sf as i64, min_val, max_val];
        let shift_val_i64 = -min_val * 2; // FIXME
        let shift_val_f = F::from(shift_val_i64 as u64);
        for (i, val) in vals.iter().enumerate() {
          let assignment_idx = i as usize;
          let row_idx = assignment_idx / gadget_config.columns.len();
          let col_idx = assignment_idx % gadget_config.columns.len();
          let cell = region.assign_advice(
            || format!("constant_{}", i),
            gadget_config.columns[col_idx],
            row_idx,
            || Value::known(F::from((val + shift_val_i64) as u64) - shift_val_f),
          )?;
          constants.insert(*val, Rc::new(cell));
        }

        // TODO: I've made some very bad life decisions
        // TOOD: read this from the config
        let r_base = F::from(0x123456789abcdef);
        let mut r = r_base.clone();
        for i in 0..NUM_RANDOMS {
          let assignment_idx = constants.len();
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

        for (k, v) in fixed_constants.iter() {
          let v2 = constants.get(k).unwrap();
          region.constrain_equal(v.cell(), v2.cell()).unwrap();
        }
        Ok(constants)
      },
    )?;
    Ok(constants)
  }

  pub fn generate_from_file(config_file: &str, inp_file: &str) -> ModelCircuit<F> {
    let config: ModelMsgpack = load_model_msgpack(config_file, inp_file);

    let to_field = |x: i64| {
      let bias = 1 << 31;
      let x_pos = x + bias;
      F::from(x_pos as u64) - F::from(bias as u64)
    };

    let match_layer = |x: &str| match x {
      "AveragePool2D" => LayerType::AvgPool2D,
      "Add" => LayerType::Add,
      "BatchMatMul" => LayerType::BatchMatMul,
      "Broadcast" => LayerType::Broadcast,
      "Concatenation" => LayerType::Concatenation,
      "Conv2D" => LayerType::Conv2D,
      "Div" => LayerType::DivFixed, // TODO: rename to DivFixed
      "DivVar" => LayerType::DivVar,
      "FullyConnected" => LayerType::FullyConnected,
      "Logistic" => LayerType::Logistic,
      "MaskNegInf" => LayerType::MaskNegInf,
      "MaxPool2D" => LayerType::MaxPool2D,
      "Mean" => LayerType::Mean,
      "Mul" => LayerType::Mul,
      "Noop" => LayerType::Noop,
      "Pack" => LayerType::Pack,
      "Pad" => LayerType::Pad,
      "Pow" => LayerType::Pow,
      "Permute" => LayerType::Permute,
      "Reshape" => LayerType::Reshape,
      "ResizeNearestNeighbor" => LayerType::ResizeNN,
      "Rotate" => LayerType::Rotate,
      "Rsqrt" => LayerType::Rsqrt,
      "Slice" => LayerType::Slice,
      "Softmax" => LayerType::Softmax,
      "Split" => LayerType::Split,
      "Sqrt" => LayerType::Sqrt,
      "Square" => LayerType::Square,
      "SquaredDifference" => LayerType::SquaredDifference,
      "Sub" => LayerType::Sub,
      "Tanh" => LayerType::Tanh,
      "Transpose" => LayerType::Transpose,
      "Update" => LayerType::Update,
      _ => panic!("unknown op: {}", x),
    };

    let mut tensors = BTreeMap::new();
    for flat in config.tensors {
      let value_flat = flat.data.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
      let shape = flat.shape.iter().map(|x| *x as usize).collect::<Vec<_>>();
      let tensor = Array::from_shape_vec(IxDyn(&shape), value_flat).unwrap();
      tensors.insert(flat.idx, tensor);
    }

    let i64_to_usize = |x: &Vec<i64>| x.iter().map(|x| *x as usize).collect::<Vec<_>>();

    let mut used_gadgets = BTreeSet::new();

    let dag_config = {
      let ops = config
        .layers
        .iter()
        .map(|layer| {
          let layer_type = match_layer(&layer.layer_type);
          let layer_gadgets = match layer_type {
            LayerType::Add => Box::new(AddChip {}) as Box<dyn GadgetConsumer>,
            LayerType::AvgPool2D => Box::new(AvgPool2DChip {}) as Box<dyn GadgetConsumer>,
            LayerType::BatchMatMul => Box::new(BatchMatMulChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Broadcast => Box::new(BroadcastChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Concatenation => Box::new(ConcatenationChip {}) as Box<dyn GadgetConsumer>,
            LayerType::DivFixed => Box::new(ConcatenationChip {}) as Box<dyn GadgetConsumer>,
            LayerType::DivVar => Box::new(DivVarChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Conv2D => Box::new(Conv2DChip {
              config: LayerConfig::default(),
              _marker: PhantomData::<F>,
            }) as Box<dyn GadgetConsumer>,
            LayerType::FullyConnected => Box::new(FullyConnectedChip {
              config: FullyConnectedConfig { normalize: true },
              _marker: PhantomData::<F>,
            }) as Box<dyn GadgetConsumer>,
            LayerType::Logistic => Box::new(LogisticChip {}) as Box<dyn GadgetConsumer>,
            LayerType::MaskNegInf => Box::new(MaskNegInfChip {}) as Box<dyn GadgetConsumer>,
            LayerType::MaxPool2D => Box::new(MaxPool2DChip {
              marker: PhantomData::<F>,
            }) as Box<dyn GadgetConsumer>,
            LayerType::Mean => Box::new(MeanChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Mul => Box::new(MulChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Noop => Box::new(NoopChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Pack => Box::new(PackChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Pad => Box::new(PadChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Pow => Box::new(PowChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Permute => Box::new(PermuteChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Reshape => Box::new(ReshapeChip {}) as Box<dyn GadgetConsumer>,
            LayerType::ResizeNN => Box::new(ResizeNNChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Rotate => Box::new(RotateChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Rsqrt => Box::new(RsqrtChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Slice => Box::new(SliceChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Softmax => Box::new(SoftmaxChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Split => Box::new(SplitChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Sqrt => Box::new(SqrtChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Square => Box::new(SquareChip {}) as Box<dyn GadgetConsumer>,
            LayerType::SquaredDifference => Box::new(SquaredDiffChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Sub => Box::new(SubChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Tanh => Box::new(TanhChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Transpose => Box::new(TransposeChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Update => Box::new(UpdateChip {}) as Box<dyn GadgetConsumer>,
          }
          .used_gadgets(layer.params.clone());
          for gadget in layer_gadgets {
            used_gadgets.insert(gadget);
          }

          LayerConfig {
            layer_type,
            layer_params: layer.params.clone(),
            inp_shapes: layer.inp_shapes.iter().map(|x| i64_to_usize(x)).collect(),
            out_shapes: layer.out_shapes.iter().map(|x| i64_to_usize(x)).collect(),
            mask: layer.mask.clone(),
          }
        })
        .collect::<Vec<_>>();
      let inp_idxes = config
        .layers
        .iter()
        .map(|layer| i64_to_usize(&layer.inp_idxes))
        .collect::<Vec<_>>();
      let out_idxes = config
        .layers
        .iter()
        .map(|layer| i64_to_usize(&layer.out_idxes))
        .collect::<Vec<_>>();
      let final_out_idxes = config
        .out_idxes
        .iter()
        .map(|x| *x as usize)
        .collect::<Vec<_>>();
      DAGLayerConfig {
        inp_idxes,
        out_idxes,
        ops,
        final_out_idxes,
      }
    };

    // The input lookup is always used
    used_gadgets.insert(GadgetType::InputLookup);
    let used_gadgets = Arc::new(used_gadgets);
    let gadget = &GADGET_CONFIG;
    let cloned_gadget = gadget.lock().unwrap().clone();
    *gadget.lock().unwrap() = GadgetConfig {
      scale_factor: config.global_sf as u64,
      shift_min_val: -(config.global_sf * config.global_sf * (1 << 17)),
      div_outp_min_val: -(1 << (config.k - 1)),
      min_val: -(1 << (config.k - 1)),
      max_val: (1 << (config.k - 1)) - 10,
      k: config.k as usize,
      num_rows: (1 << config.k) - 10 + 1,
      num_cols: config.num_cols as usize,
      used_gadgets: used_gadgets.clone(),
      commit: config.commit.unwrap_or(true),
      use_selectors: config.use_selectors.unwrap_or(true),
      ..cloned_gadget
    };

    ModelCircuit {
      tensors,
      _marker: PhantomData,
      dag_config,
      used_gadgets,
      k: config.k as usize,
      inp_idxes: config.inp_idxes.clone(),
      commit: config.commit.unwrap_or(true),
    }
  }

  pub fn commit(
    &self,
    mut layouter: impl Layouter<F>,
    constants: &HashMap<i64, CellRc<F>>,
    config: &ModelConfig<F>,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    // TODO: sometimes it can be less than k, but k is a good upper bound
    let num_bits = self.k;
    let packer_config = PackerChip::<F>::construct(num_bits, config.gadget_config.as_ref());
    let packer_chip = PackerChip::<F> {
      config: packer_config,
    };
    let (tensor_map, packed) = packer_chip
      .assign_and_pack(
        layouter.namespace(|| "packer"),
        config.gadget_config.clone(),
        constants,
        &self.tensors,
      )
      .unwrap();
    let mut inp_packed = vec![];
    let mut weight_packed = vec![];
    // This is sorted
    for (key, _) in tensor_map.iter() {
      if self.inp_idxes.contains(key) {
        inp_packed.push(packed[*key as usize].clone());
      } else {
        weight_packed.push(packed[*key as usize].clone());
      }
    }

    let smallest_tensor = tensor_map
      .iter()
      .min_by_key(|(_, tensor)| tensor.len())
      .unwrap()
      .1;
    let max_tensor_key = tensor_map
      .iter()
      .max_by_key(|(key, _)| *key)
      .unwrap()
      .0
      .clone();
    let mut tensors = vec![];
    for i in 0..max_tensor_key + 1 {
      let tensor = tensor_map.get(&i).unwrap_or(smallest_tensor);
      tensors.push(tensor.clone());
    }

    let pit_commit_chip = PITCommitChip::<F> {
      marker: PhantomData,
    };

    let zero = constants.get(&0).unwrap().clone();
    // TODO: commitments must be made public
    // TODO: the commitment is done twice to verify the Reed Solomon Fingerprint. Need to do it with the random transcript the second time.
    let _input_commitments1 = pit_commit_chip
      .commit(
        layouter.namespace(|| "commit"),
        config.gadget_config.clone(),
        constants,
        &inp_packed,
        zero.clone(),
      )
      .unwrap();
    let _input_commitments2 = pit_commit_chip
      .commit(
        layouter.namespace(|| "commit"),
        config.gadget_config.clone(),
        constants,
        &inp_packed,
        zero.clone(),
      )
      .unwrap();
    let _weight_commitments1 = pit_commit_chip
      .commit(
        layouter.namespace(|| "commit"),
        config.gadget_config.clone(),
        constants,
        &weight_packed,
        zero.clone(),
      )
      .unwrap();
    let _weight_commitments2 = pit_commit_chip
      .commit(
        layouter.namespace(|| "commit"),
        config.gadget_config.clone(),
        constants,
        &weight_packed,
        zero,
      )
      .unwrap();

    Ok(tensors)
  }
}

impl<F: PrimeField + Ord> Circuit<F> for ModelCircuit<F> {
  type Config = ModelConfig<F>;
  type FloorPlanner = SimpleFloorPlanner;

  fn without_witnesses(&self) -> Self {
    todo!()
  }

  fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
    let mut gadget_config = crate::model::GADGET_CONFIG.lock().unwrap().clone();
    let columns = (0..gadget_config.num_cols)
      .map(|_| meta.advice_column())
      .collect::<Vec<_>>();
    for col in columns.iter() {
      meta.enable_equality(*col);
    }
    gadget_config.columns = columns;

    let public_col = meta.instance_column();
    meta.enable_equality(public_col);

    gadget_config.fixed_columns = vec![meta.fixed_column()];
    meta.enable_equality(gadget_config.fixed_columns[0]);

    // The input lookup is always loaded
    gadget_config = InputLookupChip::<F>::configure(meta, gadget_config);

    let used_gadgets = gadget_config.used_gadgets.clone();
    for gadget_type in used_gadgets.iter() {
      gadget_config = match gadget_type {
        GadgetType::AddPairs => AddPairsChip::<F>::configure(meta, gadget_config),
        GadgetType::Adder => AdderChip::<F>::configure(meta, gadget_config),
        GadgetType::BiasDivRoundRelu6 => BiasDivRoundRelu6Chip::<F>::configure(meta, gadget_config),
        GadgetType::BiasDivFloorRelu6 => panic!(),
        GadgetType::DotProduct => DotProductChip::<F>::configure(meta, gadget_config),
        GadgetType::Exp => ExpGadgetChip::<F>::configure(meta, gadget_config),
        GadgetType::Logistic => LogisticGadgetChip::<F>::configure(meta, gadget_config),
        GadgetType::Max => MaxChip::<F>::configure(meta, gadget_config),
        GadgetType::MulPairs => MulPairsChip::<F>::configure(meta, gadget_config),
        GadgetType::Pow => PowGadgetChip::<F>::configure(meta, gadget_config),
        GadgetType::Relu => ReluChip::<F>::configure(meta, gadget_config),
        GadgetType::Rsqrt => RsqrtGadgetChip::<F>::configure(meta, gadget_config),
        GadgetType::Sqrt => SqrtGadgetChip::<F>::configure(meta, gadget_config),
        GadgetType::SqrtBig => SqrtBigChip::<F>::configure(meta, gadget_config),
        GadgetType::Square => SquareGadgetChip::<F>::configure(meta, gadget_config),
        GadgetType::SquaredDiff => SquaredDiffGadgetChip::<F>::configure(meta, gadget_config),
        GadgetType::SubPairs => SubPairsChip::<F>::configure(meta, gadget_config),
        GadgetType::Tanh => TanhGadgetChip::<F>::configure(meta, gadget_config),
        GadgetType::VarDivRound => VarDivRoundChip::<F>::configure(meta, gadget_config),
        GadgetType::VarDivRoundBig => VarDivRoundBigChip::<F>::configure(meta, gadget_config),
        GadgetType::VarDivRoundBig3 => VarDivRoundBig3Chip::<F>::configure(meta, gadget_config),
        GadgetType::InputLookup => gadget_config, // This is always loaded
        GadgetType::Update => UpdateGadgetChip::<F>::configure(meta, gadget_config),
        GadgetType::Packer => panic!(),
      };
    }

    if gadget_config.commit {
      let packer_config = PackerChip::<F>::construct(gadget_config.k, &gadget_config);
      gadget_config = PackerChip::<F>::configure(meta, packer_config, gadget_config);
    }

    ModelConfig {
      gadget_config: gadget_config.into(),
      public_col,
      _marker: PhantomData,
    }
  }

  fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
    // Assign tables
    let gadget_rc: Rc<GadgetConfig> = config.gadget_config.clone().into();
    for gadget in self.used_gadgets.iter() {
      match gadget {
        GadgetType::AddPairs => {
          let chip = AddPairsChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "add pairs lookup"))?;
        }
        GadgetType::Adder => {
          let chip = AdderChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "adder lookup"))?;
        }
        GadgetType::BiasDivRoundRelu6 => {
          let chip = BiasDivRoundRelu6Chip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "bias div round relu6 lookup"))?;
        }
        GadgetType::DotProduct => {
          let chip = DotProductChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "dot product lookup"))?;
        }
        GadgetType::VarDivRound => {
          let chip = VarDivRoundChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "var div lookup"))?;
        }
        GadgetType::Pow => {
          let chip = PowGadgetChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "pow lookup"))?;
        }
        GadgetType::Relu => {
          let chip = ReluChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "relu lookup"))?;
        }
        GadgetType::Rsqrt => {
          let chip = RsqrtGadgetChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "rsqrt lookup"))?;
        }
        GadgetType::Sqrt => {
          let chip = SqrtGadgetChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "sqrt lookup"))?;
        }
        GadgetType::Tanh => {
          let chip = TanhGadgetChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "tanh lookup"))?;
        }
        GadgetType::Exp => {
          let chip = ExpGadgetChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "exp lookup"))?;
        }
        GadgetType::Logistic => {
          let chip = LogisticGadgetChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "logistic lookup"))?;
        }
        GadgetType::InputLookup => {
          let chip = InputLookupChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "input lookup"))?;
        }
        GadgetType::VarDivRoundBig => {}
        GadgetType::VarDivRoundBig3 => {}
        GadgetType::Max => {}
        GadgetType::MulPairs => {}
        GadgetType::SqrtBig => {}
        GadgetType::Square => {}
        GadgetType::SquaredDiff => {}
        GadgetType::SubPairs => {}
        GadgetType::Update => {}
        _ => panic!("unsupported gadget {:?}", gadget),
      }
    }

    // Assign weights and constants
    let constants_base = self
      .assign_constants(
        layouter.namespace(|| "constants"),
        config.gadget_config.clone(),
      )
      .unwrap();
    // Some halo2 cancer
    let constants = self
      .assign_constants2(
        layouter.namespace(|| "constants 2"),
        config.gadget_config.clone(),
        &constants_base,
      )
      .unwrap();

    let tensors = if self.commit {
      self
        .commit(layouter.namespace(|| "commit"), &constants, &config)
        .unwrap()
    } else {
      self.assign_tensors(
        layouter.namespace(|| "assignment"),
        &config.gadget_config.columns,
        &self.tensors,
      )?
    };

    // Perform the dag
    let dag_chip = DAGLayerChip::<F>::construct(self.dag_config.clone());
    let result = dag_chip.forward(
      layouter.namespace(|| "dag"),
      &tensors,
      &constants,
      config.gadget_config.clone(),
      &LayerConfig::default(),
    )?;

    let mut pub_layouter = layouter.namespace(|| "public");
    let mut total_idx = 0;
    let mut new_public_vals = vec![];
    for tensor in result {
      for cell in tensor.iter() {
        pub_layouter
          .constrain_instance(cell.as_ref().cell(), config.public_col, total_idx)
          .unwrap();
        let val = convert_pos_int(cell.value().map(|x| x.to_owned()));
        new_public_vals.push(val);
        total_idx += 1;
      }
    }
    *PUBLIC_VALS.lock().unwrap() = new_public_vals;

    Ok(())
  }
}
