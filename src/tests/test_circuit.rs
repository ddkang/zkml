use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use std::{
  collections::{HashMap, HashSet},
  marker::PhantomData,
  rc::Rc,
  sync::Mutex,
};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
  halo2curves::FieldExt,
  plonk::{Advice, Circuit, Column, ConstraintSystem, Error},
};
use lazy_static::lazy_static;
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    add_pairs::AddPairsChip,
    adder::AdderChip,
    bias_div_round_relu6::BiasDivRoundRelu6Chip,
    dot_prod::DotProductChip,
    gadget::{Gadget, GadgetConfig, GadgetType},
    mul_pairs::MulPairsChip,
    nonlinear::exp::ExpChip,
    nonlinear::{logistic::LogisticGadgetChip, rsqrt::RsqrtGadgetChip},
    sqrt_big::SqrtBigChip,
    squared_diff::SquaredDiffGadgetChip,
    sub_pairs::SubPairsChip,
    var_div::VarDivRoundChip,
  },
  layers::{
    dag::{DAGLayerChip, DAGLayerConfig},
    layer::{Layer, LayerConfig, LayerType},
  },
  utils::loader::{load_model_msgpack, ModelMsgpack},
};

//
// Contains a way to unit-test each of the files
//

// Note this useful idiom: importing names from outer (for mod tests) scope.
use super::*;
use crate::{gadgets::sqrt_big, utils::loader::TensorMsgpack};
use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};

pub const K: u64 = 19;
pub const GLOBAL_SF: i64 = 2 ^ 16;
pub const NUM_COLS: u64 = 6;

lazy_static! {
  pub static ref GADGET_CONFIG: Mutex<GadgetConfig> = Mutex::new(GadgetConfig::default());
}

pub trait TestCircFunc<F: FieldExt> {
  fn compute_tensor(
    layouter: &mut impl Layouter<F>,
    config: &TestConfig<F>,
    tensors: &Vec<ArrayBase<OwnedRepr<AssignedCell<F, F>>, Dim<IxDynImpl>>>,
    hash_map: &HashMap<i64, AssignedCell<F, F>>,
  );
}

#[derive(Clone, Debug)]
pub struct TestCircuit<F: FieldExt, G: TestCircFunc<F>> {
  pub used_gadgets: HashSet<GadgetType>,
  pub tensors: HashMap<i64, Array<Value<F>, IxDyn>>,
  pub _marker: PhantomData<F>,
  pub _func_marker: PhantomData<G>,
}

impl<F: FieldExt, G: TestCircFunc<F>> TestCircuit<F, G> {
  pub fn new(input_tensors: Vec<TensorMsgpack>) -> Self {
    let gadget = &GADGET_CONFIG;
    let cloned_gadget = gadget.lock().unwrap().clone();
    *gadget.lock().unwrap() = GadgetConfig {
      scale_factor: GLOBAL_SF as u64,
      shift_min_val: -(GLOBAL_SF * GLOBAL_SF * 1024),
      div_outp_min_val: -(1 << (K - 1)),
      min_val: -(1 << (K - 1)),
      max_val: (1 << (K - 1)) - 10,
      num_rows: (1 << K) - 10,
      num_cols: NUM_COLS as usize,
      ..cloned_gadget
    };

    let to_value = |x: i64| {
      let bias = 1 << 31;
      let x_pos = x + bias;
      Value::known(F::from(x_pos as u64)) - Value::known(F::from(bias as u64))
    };

    let mut tensors = HashMap::new();

    for flat in input_tensors {
      let value_flat = flat.data.iter().map(|x| to_value(*x)).collect::<Vec<_>>();
      let shape = flat.shape.iter().map(|x| *x as usize).collect::<Vec<_>>();
      let tensor = Array::from_shape_vec(IxDyn(&shape), value_flat).unwrap();
      tensors.insert(flat.idx, tensor);
    }

    let mut used_gadgets = HashSet::new();
    used_gadgets.insert(GadgetType::AddPairs);
    used_gadgets.insert(GadgetType::Adder);
    used_gadgets.insert(GadgetType::BiasDivRoundRelu6);
    used_gadgets.insert(GadgetType::DotProduct);
    used_gadgets.insert(GadgetType::Rsqrt);
    used_gadgets.insert(GadgetType::Exp);
    used_gadgets.insert(GadgetType::Logistic);

    Self {
      used_gadgets: used_gadgets,
      tensors: tensors,
      _marker: PhantomData,
      _func_marker: PhantomData,
    }
  }

  pub fn assign_tensors(
    &self,
    mut layouter: impl Layouter<F>,
    columns: &Vec<Column<Advice>>,
    tensors: &HashMap<i64, Array<Value<F>, IxDyn>>,
  ) -> Result<Vec<Array<AssignedCell<F, F>, IxDyn>>, Error> {
    let tensors = layouter.assign_region(
      || "asssignment",
      |mut region| {
        let mut cell_idx = 0;
        let mut assigned_tensors = vec![];

        for (tensor_idx, tensor) in tensors {
          let tensor_idx = *tensor_idx as usize;
          let mut flat = vec![];
          for val in tensor.iter() {
            let row_idx = cell_idx / columns.len();
            let col_idx = cell_idx % columns.len();
            let cell = region.assign_advice(|| "assignment", columns[col_idx], row_idx, || *val)?;
            flat.push(cell);
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

  // FIXME: assign to public
  pub fn assign_constants(
    &self,
    mut layouter: impl Layouter<F>,
    model_config: &TestConfig<F>,
  ) -> Result<HashMap<i64, AssignedCell<F, F>>, Error> {
    let sf = model_config.gadget_config.scale_factor;
    let min_val = model_config.gadget_config.min_val;
    let max_val = model_config.gadget_config.max_val;

    let constants = layouter.assign_region(
      || "constants",
      |mut region| {
        let mut constants: HashMap<i64, AssignedCell<F, F>> = HashMap::new();
        let zero = region.assign_fixed(
          || "zero",
          model_config.gadget_config.fixed_columns[0],
          0,
          || Value::known(F::zero()),
        )?;
        let one = region.assign_fixed(
          || "one",
          model_config.gadget_config.fixed_columns[0],
          1,
          || Value::known(F::one()),
        )?;
        let sf_cell = region.assign_fixed(
          || "sf",
          model_config.gadget_config.fixed_columns[0],
          2,
          || Value::known(F::from(sf)),
        )?;
        let min_val_cell = region.assign_fixed(
          || "min_val",
          model_config.gadget_config.fixed_columns[0],
          3,
          || Value::known(F::zero() - F::from((-min_val) as u64)),
        )?;
        // TODO: the table goes from min_val to max_val - 1... fix this
        let max_val_cell = region.assign_fixed(
          || "max_val",
          model_config.gadget_config.fixed_columns[0],
          4,
          || Value::known(F::from((max_val - 1) as u64)),
        )?;

        constants.insert(0, zero);
        constants.insert(1, one);
        constants.insert(sf as i64, sf_cell);
        constants.insert(min_val, min_val_cell);
        constants.insert(max_val, max_val_cell);
        Ok(constants)
      },
    )?;
    Ok(constants)
  }
}

impl<F: FieldExt, G: TestCircFunc<F>> Circuit<F> for TestCircuit<F, G> {
  type Config = TestConfig<F>;
  type FloorPlanner = SimpleFloorPlanner;

  fn without_witnesses(&self) -> Self {
    todo!()
  }

  fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
    // FIXME: decide which gadgets to make
    let mut gadget_config = GADGET_CONFIG.lock().unwrap().clone();

    let columns = (0..gadget_config.num_cols)
      .map(|_| meta.advice_column())
      .collect::<Vec<_>>();
    for col in columns.iter() {
      meta.enable_equality(*col);
    }
    gadget_config.columns = columns;

    gadget_config.public_columns = vec![meta.instance_column()];
    meta.enable_equality(gadget_config.public_columns[0]);

    gadget_config.fixed_columns = vec![meta.fixed_column()];
    meta.enable_equality(gadget_config.fixed_columns[0]);

    // FIXME: fix this shit
    gadget_config = AddPairsChip::<F>::configure(meta, gadget_config);
    gadget_config = AdderChip::<F>::configure(meta, gadget_config);
    gadget_config = BiasDivRoundRelu6Chip::<F>::configure(meta, gadget_config);
    gadget_config = DotProductChip::<F>::configure(meta, gadget_config);
    gadget_config = VarDivRoundChip::<F>::configure(meta, gadget_config);
    gadget_config = RsqrtGadgetChip::<F>::configure(meta, gadget_config);
    gadget_config = MulPairsChip::<F>::configure(meta, gadget_config);
    gadget_config = SubPairsChip::<F>::configure(meta, gadget_config);
    gadget_config = ExpChip::<F>::configure(meta, gadget_config);
    gadget_config = LogisticGadgetChip::<F>::configure(meta, gadget_config);
    gadget_config = SquaredDiffGadgetChip::<F>::configure(meta, gadget_config);
    gadget_config = SqrtBigChip::<F>::configure(meta, gadget_config);

    TestConfig {
      gadget_config: gadget_config.into(),
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
        GadgetType::Rsqrt => {
          let chip = RsqrtGadgetChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "rsqrt lookup"))?;
        }
        GadgetType::Exp => {
          let chip = ExpChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "exp lookup"))?;
        }
        GadgetType::Logistic => {
          let chip = LogisticGadgetChip::<F>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "logistic lookup"))?;
        }
        _ => panic!("unsupported gadget"),
      }
    }

    // Assign weights and constants
    let tensors = self.assign_tensors(
      layouter.namespace(|| "assignment"),
      &config.gadget_config.columns,
      &self.tensors,
    )?;
    let constants = self.assign_constants(layouter.namespace(|| "constants"), &config)?;

    G::compute_tensor(&mut layouter, &config, &tensors, &constants);

    Ok(())
  }
}

#[derive(Clone, Debug)]
pub struct TestConfig<F: FieldExt> {
  pub gadget_config: Rc<GadgetConfig>,
  pub _marker: PhantomData<F>,
}
