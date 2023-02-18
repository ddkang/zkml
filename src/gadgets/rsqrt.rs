use std::{collections::HashMap, marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Region},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use super::{
  gadget::{Gadget, GadgetConfig, GadgetType},
  non_linearity::NonLinearGadget,
};

type RsqrtGadgetConfig = GadgetConfig;

const NUM_COLS_PER_OP: usize = 2;

pub struct RsqrtGadgetChip<F: FieldExt> {
  config: Rc<RsqrtGadgetConfig>,
  _marker: PhantomData<F>,
}

// TODO: load lookups
impl<F: FieldExt> RsqrtGadgetChip<F> {
  pub fn construct(config: Rc<RsqrtGadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn get_map(scale_factor: u64, min_val: i64, max_val: i64) -> HashMap<i64, i64> {
    let range = max_val - min_val;

    let mut map = HashMap::new();
    for i in 0..range {
      let shifted = i + min_val;
      let x = (shifted as f64) / (scale_factor as f64);
      let sqrt = x.sqrt();
      let rsqrt = 1.0 / sqrt;
      let rsqrt = (rsqrt * (scale_factor as f64)).round() as i64;
      map.insert(i as i64, rsqrt);
    }
    map
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.complex_selector();
    let columns = gadget_config.columns;

    let mut tables = gadget_config.tables;
    let inp_lookup = if tables.contains_key(&GadgetType::BiasDivRoundRelu6) {
      tables.get(&GadgetType::BiasDivRoundRelu6).unwrap()[0]
    } else {
      meta.lookup_table_column()
    };
    let outp_lookup = meta.lookup_table_column();

    for op_idx in 0..columns.len() / NUM_COLS_PER_OP {
      let offset = op_idx * NUM_COLS_PER_OP;
      meta.lookup("rsqrt lookup", |meta| {
        let s = meta.query_selector(selector);
        let inp = meta.query_advice(columns[offset + 0], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 1], Rotation::cur());
        let shift_val = gadget_config.shift_min_val;
        let shift_val = Expression::Constant(F::from((-shift_val) as u64));

        // Constrains that output \in [0, 6 * SF]
        vec![
          (s.clone() * (inp + shift_val), inp_lookup),
          (s.clone() * outp, outp_lookup),
        ]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Rsqrt, vec![selector]);

    tables.insert(GadgetType::Rsqrt, vec![inp_lookup, outp_lookup]);

    let mut maps = gadget_config.maps;
    let relu_map = Self::get_map(
      gadget_config.scale_factor,
      gadget_config.min_val,
      gadget_config.max_val,
    );
    maps.insert(GadgetType::Rsqrt, vec![relu_map]);

    GadgetConfig {
      columns,
      selectors,
      tables,
      maps,
      ..gadget_config
    }
  }
}

impl<F: FieldExt> NonLinearGadget<F> for RsqrtGadgetChip<F> {
  fn get_map(&self) -> HashMap<i64, i64> {
    self.config.maps.get(&GadgetType::Rsqrt).unwrap()[0].clone()
  }

  fn get_selector(&self) -> halo2_proofs::plonk::Selector {
    self.config.selectors.get(&GadgetType::Rsqrt).unwrap()[0]
  }
}

impl<F: FieldExt> Gadget<F> for RsqrtGadgetChip<F> {
  fn name(&self) -> String {
    "RsqrtGadget".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    NUM_COLS_PER_OP
  }

  fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() / NUM_COLS_PER_OP
  }

  fn num_outputs_per_row(&self) -> usize {
    self.config.columns.len() / NUM_COLS_PER_OP
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    NonLinearGadget::op_row_region(
      self,
      region,
      row_offset,
      vec_inputs,
      single_inputs,
      self.config.clone(),
    )
  }

  fn forward(
    &self,
    layouter: impl halo2_proofs::circuit::Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    NonLinearGadget::forward(self, layouter, vec_inputs, single_inputs)
  }
}
