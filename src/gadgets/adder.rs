use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type AdderConfig = GadgetConfig;

pub struct AdderChip<F: PrimeField> {
  config: Rc<AdderConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> AdderChip<F> {
  pub fn construct(config: Rc<AdderConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.selector();
    let columns = gadget_config.columns;

    meta.create_gate("adder gate", |meta| {
      let s = meta.query_selector(selector);
      let gate_inp = columns[0..columns.len() - 1]
        .iter()
        .map(|col| meta.query_advice(*col, Rotation::cur()))
        .collect::<Vec<_>>();
      let gate_output = meta.query_advice(*columns.last().unwrap(), Rotation::cur());

      let res = gate_inp
        .iter()
        .fold(Expression::Constant(F::ZERO), |a, b| a + b.clone());

      vec![s * (res - gate_output)]
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Adder, vec![selector]);

    GadgetConfig {
      columns,
      selectors,
      ..gadget_config
    }
  }
}

// NOTE: The forward pass of the adder adds _everything_ into one cell
impl<F: PrimeField> Gadget<F> for AdderChip<F> {
  fn name(&self) -> String {
    "adder".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    self.config.columns.len()
  }

  fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() - 1
  }

  fn num_outputs_per_row(&self) -> usize {
    1
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    _single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    assert_eq!(vec_inputs.len(), 1);
    let inp = &vec_inputs[0];

    if self.config.use_selectors {
      let selector = self.config.selectors.get(&GadgetType::Adder).unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    inp
      .iter()
      .enumerate()
      .map(|(i, cell)| cell.0.copy_advice(|| "", region, self.config.columns[i], row_offset))
      .collect::<Result<Vec<_>, _>>()?;

    let e = inp.iter().fold(F::ZERO, |a, b| {
      a + b.1
    });
    let res = region.assign_advice(
      || "",
      *self.config.columns.last().unwrap(),
      row_offset,
      || Value::known(e),
    )?;

    Ok(vec![(res, e)])
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    assert_eq!(single_inputs.len(), 1);

    let mut inputs = vec_inputs[0].clone();
    let zero = single_inputs[0].clone();

    while inputs.len() % self.num_inputs_per_row() != 0 {
      inputs.push(zero);
    }

    let mut outputs = self.op_aligned_rows(
      layouter.namespace(|| "adder forward"),
      &vec![inputs],
      single_inputs,
    )?;
    while outputs.len() != 1 {
      while outputs.len() % self.num_inputs_per_row() != 0 {
        outputs.push((zero.0.clone(), zero.1));
      }
      let tmp = outputs.iter().map(|x| (&x.0, x.1)).collect::<Vec<_>>();
      outputs = self.op_aligned_rows(
        layouter.namespace(|| "adder forward"),
        &vec![tmp],
        single_inputs,
      )?;
    }

    Ok(outputs)
  }
}
