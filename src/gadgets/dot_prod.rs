use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{Advice, Column, ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use crate::gadgets::adder::AdderChip;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type DotProductConfig = GadgetConfig;

pub struct DotProductChip<F: PrimeField> {
  config: Rc<DotProductConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> DotProductChip<F> {
  pub fn construct(config: Rc<DotProductConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn get_input_columns(config: &GadgetConfig) -> Vec<Column<Advice>> {
    let num_inputs = (config.columns.len() - 1) / 2;
    config.columns[0..num_inputs].to_vec()
  }

  pub fn get_weight_columns(config: &GadgetConfig) -> Vec<Column<Advice>> {
    let num_inputs = (config.columns.len() - 1) / 2;
    config.columns[num_inputs..config.columns.len() - 1].to_vec()
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.selector();
    let columns = &gadget_config.columns;

    meta.create_gate("dot product gate", |meta| {
      let s = meta.query_selector(selector);
      let gate_inp = DotProductChip::<F>::get_input_columns(&gadget_config)
        .iter()
        .map(|col| meta.query_advice(*col, Rotation::cur()))
        .collect::<Vec<_>>();
      let gate_weights = DotProductChip::<F>::get_weight_columns(&gadget_config)
        .iter()
        .map(|col| meta.query_advice(*col, Rotation::cur()))
        .collect::<Vec<_>>();
      let gate_output = meta.query_advice(columns[columns.len() - 1], Rotation::cur());

      let res = gate_inp
        .iter()
        .zip(gate_weights)
        .map(|(a, b)| a.clone() * b.clone())
        .fold(Expression::Constant(F::ZERO), |a, b| a + b);

      vec![s * (res - gate_output)]
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::DotProduct, vec![selector]);

    GadgetConfig {
      columns: gadget_config.columns,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for DotProductChip<F> {
  fn name(&self) -> String {
    "dot product".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    self.config.columns.len()
  }

  fn num_inputs_per_row(&self) -> usize {
    (self.config.columns.len() - 1) / 2
  }

  fn num_outputs_per_row(&self) -> usize {
    1
  }

  // The caller is expected to pad the inputs
  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>,F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    assert_eq!(vec_inputs.len(), 2);

    let inp = &vec_inputs[0];
    let weights = &vec_inputs[1];
    assert_eq!(inp.len(), weights.len());
    assert_eq!(inp.len(), self.num_inputs_per_row());
    // println!("Weight: {:?}", weights);

    let zero = &single_inputs[0];

    if self.config.use_selectors {
      let selector = self.config.selectors.get(&GadgetType::DotProduct).unwrap()[0];
      selector.enable(region, row_offset).unwrap();
    }

    let inp_cols = DotProductChip::<F>::get_input_columns(&self.config);
    inp
      .iter()
      .enumerate()
      .map(|(i, cell)| cell.0.copy_advice(|| "", region, inp_cols[i], row_offset))
      .collect::<Result<Vec<_>, _>>()
      .unwrap();

    let weight_cols = DotProductChip::<F>::get_weight_columns(&self.config);
    weights
      .iter()
      .enumerate()
      .map(|(i, cell)| cell.0.copy_advice(|| "", region, weight_cols[i], row_offset))
      .collect::<Result<Vec<_>, _>>()
      .unwrap();

    // All columns need to be assigned
    if self.config.columns.len() % 2 == 0 {
      zero
        .0
        .copy_advice(
          || "",
          region,
          self.config.columns[self.config.columns.len() - 2],
          row_offset,
        )
        .unwrap();
    }

    let e = inp
      .iter()
      .zip(weights.iter())
      .map(|(a, b)|  a.1 * b.1)
      .reduce(|a, b| a + b)
      .unwrap();

    let res = region
      .assign_advice(
        || "",
        self.config.columns[self.config.columns.len() - 1],
        row_offset,
        || Value::known(e),
      )
      .unwrap();

    Ok(vec![(res, e)])
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    assert_eq!(vec_inputs.len(), 2);
    assert_eq!(single_inputs.len(), 1);
    let zero = &single_inputs[0];

    let mut inputs = vec_inputs[0].clone();
    let mut weights = vec_inputs[1].clone();
    while inputs.len() % self.num_inputs_per_row() != 0 {
      inputs.push(*zero);
      weights.push(*zero);
    }

    let outputs = layouter
      .assign_region(
        || "dot prod rows",
        |mut region| {
          let mut outputs = vec![];
          for i in 0..inputs.len() / self.num_inputs_per_row() {
            let inp =
              inputs[i * self.num_inputs_per_row()..(i + 1) * self.num_inputs_per_row()].to_vec();
            let weights =
              weights[i * self.num_inputs_per_row()..(i + 1) * self.num_inputs_per_row()].to_vec();
            let res = self
              .op_row_region(&mut region, i, &vec![inp, weights], &vec![zero.clone()])
              .unwrap();
            outputs.push(res[0].clone());
          }
          Ok(outputs)
        },
      )
      .unwrap();

    let adder_chip = AdderChip::<F>::construct(self.config.clone());
    let tmp = outputs.iter().map(|x| (&x.0, x.1)).collect::<Vec<_>>();
    Ok(
      adder_chip
        .forward(
          layouter.namespace(|| "dot prod adder"),
          &vec![tmp],
          single_inputs,
        )
        .unwrap(),
    )
  }
}
