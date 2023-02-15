use std::marker::PhantomData;

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::FieldExt,
  plonk::{Advice, Column, ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use crate::gadgets::adder::AdderChip;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type DotProductConfig = GadgetConfig;

pub struct DotProductChip<F: FieldExt> {
  config: DotProductConfig,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> DotProductChip<F> {
  pub fn construct(config: DotProductConfig) -> Self {
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
        .fold(Expression::Constant(F::zero()), |a, b| {
          a.clone() + b.clone()
        });

      vec![s * (res - gate_output.clone())]
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

impl<F: FieldExt> Gadget<F> for DotProductChip<F> {
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
  fn op_row(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert_eq!(vec_inputs.len(), 2);

    let inp = &vec_inputs[0];
    let weights = &vec_inputs[1];
    assert_eq!(inp.len(), weights.len());
    assert_eq!(inp.len(), self.num_inputs_per_row());

    let selector = self.config.selectors.get(&GadgetType::DotProduct).unwrap()[0];
    let zero = single_inputs[0].clone();

    let output_cell = layouter.assign_region(
      || "",
      |mut region| {
        selector.enable(&mut region, 0)?;

        let inp_cols = DotProductChip::<F>::get_input_columns(&self.config);
        inp
          .iter()
          .enumerate()
          .map(|(i, cell)| cell.copy_advice(|| "", &mut region, inp_cols[i], 0))
          .collect::<Result<Vec<_>, _>>()?;

        let weight_cols = DotProductChip::<F>::get_weight_columns(&self.config);
        weights
          .iter()
          .enumerate()
          .map(|(i, cell)| cell.copy_advice(|| "", &mut region, weight_cols[i], 0))
          .collect::<Result<Vec<_>, _>>()?;

        // All columns need to be assigned
        if self.config.columns.len() % 2 == 0 {
          zero.copy_advice(
            || "",
            &mut region,
            self.config.columns[self.config.columns.len() - 2],
            0,
          )?;
        }

        let e = inp
          .iter()
          .zip(weights.iter())
          .map(|(a, b)| a.value().map(|x: &F| x.to_owned()) * b.value().map(|x: &F| x.to_owned()))
          .reduce(|a, b| a + b)
          .unwrap();

        let res = region.assign_advice(
          || "",
          self.config.columns[self.config.columns.len() - 1],
          0,
          || e,
        );
        Ok(res?)
      },
    )?;

    Ok(vec![output_cell])
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert_eq!(vec_inputs.len(), 2);

    let inp = &vec_inputs[0];
    let weights = &vec_inputs[1];
    assert_eq!(inp.len(), weights.len());
    assert_eq!(inp.len(), self.num_inputs_per_row());

    let selector = self.config.selectors.get(&GadgetType::DotProduct).unwrap()[0];
    let zero = single_inputs[0].clone();

    selector.enable(region, row_offset)?;

    let inp_cols = DotProductChip::<F>::get_input_columns(&self.config);
    inp
      .iter()
      .enumerate()
      .map(|(i, cell)| cell.copy_advice(|| "", region, inp_cols[i], row_offset))
      .collect::<Result<Vec<_>, _>>()?;

    let weight_cols = DotProductChip::<F>::get_weight_columns(&self.config);
    weights
      .iter()
      .enumerate()
      .map(|(i, cell)| cell.copy_advice(|| "", region, weight_cols[i], row_offset))
      .collect::<Result<Vec<_>, _>>()?;

    // All columns need to be assigned
    if self.config.columns.len() % 2 == 0 {
      zero.copy_advice(
        || "",
        region,
        self.config.columns[self.config.columns.len() - 2],
        row_offset,
      )?;
    }

    let e = inp
      .iter()
      .zip(weights.iter())
      .map(|(a, b)| a.value().map(|x: &F| x.to_owned()) * b.value().map(|x: &F| x.to_owned()))
      .reduce(|a, b| a + b)
      .unwrap();

    let res = region.assign_advice(
      || "",
      self.config.columns[self.config.columns.len() - 1],
      row_offset,
      || e,
    )?;

    Ok(vec![res])
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert_eq!(vec_inputs.len(), 2);
    assert_eq!(single_inputs.len(), 1);
    let zero = single_inputs[0].clone();

    let mut inputs = vec_inputs[0].clone();
    let mut weights = vec_inputs[1].clone();
    while inputs.len() % self.num_inputs_per_row() != 0 {
      inputs.push(zero.clone());
      weights.push(zero.clone());
    }

    let mut outputs = vec![];
    for i in 0..inputs.len() / self.num_inputs_per_row() {
      let inp = inputs[i * self.num_inputs_per_row()..(i + 1) * self.num_inputs_per_row()].to_vec();
      let weights =
        weights[i * self.num_inputs_per_row()..(i + 1) * self.num_inputs_per_row()].to_vec();
      let res = self.op_row(
        layouter.namespace(|| "dot prod"),
        &vec![inp, weights],
        &vec![zero.clone()],
      )?;
      outputs.push(res[0].clone());
    }

    let adder_chip = AdderChip::<F>::construct(self.config.clone());
    Ok(adder_chip.forward(
      layouter.namespace(|| "dot prod adder"),
      &vec![outputs],
      single_inputs,
    )?)
  }
}
