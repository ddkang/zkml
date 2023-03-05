use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error},
  poly::Rotation,
};

use crate::gadgets::gadget::convert_to_u64;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

pub struct MaxChip<F: FieldExt> {
  config: Rc<GadgetConfig>,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> MaxChip<F> {
  pub fn construct(config: Rc<GadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.complex_selector();
    let columns = gadget_config.columns;
    let tables = gadget_config.tables;

    let inp_lookup = if tables.contains_key(&GadgetType::BiasDivRoundRelu6) {
      tables.get(&GadgetType::BiasDivRoundRelu6).unwrap()[0]
    } else {
      panic!("currently only supports with BiasDivRoundRelu6");
      #[allow(unreachable_code)]
      meta.lookup_table_column()
    };

    // TODO: need to check that the max is equal to one of the inputs
    for idx in 0..columns.len() - 1 {
      meta.lookup("max", |meta| {
        let s = meta.query_selector(selector);
        let inp = meta.query_advice(columns[idx], Rotation::cur());
        let max = meta.query_advice(columns[columns.len() - 1], Rotation::cur());
        vec![(s.clone() * (max - inp), inp_lookup)]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Max, vec![selector]);

    GadgetConfig {
      columns,
      selectors,
      tables,
      ..gadget_config
    }
  }
}

impl<F: FieldExt> Gadget<F> for MaxChip<F> {
  fn name(&self) -> String {
    "max".to_string()
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
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    _single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert_eq!(vec_inputs.len(), 1);
    let inp = &vec_inputs[0];

    if self.config.use_selectors {
      let selector = self.config.selectors.get(&GadgetType::Max).unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    let _assigned_inp = inp
      .iter()
      .enumerate()
      .map(|(i, cell)| {
        cell
          .copy_advice(|| "", region, self.config.columns[i], row_offset)
          .unwrap()
      })
      .collect::<Vec<_>>();

    let min_val_pos = F::from((-self.config.div_outp_min_val) as u64);

    let vals = inp
      .iter()
      .map(|cell| cell.value().map(|x| convert_to_u64(&(*x + min_val_pos))))
      .collect::<Vec<_>>();
    let max = vals
      .iter()
      .skip(1)
      .fold(vals[0], |a, b| a.zip(*b).map(|(a, b)| a.max(b)));
    let res = region
      .assign_advice(
        || "",
        *self.config.columns.last().unwrap(),
        row_offset,
        || max.map(|x| F::from(x) - min_val_pos),
      )
      .unwrap();

    Ok(vec![res])
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let mut inputs = vec_inputs[0].clone();
    let first = inputs[0];

    while inputs.len() % self.num_inputs_per_row() != 0 {
      inputs.push(first);
    }

    let mut outputs = self.op_aligned_rows(
      layouter.namespace(|| "max forward"),
      &vec![inputs],
      single_inputs,
    )?;
    while outputs.len() != 1 {
      while outputs.len() % self.num_inputs_per_row() != 0 {
        outputs.push(first.clone());
      }
      let tmp = outputs.iter().map(|x| x).collect::<Vec<_>>();
      outputs = self.op_aligned_rows(
        layouter.namespace(|| "max forward"),
        &vec![tmp],
        single_inputs,
      )?;
    }

    Ok(outputs)
  }
}
