use std::{collections::HashMap, marker::PhantomData};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use crate::gadgets::gadget::convert_to_u64;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type BiasDivFloorRelu6Config = GadgetConfig;

const SHIFT_MIN_VAL: i64 = -(1 << 30);

pub struct BiasDivFloorRelu6Chip<F: PrimeField> {
  config: BiasDivFloorRelu6Config,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> BiasDivFloorRelu6Chip<F> {
  pub fn construct(config: BiasDivFloorRelu6Config) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn get_map(scale_factor: u64, num_rows: i64, div_outp_min_val: i64) -> HashMap<i64, i64> {
    let div_val = scale_factor;
    let div_outp_min_val = div_outp_min_val;

    let mut map = HashMap::new();
    for i in 0..num_rows {
      let shifted = i + div_outp_min_val;
      let val = shifted.clamp(0, 6 * div_val as i64);
      map.insert(i as i64, val);
    }
    map
  }

  pub fn num_cols_per_op() -> usize {
    5
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.complex_selector();
    let sf = Expression::Constant(F::from(gadget_config.scale_factor));
    let columns = gadget_config.columns;

    let mod_lookup = meta.lookup_table_column();
    let relu_lookup = meta.lookup_table_column();
    let div_lookup = meta.lookup_table_column();

    meta.create_gate("bias_mul", |meta| {
      let s = meta.query_selector(selector);

      let mut constraints = vec![];
      for op_idx in 0..columns.len() / Self::num_cols_per_op() {
        let offset = op_idx * Self::num_cols_per_op();
        let inp = meta.query_advice(columns[offset + 0], Rotation::cur());
        let bias = meta.query_advice(columns[offset + 1], Rotation::cur());
        let div_res = meta.query_advice(columns[offset + 2], Rotation::cur());
        let mod_res = meta.query_advice(columns[offset + 3], Rotation::cur());

        constraints.push(s.clone() * (inp - (sf.clone() * (div_res - bias) + mod_res)));
      }

      constraints
    });

    for op_idx in 0..columns.len() / Self::num_cols_per_op() {
      let offset = op_idx * Self::num_cols_per_op();
      meta.lookup("bias_div_relu6 lookup", |meta| {
        let s = meta.query_selector(selector);
        let mod_res = meta.query_advice(columns[offset + 3], Rotation::cur());

        // Constrains that the modulus \in [0, DIV_VAL)
        vec![(s.clone() * mod_res.clone(), mod_lookup)]
      });
      meta.lookup("bias_div_relu6 lookup", |meta| {
        let s = meta.query_selector(selector);
        let div = meta.query_advice(columns[offset + 2], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 4], Rotation::cur());
        let div_outp_min_val = Expression::Constant(F::from((-SHIFT_MIN_VAL) as u64));

        // Constrains that output \in [0, 6 * SF]
        vec![
          (s.clone() * outp, relu_lookup),
          (s * (div + div_outp_min_val), div_lookup),
        ]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::BiasDivFloorRelu6, vec![selector]);

    let mut tables = gadget_config.tables;
    tables.insert(
      GadgetType::BiasDivFloorRelu6,
      vec![mod_lookup, relu_lookup, div_lookup],
    );

    let mut maps = gadget_config.maps;
    let relu_map = Self::get_map(
      gadget_config.scale_factor,
      gadget_config.num_rows as i64,
      gadget_config.div_outp_min_val,
    );
    maps.insert(GadgetType::BiasDivFloorRelu6, vec![relu_map]);

    GadgetConfig {
      columns,
      selectors,
      tables,
      maps,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for BiasDivFloorRelu6Chip<F> {
  fn name(&self) -> String {
    "BiasDivRelu6".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    Self::num_cols_per_op()
  }

  fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() / self.num_cols_per_op()
  }

  fn num_outputs_per_row(&self) -> usize {
    self.num_inputs_per_row()
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    _single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let div_val = self.config.scale_factor as i64;

    let div_outp_min_val_i64 = -self.config.div_outp_min_val;

    let div_inp_min_val_pos_i64 = -SHIFT_MIN_VAL;
    let div_inp_min_val_pos = F::from(div_inp_min_val_pos_i64 as u64);

    let inp = &vec_inputs[0];
    let bias = &vec_inputs[1];
    assert_eq!(inp.len(), bias.len());
    assert_eq!(inp.len() % self.num_inputs_per_row(), 0);

    let relu_map = &self
      .config
      .maps
      .get(&GadgetType::BiasDivFloorRelu6)
      .unwrap()[0];

    if self.config.use_selectors {
      let selector = self
        .config
        .selectors
        .get(&GadgetType::BiasDivFloorRelu6)
        .unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    let mut outp_cells = vec![];
    for (i, (inp, bias)) in inp.iter().zip(bias.iter()).enumerate() {
      let offset = i * self.num_cols_per_op();

      let inp_f = inp.1;
      let bias_f = {
        let a = bias.1 + div_inp_min_val_pos;
        let a = convert_to_u64(&a) as i64 - div_inp_min_val_pos_i64;
        a
      };
      let div_mod_res = {
        let x_pos = inp_f + div_inp_min_val_pos;
        let inp = convert_to_u64(&x_pos);
        // println!("inp: {:?}, bias: {:?}, x_pos: {:?}", inp, bias, x_pos);
        let div_res = inp as i64 / div_val - (div_inp_min_val_pos_i64 / div_val);
        let mod_res = inp as i64 % div_val;
        // println!("div_res: {:?}, mod_res: {:?}", div_res, mod_res);
        (div_res, mod_res)
      };
      let div_res = div_mod_res.0 + bias_f;
      let mod_res = div_mod_res.1;

      let outp = {
        let mut x_pos = div_res - div_outp_min_val_i64;
        // if !relu_map.contains_key(&(x_pos)) {
        //   println!("x: {}, x_pos: {}", div_res, x_pos);
        //   x_pos = 0;
        // }
        let outp_val = relu_map.get(&(x_pos)).unwrap();
        // println!("x: {}, x_pos: {}, outp_val: {}", x, x_pos, outp_val);
        F::from(*outp_val as u64)
      };

      // Assign inp, bias
      inp.0.copy_advice(|| "", region, self.config.columns[offset + 0], row_offset)?;
      bias.0.copy_advice(|| "", region, self.config.columns[offset + 1], row_offset)?;

      // Assign div_res, mod_res
      let div_res_cell = region
        .assign_advice(
          || "div_res",
          self.config.columns[offset + 2],
          row_offset,
          || Value::known(F::from((div_res - div_outp_min_val_i64) as u64) - F::from(-div_outp_min_val_i64 as u64))
        )
        .unwrap();
      let _mod_res_cell = region
        .assign_advice(
          || "mod_res",
          self.config.columns[offset + 3],
          row_offset,
          || Value::known(F::from(mod_res as u64)),
        )
        .unwrap();

      let outp_cell = region
        .assign_advice(
          || "outp",
          self.config.columns[offset + 4],
          row_offset,
          || Value::known(outp),
        )
        .unwrap();

      // outp_cells.push((outp_cell, div_res_cell));
      outp_cells.push((outp_cell, outp));
      outp_cells.push((
        div_res_cell, 
        F::from((div_res - div_outp_min_val_i64) as u64) - F::from(-div_outp_min_val_i64 as u64)
      ));
    }

    Ok(outp_cells)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    let mut inps = vec_inputs[0].clone();
    let mut biases = vec_inputs[1].clone();

    // Needed to pad: bias - bias = 0
    let default = biases[0].clone();
    while inps.len() % self.num_inputs_per_row() != 0 {
      inps.push(default);
      biases.push(default);
    }

    let res = self.op_aligned_rows(
      layouter.namespace(|| "bias_div_relu6"),
      &vec![inps, biases],
      single_inputs,
    )?;
    Ok(res)
  }
}
