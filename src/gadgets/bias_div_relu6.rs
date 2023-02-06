use std::{collections::HashMap, marker::PhantomData};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use crate::gadgets::gadget::convert_to_u64;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type BiasDivRelu6Config = GadgetConfig;

const NUM_COLS_PER_OP: usize = 5;
const SHIFT_MIN_VAL: i64 = -(1 << 30);

pub struct BiasDivRelu6Chip<F: FieldExt> {
  config: BiasDivRelu6Config,
  _marker: PhantomData<F>,
}

impl<F: FieldExt> BiasDivRelu6Chip<F> {
  pub fn construct(config: BiasDivRelu6Config) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn get_map(
    scale_factor: u64,
    min_val: i64,
    max_val: i64,
    div_outp_min_val: i64,
  ) -> HashMap<i64, i64> {
    let div_val = scale_factor;
    let min_val = min_val;
    let max_val = max_val;
    let div_outp_min_val = div_outp_min_val;
    let range = max_val - min_val;

    let mut map = HashMap::new();
    for i in 0..range {
      let shifted = i + div_outp_min_val;
      let val = shifted.clamp(0, 6 * div_val as i64);
      map.insert(i as i64, val);
    }
    map
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
      for op_idx in 0..columns.len() / NUM_COLS_PER_OP {
        let offset = op_idx * NUM_COLS_PER_OP;
        let inp = meta.query_advice(columns[offset + 0], Rotation::cur());
        let bias = meta.query_advice(columns[offset + 1], Rotation::cur());
        let div_res = meta.query_advice(columns[offset + 2], Rotation::cur());
        let mod_res = meta.query_advice(columns[offset + 3], Rotation::cur());

        constraints.push(s.clone() * (inp - (sf.clone() * (div_res - bias) + mod_res)));
      }

      constraints
    });

    for op_idx in 0..columns.len() / NUM_COLS_PER_OP {
      let offset = op_idx * NUM_COLS_PER_OP;
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
    selectors.insert(GadgetType::DotProduct, vec![selector]);

    let mut tables = gadget_config.tables;
    tables.insert(
      GadgetType::DotProduct,
      vec![mod_lookup, relu_lookup, div_lookup],
    );

    let mut maps = gadget_config.maps;
    let relu_map = Self::get_map(
      gadget_config.scale_factor,
      gadget_config.min_val,
      gadget_config.max_val,
      gadget_config.div_outp_min_val,
    );
    maps.insert(GadgetType::DotProduct, vec![relu_map]);

    GadgetConfig {
      columns,
      selectors,
      tables,
      maps,
      ..gadget_config
    }
  }

  pub fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() / NUM_COLS_PER_OP
  }
}

impl<F: FieldExt> Gadget<F> for BiasDivRelu6Chip<F> {
  fn name(&self) -> String {
    "BiasDivRelu6".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    NUM_COLS_PER_OP
  }

  fn num_outputs_per_row(&self) -> usize {
    self.num_inputs_per_row()
  }

  fn op_row(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<AssignedCell<F, F>>>,
    _single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let div_val = self.config.scale_factor as i64;

    let div_outp_min_val_i64 = -self.config.div_outp_min_val;

    let div_inp_min_val_pos_i64 = -SHIFT_MIN_VAL;
    let div_inp_min_val_pos = F::from(div_inp_min_val_pos_i64 as u64);

    let inp = &vec_inputs[0];
    let bias = &vec_inputs[1];
    assert_eq!(inp.len(), bias.len());
    assert_eq!(inp.len() % self.num_inputs_per_row(), 0);

    let selector = self.config.selectors.get(&GadgetType::DotProduct).unwrap()[0];
    let relu_map = &self.config.maps.get(&GadgetType::DotProduct).unwrap()[0];

    let outp = layouter.assign_region(
      || "",
      |mut region| {
        selector.enable(&mut region, 0)?;

        let mut outp_cells = vec![];
        for (i, (inp, bias)) in inp.iter().zip(bias.iter()).enumerate() {
          let offset = i * NUM_COLS_PER_OP;

          let inp_f = inp.value().map(|x: &F| x.to_owned());
          let bias_f = bias.value().map(|x: &F| {
            let a = *x + div_inp_min_val_pos;
            let a = convert_to_u64(&a) as i64 - div_inp_min_val_pos_i64;
            a
          });
          let div_mod_res = inp_f.map(|x: F| {
            let x_pos = x + div_inp_min_val_pos;
            let inp = convert_to_u64(&x_pos);
            // println!("inp: {:?}, bias: {:?}, x_pos: {:?}", inp, bias, x_pos);
            let div_res = inp as i64 / div_val - (div_inp_min_val_pos_i64 / div_val);
            let mod_res = inp as i64 % div_val;
            // println!("div_res: {:?}, mod_res: {:?}", div_res, mod_res);
            (div_res, mod_res)
          });
          let div_res = div_mod_res.map(|x: (i64, i64)| x.0) + bias_f;
          let mod_res = div_mod_res.map(|x: (i64, i64)| x.1);

          let outp = div_res.map(|x: i64| {
            let mut x_pos = x - div_outp_min_val_i64;
            if !relu_map.contains_key(&(x_pos)) {
              println!("x: {}, x_pos: {}", x, x_pos);
              x_pos = 0;
            }
            let outp_val = relu_map.get(&(x_pos)).unwrap();
            // println!("x: {}, x_pos: {}, outp_val: {}", x, x_pos, outp_val);
            F::from(*outp_val as u64)
          });

          // Assign inp, bias
          inp.copy_advice(|| "", &mut region, self.config.columns[offset + 0], 0)?;
          bias.copy_advice(|| "", &mut region, self.config.columns[offset + 1], 0)?;

          // Assign div_res, mod_res
          let div_res_cell = region
            .assign_advice(
              || "div_res",
              self.config.columns[offset + 2],
              0,
              || {
                div_res.map(|x: i64| {
                  F::from((x - div_outp_min_val_i64) as u64) - F::from(-div_outp_min_val_i64 as u64)
                })
              },
            )
            .unwrap();
          let _mod_res_cell = region
            .assign_advice(
              || "mod_res",
              self.config.columns[offset + 3],
              0,
              || mod_res.map(|x: i64| F::from(x as u64)),
            )
            .unwrap();

          let outp_cell = region
            .assign_advice(
              || "outp",
              self.config.columns[offset + 4],
              0,
              || outp.map(|x: F| x.to_owned()),
            )
            .unwrap();

          // outp_cells.push((outp_cell, div_res_cell));
          outp_cells.push(outp_cell);
          outp_cells.push(div_res_cell);
        }

        Ok(outp_cells)
      },
    )?;

    Ok(outp)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<AssignedCell<F, F>>>,
    single_inputs: &Vec<AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let mut inps = vec_inputs[0].clone();
    let mut biases = vec_inputs[1].clone();

    // Needed to pad: bias - bias = 0
    let default = biases[0].clone();
    while inps.len() % self.num_inputs_per_row() != 0 {
      inps.push(default.clone());
      biases.push(default.clone());
    }

    let res = self.op_aligned_rows(
      layouter.namespace(|| "bias_div_relu6"),
      &vec![inps, biases],
      single_inputs,
    )?;
    Ok(res)
  }
}
