use std::{
  cmp::{max, min},
  collections::{BTreeMap, HashMap},
  marker::PhantomData,
  rc::Rc,
};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::gadget::{GadgetConfig, GadgetType},
  layers::layer::{AssignedTensor, CellRc},
};

const NUM_BITS_PER_FIELD_ELEM: usize = 254;

pub struct PackerConfig<F: PrimeField> {
  pub num_bits_per_elem: usize,
  pub num_elem_per_packed: usize,
  pub num_packed_per_row: usize,
  pub exponents: Vec<F>,
  _marker: PhantomData<F>,
}

pub struct PackerChip<F: PrimeField> {
  pub config: PackerConfig<F>,
}

impl<F: PrimeField> PackerChip<F> {
  pub fn get_exponents(num_bits_per_elem: usize, num_exponents: usize) -> Vec<F> {
    let mul_val = F::from(1 << num_bits_per_elem);
    let mut exponents = vec![F::ONE];
    for _ in 1..num_exponents {
      exponents.push(exponents[exponents.len() - 1] * mul_val);
    }
    exponents
  }

  pub fn construct(num_bits_per_elem: usize, gadget_config: &GadgetConfig) -> PackerConfig<F> {
    let columns = &gadget_config.columns;

    let num_elem_per_packed = if NUM_BITS_PER_FIELD_ELEM / num_bits_per_elem > columns.len() - 1 {
      columns.len() - 1
    } else {
      // TODO: for many columns, pack many in a single row
      NUM_BITS_PER_FIELD_ELEM / num_bits_per_elem
    };
    // println!("column len: {}", columns.len());
    // println!("num_bits_per_elem: {}", num_bits_per_elem);
    // println!("NUM_BITS_PER_FIELD_ELEM: {}", NUM_BITS_PER_FIELD_ELEM);
    // println!("num_elem_per_packed: {}", num_elem_per_packed);

    let num_packed_per_row = max(
      1,
      columns.len() / (num_elem_per_packed * (num_bits_per_elem + 1)),
    );
    // println!("num_packed_per_row: {}", num_packed_per_row);

    let exponents = Self::get_exponents(num_bits_per_elem, num_elem_per_packed);

    let config = PackerConfig {
      num_bits_per_elem,
      num_elem_per_packed,
      num_packed_per_row,
      exponents,
      _marker: PhantomData,
    };
    config
  }

  pub fn configure(
    meta: &mut ConstraintSystem<F>,
    packer_config: PackerConfig<F>,
    gadget_config: GadgetConfig,
  ) -> GadgetConfig {
    let selector = meta.complex_selector();
    let columns = gadget_config.columns;
    let lookup = gadget_config.tables.get(&GadgetType::InputLookup).unwrap()[0];

    let exponents = &packer_config.exponents;

    let num_bits_per_elem = packer_config.num_bits_per_elem;
    let shift_val = 1 << (num_bits_per_elem - 1);
    let shift_val = Expression::Constant(F::from(shift_val as u64));

    meta.create_gate("packer", |meta| {
      let s = meta.query_selector(selector);
      let mut constraints = vec![];
      for i in 0..packer_config.num_packed_per_row {
        let offset = i * (packer_config.num_elem_per_packed + 1);
        let inps = columns[offset..offset + packer_config.num_elem_per_packed]
          .iter()
          .map(|col| meta.query_advice(*col, Rotation::cur()))
          .collect::<Vec<_>>();

        let outp = meta.query_advice(
          columns[offset + packer_config.num_elem_per_packed],
          Rotation::cur(),
        );

        let res = inps
          .into_iter()
          .zip(exponents.iter())
          .map(|(inp, exp)| (inp + shift_val.clone()) * (*exp))
          .fold(Expression::Constant(F::ZERO), |acc, prod| acc + prod);
        constraints.push(s.clone() * (res - outp));
        // constraints.push(s.clone() * Expression::Constant(F::zero()));
      }

      constraints
    });

    // Ensure that the weights/inputs are in the correct range
    for i in 0..packer_config.num_packed_per_row {
      let offset = i * (packer_config.num_elem_per_packed + 1);
      for j in 0..packer_config.num_elem_per_packed {
        meta.lookup("packer lookup", |meta| {
          let s = meta.query_selector(selector);
          let inp = meta.query_advice(columns[offset + j], Rotation::cur());

          vec![(s * (inp + shift_val.clone()), lookup)]
        });
      }
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Packer, vec![selector]);

    GadgetConfig {
      columns,
      selectors,
      ..gadget_config
    }
  }

  pub fn copy_and_pack_row(
    &self,
    mut layouter: impl Layouter<F>,
    gadget_config: Rc<GadgetConfig>,
    cells: Vec<(CellRc<F>, F)>,
    zero: &AssignedCell<F, F>,
  ) -> Result<Vec<CellRc<F>>, Error> {
    let columns = &gadget_config.columns;
    let selector = gadget_config.selectors.get(&GadgetType::Packer).unwrap()[0];

    let num_bits_per_elem = gadget_config.num_bits_per_elem;
    let shift_val = 1 << (num_bits_per_elem - 1);
    let shift_val = F::from(shift_val as u64);

    let outp = layouter.assign_region(
      || "pack row",
      |mut region| {
        if gadget_config.use_selectors {
          selector.enable(&mut region, 0)?;
        }

        let mut packed = vec![];
        for i in 0..self.config.num_packed_per_row {
          let val_offset = i * self.config.num_elem_per_packed;
          let col_offset = i * (self.config.num_elem_per_packed + 1);

          let mut vals = cells
            [val_offset..min(val_offset + self.config.num_elem_per_packed, cells.len())]
            .iter()
            .enumerate()
            .map(|(i, x)| {
              x.0.copy_advice(|| "", &mut region, columns[col_offset + i], 0)
                .unwrap();
              x.0.value().copied()
            })
            .collect::<Vec<_>>();

          let zero_copied = (cells.len()..self.config.num_elem_per_packed)
            .map(|i| {
              zero
                .copy_advice(|| "", &mut region, columns[col_offset + i], 0)
                .unwrap();
              zero.value().copied()
            })
            .collect::<Vec<_>>();
          vals.extend(zero_copied);

          let res = vals.iter().zip(self.config.exponents.iter()).fold(
            Value::known(F::ZERO),
            |acc, (inp, exp)| {
              let res = acc + (*inp + Value::known(shift_val)) * Value::known(*exp);
              res
            },
          );

          let outp = region.assign_advice(
            || "",
            columns[col_offset + self.config.num_elem_per_packed],
            0,
            || res,
          )?;
          packed.push(Rc::new(outp));
        }

        Ok(packed)
      },
    )?;

    Ok(outp)
  }

  pub fn assign_and_pack_row(
    &self,
    mut layouter: impl Layouter<F>,
    gadget_config: Rc<GadgetConfig>,
    values: Vec<&F>,
    zero: &AssignedCell<F, F>,
  ) -> Result<(Vec<CellRc<F>>, Vec<(CellRc<F>, F)>), Error> {
    let columns = &gadget_config.columns;
    let selector = gadget_config.selectors.get(&GadgetType::Packer).unwrap()[0];

    let num_bits_per_elem = gadget_config.num_bits_per_elem;
    let shift_val = 1 << (num_bits_per_elem - 1);
    let shift_val = F::from(shift_val as u64);

    let outp = layouter.assign_region(
      || "pack row",
      |mut region| {
        if gadget_config.use_selectors {
          selector.enable(&mut region, 0)?;
        }

        let mut packed = vec![];
        let mut assigned = vec![];
        for i in 0..self.config.num_packed_per_row {
          let val_offset = i * self.config.num_elem_per_packed;
          let col_offset = i * (self.config.num_elem_per_packed + 1);

          let mut values = values
            [val_offset..min(val_offset + self.config.num_elem_per_packed, values.len())]
            .iter()
            .map(|x| **x)
            .collect::<Vec<_>>();
          let vals = values
            .iter()
            .enumerate()
            .map(|(i, x)| {
              let tmp = region
                .assign_advice(|| "", columns[col_offset + i], 0, || Value::known(*x))
                .unwrap();
              (Rc::new(tmp), *x)
            })
            .collect::<Vec<_>>();
          assigned.extend(vals);

          let zero_vals = (values.len()..self.config.num_elem_per_packed)
            .map(|i| {
              zero
                .copy_advice(|| "", &mut region, columns[col_offset + i], 0)
                .unwrap();
              F::ZERO
            })
            .collect::<Vec<_>>();
          values.extend(zero_vals);

          let res =
            values
              .iter()
              .zip(self.config.exponents.iter())
              .fold(F::ZERO, |acc, (inp, exp)| {
                let res = acc + (*inp + shift_val) * (*exp);
                res
              });

          let outp = region.assign_advice(
            || "",
            columns[col_offset + self.config.num_elem_per_packed],
            0,
            || Value::known(res),
          )?;
          packed.push(Rc::new(outp));
        }

        Ok((packed, assigned))
      },
    )?;

    Ok(outp)
  }

  pub fn assign_and_pack(
    &self,
    mut layouter: impl Layouter<F>,
    gadget_config: Rc<GadgetConfig>,
    constants: &HashMap<i64, CellRc<F>>,
    tensors: &BTreeMap<i64, Array<F, IxDyn>>,
  ) -> Result<(BTreeMap<i64, AssignedTensor<F>>, Vec<CellRc<F>>), Error> {
    let mut values = vec![];
    for (_, tensor) in tensors {
      for value in tensor.iter() {
        values.push(value);
      }
    }

    let mut packed = vec![];
    let mut assigned = vec![];
    let zero = constants.get(&0).unwrap().clone();

    let num_elems_per_row = self.config.num_packed_per_row * self.config.num_elem_per_packed;
    for i in 0..(values.len().div_ceil(num_elems_per_row)) {
      let row =
        values[i * num_elems_per_row..min((i + 1) * num_elems_per_row, values.len())].to_vec();
      let (row_packed, row_assigned) = self
        .assign_and_pack_row(
          layouter.namespace(|| "pack row"),
          gadget_config.clone(),
          row,
          zero.as_ref(),
        )
        .unwrap();
      packed.extend(row_packed);
      assigned.extend(row_assigned);
    }

    let mut assigned_tensors = BTreeMap::new();
    let mut start_idx = 0;
    for (tensor_id, tensor) in tensors {
      let num_el = tensor.len();
      let v = assigned[start_idx..start_idx + num_el].to_vec();
      let new_tensor = Array::from_shape_vec(tensor.raw_dim(), v).unwrap();
      assigned_tensors.insert(*tensor_id, new_tensor);
      start_idx += num_el;
    }

    Ok((assigned_tensors, packed))
  }

  pub fn copy_and_pack(
    &self,
    mut layouter: impl Layouter<F>,
    gadget_config: Rc<GadgetConfig>,
    constants: &HashMap<i64, CellRc<F>>,
    tensors: &BTreeMap<i64, AssignedTensor<F>>,
  ) -> Result<Vec<CellRc<F>>, Error> {
    let mut values = vec![];
    for (_, tensor) in tensors {
      for value in tensor.iter() {
        values.push(value.clone());
      }
    }

    let mut packed = vec![];
    let zero = constants.get(&0).unwrap().clone();

    let num_elems_per_row = self.config.num_packed_per_row * self.config.num_elem_per_packed;
    for i in 0..(values.len().div_ceil(num_elems_per_row)) {
      let row =
        values[i * num_elems_per_row..min((i + 1) * num_elems_per_row, values.len())].to_vec();
      let row_packed = self
        .copy_and_pack_row(
          layouter.namespace(|| "pack row"),
          gadget_config.clone(),
          row,
          zero.as_ref(),
        )
        .unwrap();
      packed.extend(row_packed);
    }

    Ok(packed)
  }
}
