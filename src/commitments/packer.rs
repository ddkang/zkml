use std::{
  cmp::{max, min},
  marker::PhantomData,
  rc::Rc,
};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::FieldExt,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use crate::gadgets::gadget::{GadgetConfig, GadgetType};

const NUM_BITS_PER_FIELD_ELEM: usize = 254;

pub struct PackerConfig<F: FieldExt> {
  pub num_bits_per_elem: usize,
  pub num_elem_per_packed: usize,
  pub num_packed_per_row: usize,
  pub exponents: Vec<F>,
  _marker: PhantomData<F>,
}

pub struct PackerChip<F: FieldExt> {
  config: PackerConfig<F>,
  gadget_config: Rc<GadgetConfig>,
}

impl<F: FieldExt> PackerChip<F> {
  pub fn get_exponents(num_bits_per_elem: usize, num_exponents: usize) -> Vec<F> {
    let mut exponents = vec![F::one()];
    for _ in 1..num_exponents {
      exponents.push(*exponents.last().unwrap() * F::from(1 << num_bits_per_elem));
    }
    exponents
  }

  pub fn construct(num_bits_per_elem: usize, gadget_config: Rc<GadgetConfig>) -> Self {
    let columns = &gadget_config.columns;

    let num_elem_per_packed = min(
      columns.len() / (num_bits_per_elem + 1),
      NUM_BITS_PER_FIELD_ELEM / num_bits_per_elem,
    );

    let num_packed_per_row = max(
      1,
      columns.len() / (num_elem_per_packed * (num_bits_per_elem + 1)),
    );

    let exponents = Self::get_exponents(num_bits_per_elem, num_elem_per_packed);

    let config = PackerConfig {
      num_bits_per_elem,
      num_elem_per_packed,
      num_packed_per_row,
      exponents,
      _marker: PhantomData,
    };
    Self {
      config,
      gadget_config,
    }
  }

  pub fn configure(
    meta: &mut ConstraintSystem<F>,
    packer_config: PackerConfig<F>,
    gadget_config: GadgetConfig,
  ) -> GadgetConfig {
    let selector = meta.selector();
    let columns = gadget_config.columns;

    let exponents = packer_config
      .exponents
      .iter()
      .map(|x| Expression::Constant(*x))
      .collect::<Vec<_>>();

    meta.create_gate("packer", |meta| {
      let s = meta.query_selector(selector);
      let mut constraints = vec![];
      for i in 0..packer_config.num_packed_per_row {
        let offset = i * (packer_config.num_bits_per_elem + 1);
        let inps = columns[offset..offset + packer_config.num_elem_per_packed]
          .iter()
          .map(|col| meta.query_advice(*col, Rotation::cur()))
          .collect::<Vec<_>>();
        let outp = meta.query_advice(
          columns[offset + packer_config.num_elem_per_packed],
          Rotation::cur(),
        );

        let res = inps
          .iter()
          .zip(exponents.iter())
          .fold(Expression::Constant(F::zero()), |acc, (inp, exp)| {
            acc + inp.clone() * exp.clone()
          });
        constraints.append(&mut vec![s.clone() * (res - outp)])
      }

      constraints
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Packer, vec![selector]);

    GadgetConfig {
      columns,
      selectors,
      ..gadget_config
    }
  }

  pub fn pack_row(
    &self,
    mut layouter: impl Layouter<F>,
    values: Vec<F>,
    zero: AssignedCell<F, F>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let columns = &self.gadget_config.columns;
    let selector = self
      .gadget_config
      .selectors
      .get(&GadgetType::Packer)
      .unwrap()[0];

    let outp = layouter.assign_region(
      || "pack row",
      |mut region| {
        selector.enable(&mut region, 0)?;

        let mut packed = vec![];
        for i in 0..self.config.num_packed_per_row {
          let val_offset = i * self.config.num_elem_per_packed;
          let col_offset = i * (self.config.num_elem_per_packed + 1);

          let values = values
            [val_offset..min(val_offset + self.config.num_elem_per_packed, values.len())]
            .to_vec();
          let _vals = values
            .iter()
            .enumerate()
            .map(|(i, x)| {
              region.assign_advice(|| "", columns[col_offset + i], 0, || Value::known(*x))
            })
            .collect::<Vec<_>>();

          let _zero = (values.len()..self.config.num_elem_per_packed)
            .map(|i| zero.copy_advice(|| "", &mut region, columns[col_offset + i], 0))
            .collect::<Vec<_>>();

          let res = values
            .iter()
            .zip(self.config.exponents.iter())
            .fold(F::zero(), |acc, (inp, exp)| acc + *inp * *exp);

          let outp = region.assign_advice(
            || "",
            columns[col_offset + self.config.num_elem_per_packed],
            0,
            || Value::known(res),
          )?;
          packed.push(outp);
        }

        Ok(packed)
      },
    )?;

    Ok(outp)
  }

  // The packer takes values, assigns them, and packs them
  pub fn pack(
    &self,
    mut layouter: impl Layouter<F>,
    values: Vec<F>,
    zero: AssignedCell<F, F>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let mut packed = vec![];

    let num_elems_per_row = self.config.num_packed_per_row * self.config.num_elem_per_packed;
    for i in 0..(values.len().div_ceil(num_elems_per_row)) {
      let row =
        values[i * num_elems_per_row..min((i + 1) * num_elems_per_row, values.len())].to_vec();
      packed.append(&mut self.pack_row(layouter.namespace(|| "pack row"), row, zero.clone())?);
    }

    Ok(packed)
  }
}
