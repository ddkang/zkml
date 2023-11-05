use std::{
  collections::{BTreeSet, HashMap},
  sync::Arc,
};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::group::ff::PrimeField,
  plonk::{Advice, Column, Error, Fixed, Selector, TableColumn},
};
use num_bigint::{BigUint, ToBigUint};
use num_traits::cast::ToPrimitive;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum GadgetType {
  AddPairs,
  Adder,
  BiasDivRoundRelu6,
  BiasDivFloorRelu6,
  DotProduct,
  Exp,
  Logistic,
  Max,
  Pow,
  Relu,
  Rsqrt,
  Sqrt,
  SqrtBig,
  Square,
  SquaredDiff,
  SubPairs,
  Tanh,
  MulPairs,
  VarDivRound,
  VarDivRoundBig,
  VarDivRoundBig3,
  Packer,      // This is a special case
  InputLookup, // Dummy placeholder for the input lookup
  Update,
}

#[derive(Clone, Debug, Default)]
pub struct GadgetConfig {
  pub used_gadgets: Arc<BTreeSet<GadgetType>>,
  pub columns: Vec<Column<Advice>>,
  pub witness_columns: Vec<Column<Advice>>,
  pub fixed_columns: Vec<Column<Fixed>>,
  pub selectors: HashMap<GadgetType, Vec<Selector>>,
  pub tables: HashMap<GadgetType, Vec<TableColumn>>,
  pub maps: HashMap<GadgetType, Vec<HashMap<i64, i64>>>,
  pub scale_factor: u64,
  pub shift_min_val: i64, // MUST be divisible by 2 * scale_factor
  pub num_rows: usize,
  pub num_cols: usize,
  pub k: usize,
  pub eta: f64,
  pub min_val: i64,
  pub max_val: i64,
  pub div_outp_min_val: i64,
  pub use_selectors: bool,
  pub commit_before: Vec<Vec<i64>>,
  pub commit_after: Vec<Vec<i64>>,
  pub num_bits_per_elem: i64,
}

// TODO: refactor
pub fn convert_to_u64<F: PrimeField>(x: &F) -> u64 {
  let big = BigUint::from_bytes_le(x.to_repr().as_ref());
  let big_digits = big.to_u64_digits();
  // if big_digits.len() > 2 {
  //   println!("big_digits: {:?}", big_digits);
  // }
  if big_digits.len() == 1 {
    big_digits[0] as u64
  } else if big_digits.len() == 0 {
    0
  } else {
    panic!();
  }
}

pub fn convert_to_u128<F: PrimeField>(x: &F) -> u128 {
  let big = BigUint::from_bytes_le(x.to_repr().as_ref());
  big.to_biguint().unwrap().to_u128().unwrap()
}

pub trait Gadget<F: PrimeField> {
  fn name(&self) -> String;

  fn num_cols_per_op(&self) -> usize;

  fn num_inputs_per_row(&self) -> usize;

  fn num_outputs_per_row(&self) -> usize;

  fn load_lookups(&self, _layouter: impl Layouter<F>) -> Result<(), Error> {
    Ok(())
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error>;

  // The caller is required to ensure that the inputs are of the correct length.
  fn op_aligned_rows(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    // Sanity check inputs
    for inp in vec_inputs.iter() {
      assert_eq!(inp.len() % self.num_inputs_per_row(), 0);
    }

    let outputs = layouter.assign_region(
      || format!("gadget {}", self.name()),
      |mut region| {
        let mut outputs = vec![];
        for i in 0..vec_inputs[0].len() / self.num_inputs_per_row() {
          let mut vec_inputs_row = vec![];
          for inp in vec_inputs.iter() {
            vec_inputs_row.push(
              inp[i * self.num_inputs_per_row()..(i + 1) * self.num_inputs_per_row()].to_vec(),
            );
          }
          let row_outputs = self.op_row_region(&mut region, i, &vec_inputs_row, &single_inputs)?;
          assert_eq!(row_outputs.len(), self.num_outputs_per_row());
          outputs.extend(row_outputs);
        }
        Ok(outputs)
      },
    )?;

    Ok(outputs)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<(&AssignedCell<F, F>, F)>>,
    single_inputs: &Vec<(&AssignedCell<F, F>, F)>,
  ) -> Result<Vec<(AssignedCell<F, F>, F)>, Error> {
    self.op_aligned_rows(
      layouter.namespace(|| format!("forward row {}", self.name())),
      vec_inputs,
      single_inputs,
    )
  }
}
