use halo2_proofs::{
  circuit::{AssignedCell, Value},
  halo2curves::ff::PrimeField,
};
use ndarray::{Array, IxDyn};
use num_bigint::BigUint;

use crate::{gadgets::gadget::convert_to_u128, model::PUBLIC_VALS};

// TODO: this is very bad
pub const RAND_START_IDX: i64 = i64::MIN;
pub const NUM_RANDOMS: i64 = 20001;

// Conversion / printing functions
pub fn convert_to_bigint<F: PrimeField>(x: Value<F>) -> BigUint {
  let mut big = Default::default();
  x.map(|x| {
    big = BigUint::from_bytes_le(x.to_repr().as_ref());
  });
  big
}

pub fn convert_pos_int<F: PrimeField>(x: Value<F>) -> i128 {
  let bias = 1 << 60;
  let x_pos = x + Value::known(F::from(bias as u64));
  let mut outp: i128 = 0;
  x_pos.map(|x| {
    let x_pos = convert_to_u128(&x);
    let tmp = x_pos as i128 - bias;
    outp = tmp;
  });
  return outp;
}

pub fn print_pos_int<F: PrimeField>(prefix: &str, x: Value<F>, scale_factor: u64) {
  let tmp = convert_pos_int(x);
  let tmp_float = tmp as f64 / scale_factor as f64;
  println!("{} x: {} ({})", prefix, tmp, tmp_float);
}

pub fn print_assigned_arr<F: PrimeField>(
  prefix: &str,
  arr: &Vec<&AssignedCell<F, F>>,
  scale_factor: u64,
) {
  for (idx, x) in arr.iter().enumerate() {
    print_pos_int(
      &format!("{}[{}]", prefix, idx),
      x.value().map(|x: &F| x.to_owned()),
      scale_factor,
    );
  }
}

// Get the public values
pub fn get_public_values<F: PrimeField>() -> Vec<F> {
  let mut public_vals = vec![];
  for val in PUBLIC_VALS.lock().unwrap().iter() {
    let val = F::from_str_vartime(&val.to_str_radix(10));
    public_vals.push(val.unwrap());
  }
  public_vals
}

// Broadcast
fn shape_dominates(s1: &[usize], s2: &[usize]) -> bool {
  if s1.len() != s2.len() {
    return false;
  }

  for (x1, x2) in s1.iter().zip(s2.iter()) {
    if x1 < x2 {
      return false;
    }
  }

  true
}

// Precondition: s1.len() < s2.len()
fn intermediate_shape(s1: &[usize], s2: &[usize]) -> Vec<usize> {
  let mut res = vec![1; s2.len() - s1.len()];
  for s in s1.iter() {
    res.push(*s);
  }
  res
}

fn final_shape(s1: &[usize], s2: &[usize]) -> Vec<usize> {
  let mut res = vec![];
  for (x1, x2) in s1.iter().zip(s2.iter()) {
    res.push(std::cmp::max(*x1, *x2));
  }
  res
}

pub fn broadcast<G: Clone>(
  x1: &Array<G, IxDyn>,
  x2: &Array<G, IxDyn>,
) -> (Array<G, IxDyn>, Array<G, IxDyn>) {
  if x1.shape() == x2.shape() {
    return (x1.clone(), x2.clone());
  }

  if x1.ndim() == x2.ndim() {
    let s1 = x1.shape();
    let s2 = x2.shape();
    if shape_dominates(s1, s2) {
      return (x1.clone(), x2.broadcast(s1).unwrap().into_owned());
    } else if shape_dominates(x2.shape(), x1.shape()) {
      return (x1.broadcast(s2).unwrap().into_owned(), x2.clone());
    }
  }

  let (tmp1, tmp2) = if x1.ndim() < x2.ndim() {
    (x1, x2)
  } else {
    (x2, x1)
  };

  // tmp1.ndim() < tmp2.ndim()
  let s1 = tmp1.shape();
  let s2 = tmp2.shape();
  let s = intermediate_shape(s1, s2);
  let final_shape = final_shape(s2, s.as_slice());

  let tmp1 = tmp1.broadcast(s.clone()).unwrap().into_owned();
  let tmp1 = tmp1.broadcast(final_shape.as_slice()).unwrap().into_owned();
  let tmp2 = tmp2.broadcast(final_shape.as_slice()).unwrap().into_owned();
  // println!("x1: {:?} x2: {:?}", x1.shape(), x2.shape());
  // println!("s1: {:?} s2: {:?} s: {:?}", s1, s2, s);
  // println!("tmp1 shape: {:?}", tmp1.shape());
  // println!("tmp2 shape: {:?}", tmp2.shape());

  if x1.ndim() < x2.ndim() {
    return (tmp1, tmp2);
  } else {
    return (tmp2, tmp1);
  }
}
