use halo2_proofs::{
  circuit::{AssignedCell, Value},
  halo2curves::FieldExt,
};

use crate::gadgets::gadget::convert_to_u64;

pub fn print_pos_int<F: FieldExt>(prefix: &str, x: Value<F>) {
  let bias = 1 << 40;
  let x_pos = x + Value::known(F::from(bias as u64));
  x_pos.map(|x| {
    let x_pos = convert_to_u64(&x);
    let tmp = x_pos as i64 - bias;
    println!("{} x: {}", prefix, tmp);
  });
}

pub fn print_assigned_arr<F: FieldExt>(prefix: &str, arr: &Vec<AssignedCell<F, F>>) {
  for (idx, x) in arr.iter().enumerate() {
    print_pos_int(
      &format!("{}[{}]", prefix, idx),
      x.value().map(|x: &F| x.to_owned()),
    );
  }
}
