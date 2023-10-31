use halo2_proofs::halo2curves::bn256::Fr;
use zkml::{
  model::ModelCircuit,
  utils::bench_kzg::bench_kzg,
};

fn main() {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let inp_fname = std::env::args().nth(2).expect("input file path");
  let step = std::env::args().nth(3).expect("step");
  let circuit = ModelCircuit::<Fr>::generate_from_file(&config_fname, &inp_fname);
  bench_kzg(step, circuit);
}
