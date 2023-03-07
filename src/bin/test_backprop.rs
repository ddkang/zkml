use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use ndarray::Array;
use zkml::{
  model::ModelCircuit,
  utils::loader::{load_model_msgpack, ModelMsgpack},
};

use std::io::Read;

fn main() {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let inp_fname = std::env::args().nth(2).expect("input file path");
  let labels_fname = std::env::args().nth(3).expect("labels file path");

  let config: ModelMsgpack = load_model_msgpack(&config_fname, &inp_fname, Some(&labels_fname));

  let circuit = ModelCircuit::<Fr>::generate_from_file(&config_fname, &inp_fname, Some(&labels_fname), true);

  let outp = vec![];
  let prover = MockProver::run(config.k.try_into().unwrap(), &circuit, vec![outp.clone()]).unwrap();
  assert_eq!(prover.verify(), Ok(()));
}
