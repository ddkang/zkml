use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use ndarray::Array;
use zkml::{
  model::ModelCircuit,
  utils::loader::{load_model_msgpack, ModelMsgpack},
};

use npy::NpyData;
use std::io::Read;

fn load_numpy() {
  file = std::fs::File::open("data/pre_last_conv/flowers/0.npy").unwrap()
    .read_to_end(&mut buf).unwrap();
}

fn main() {
  let labels = Array::from_shape_vec(shape, vec![])

  let config_fname = std::env::args().nth(1).expect("config file path");
  let inp_fname = std::env::args().nth(2).expect("input file path");
  let labels_fname = std::env::args().nth(3).expect("labels file path");

  let config: ModelMsgpack = load_model_msgpack(&config_fname, &inp_fname);

  let circuit = ModelCircuit::<Fr>::generate_from_file(&config_fname, &inp_fname);

  let outp = vec![];
  let prover = MockProver::run(config.k.try_into().unwrap(), &circuit, vec![outp.clone()]).unwrap();
  assert_eq!(prover.verify(), Ok(()));
}
