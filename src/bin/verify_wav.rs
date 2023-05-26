use std::fs::File;

use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zkml::{
  model::ModelCircuit,
  utils::{
    helpers::get_public_values,
    loader::{load_config_msgpack, ModelMsgpack, TensorMsgpack},
  },
};

fn main() {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let wav_fname = std::env::args().nth(2).expect("wav file path");

  let mut wav_file = File::open(wav_fname).unwrap();
  let (_header, data) = wav::read(&mut wav_file).unwrap();
  let data = match data {
    wav::BitDepth::Sixteen(data) => data,
    _ => panic!("Unsupported bit depth"),
  };
  let data: Vec<i64> = data.iter().map(|x| *x as i64).collect();

  let base_config = load_config_msgpack(&config_fname);

  let config = ModelMsgpack {
    tensors: vec![TensorMsgpack {
      idx: 0,
      shape: vec![1, data.len().try_into().unwrap()],
      data: data,
    }],
    inp_idxes: vec![0],
    out_idxes: vec![],
    layers: vec![],
    commit_before: Some(vec![]),
    commit_after: Some(vec![vec![0]]),
    ..base_config
  };
  println!("Config: {:?}", config);
  let k = config.k;
  let circuit = ModelCircuit::<Fr>::generate_from_msgpack(config, false);

  let _prover = MockProver::run(k.try_into().unwrap(), &circuit, vec![vec![]]).unwrap();
  let public_vals: Vec<Fr> = get_public_values();
  println!("Public values: {:?}", public_vals);
}
