use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zkml::{
  model::ModelCircuit,
  utils::{
    helpers::get_public_values,
    loader::{load_model_msgpack, ModelMsgpack},
  },
};

fn main() {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let inp_fname = std::env::args().nth(2).expect("input file path");

  let config: ModelMsgpack = load_model_msgpack(&config_fname, &inp_fname);

  let circuit = ModelCircuit::<Fr>::generate_from_file(&config_fname, &inp_fname);

  let _prover = MockProver::run(config.k.try_into().unwrap(), &circuit, vec![vec![]]).unwrap();
  let public_vals = get_public_values();

  let prover = MockProver::run(config.k.try_into().unwrap(), &circuit, vec![public_vals]).unwrap();
  assert_eq!(prover.verify(), Ok(()));
}
