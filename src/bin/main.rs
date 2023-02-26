use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zkml::{
  model::ModelCircuit,
  utils::{
    loader::{load_model_msgpack, ModelMsgpack},
  },
};

fn main() {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let inp_fname = std::env::args().nth(2).expect("input file path");

  let config: ModelMsgpack = load_model_msgpack(&config_fname, &inp_fname);

  let circuit = ModelCircuit::<Fr>::generate_from_file(&config_fname, &inp_fname);

  let outp = vec![];
  let prover = MockProver::run(config.k.try_into().unwrap(), &circuit, vec![outp.clone()]).unwrap();
  assert_eq!(prover.verify(), Ok(()));

  // time_circuit_kzg(circuit, config);
}
