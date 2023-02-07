use halo2_proofs::{dev::MockProver, halo2curves::pasta::Fp};
use zkml::{
  model::ModelCircuit,
  utils::loader::{load_model_msgpack, ModelMsgpack},
};

fn main() {
  let config_fname = "/home/ddkang/zkml/zkml-public/examples/mnist/mnist.msgpack";
  let inp_fname = "/home/ddkang/zkml/zkml-public/examples/mnist/five.bin";
  let config: ModelMsgpack = load_model_msgpack(config_fname, inp_fname);
  println!("{:?}", config);

  let circuit = ModelCircuit::<Fp>::generate_from_file(config_fname, inp_fname);
  println!("{:?}", circuit);

  let outp = vec![];
  let prover = MockProver::run(config.k.try_into().unwrap(), &circuit, vec![outp.clone()]).unwrap();
  assert_eq!(prover.verify(), Ok(()));
}
