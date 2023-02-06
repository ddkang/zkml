use halo2_proofs::halo2curves::pasta::Fp;
use zkml::{
  model::ModelCircuit,
  utils::loader::{load_model_msgpack, ModelMsgpack},
};

fn main() {
  let fname = "/home/ddkang/zkml/mobilenet/data/configs_float/v2_0.35_224.msgpack";
  let config: ModelMsgpack = load_model_msgpack(fname);
  println!("{:?}", config);

  let circuit = ModelCircuit::<Fp>::generate_from_file(fname);
  println!("{:?}", circuit);
}
