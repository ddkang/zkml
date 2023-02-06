use std::{fs::File, io::BufReader};

use serde_derive::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorMsgpack {
  pub shape: Vec<i64>,
  pub data: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerMsgpack {
  pub layer_type: String,
  pub params: Vec<i64>,
  pub inp_idxes: Vec<i64>,
  pub out_idxes: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMsgpack {
  pub global_sf: i64,
  pub out_idxes: Vec<i64>,
  pub k: i64,
  pub tensors: Vec<TensorMsgpack>,
  pub layers: Vec<LayerMsgpack>,
}

pub fn load_model_msgpack(path: &str) -> ModelMsgpack {
  let file = File::open(path).unwrap();
  let mut reader = BufReader::new(file);
  let model: ModelMsgpack = rmp_serde::from_read(&mut reader).unwrap();
  model
}
