use std::{fs::File, io::BufReader};

use serde_derive::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorMsgpack {
  pub idx: i64,
  pub shape: Vec<i64>,
  pub data: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerMsgpack {
  pub layer_type: String,
  pub params: Vec<i64>,
  pub inp_idxes: Vec<i64>,
  pub inp_shapes: Vec<Vec<i64>>,
  pub out_idxes: Vec<i64>,
  pub out_shapes: Vec<Vec<i64>>,
  pub mask: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMsgpack {
  pub global_sf: i64,
  pub k: i64,
  pub num_cols: i64,
  pub inp_idxes: Vec<i64>,
  pub out_idxes: Vec<i64>,
  pub tensors: Vec<TensorMsgpack>,
  pub layers: Vec<LayerMsgpack>,
}

pub fn load_model_msgpack(config_path: &str, inp_path: &str) -> ModelMsgpack {
  let mut model: ModelMsgpack = {
    let file = File::open(config_path).unwrap();
    let mut reader = BufReader::new(file);
    rmp_serde::from_read(&mut reader).unwrap()
  };
  let inp: Vec<TensorMsgpack> = {
    let file = File::open(inp_path).unwrap();
    let mut reader = BufReader::new(file);
    rmp_serde::from_read(&mut reader).unwrap()
  };
  for tensor in inp {
    model.tensors.push(tensor);
  }
  model
}
