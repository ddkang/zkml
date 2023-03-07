use std::{fs::File, io::BufReader};

use serde_derive::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorMsgpack {
  pub idx: i64,
  pub shape: Vec<i64>,
  pub data: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelMsgpack {
  pub data: Vec<u64>,
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
  pub labels: Vec<LabelMsgpack>,
  pub layers: Vec<LayerMsgpack>,
  pub use_selectors: Option<bool>,
  pub commit: Option<bool>, // TODO: allow for different kinds of commitments
}

pub fn load_model_msgpack_training(config_path: &str, inp_path: &str, label_path: &str) -> ModelMsgpack {
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
  let label_inp: Vec<LabelMsgpack> = {
    let file = File::open(inp_path).unwrap();
    let mut reader = BufReader::new(file);
    rmp_serde::from_read(&mut reader).unwrap()
  };
  for tensor in inp {
    model.tensors.push(tensor);
  }

  // Default to using selectors, commit if use_selectors is not specified
  if model.use_selectors.is_none() {
    model.use_selectors = Some(true)
  };
  if model.commit.is_none() {
    model.commit = Some(true)
  };
  model
}

pub fn load_model_msgpack(config_path: &str, inp_path: &str, label_path: Option<&str>) -> ModelMsgpack {
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

  if label_path != None {
    let labels: Vec<LabelMsgpack> = {
      let file = File::open(label_path.unwrap()).unwrap();
      let mut reader = BufReader::new(file);
      rmp_serde::from_read(&mut reader).unwrap()
    };
    for label in labels {
      model.labels.push(label);
    }
  }

  // Default to using selectors, commit if use_selectors is not specified
  if model.use_selectors.is_none() {
    model.use_selectors = Some(true)
  };
  if model.commit.is_none() {
    model.commit = Some(true)
  };

  model
}
