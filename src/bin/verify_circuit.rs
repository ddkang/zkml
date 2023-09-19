use halo2_proofs::halo2curves::bn256::Fr;
use zkml::{
  model::ModelCircuit,
  utils::{loader::load_config_msgpack, proving_kzg::verify_circuit_kzg},
};

fn main() {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let vkey_fname = std::env::args().nth(2).expect("verification key file path");
  let proof_fname = std::env::args().nth(3).expect("proof file path");
  let public_vals_fname = std::env::args().nth(4).expect("public values file path");
  let kzg_or_ipa = std::env::args().nth(5).expect("kzg or ipa");

  if kzg_or_ipa != "kzg" && kzg_or_ipa != "ipa" {
    panic!("Must specify kzg or ipa");
  }
  if kzg_or_ipa == "kzg" {
    let config = load_config_msgpack(&config_fname);
    let circuit = ModelCircuit::<Fr>::generate_from_msgpack(config, false);
    println!("Loaded configuration");
    verify_circuit_kzg(circuit, &vkey_fname, &proof_fname, &public_vals_fname);
  } else {
    // Serialization of the verification key doesn't seem to be supported for IPA
    panic!("Not implemented");
  }
}
