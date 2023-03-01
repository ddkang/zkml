use std::{
  fs::File,
  io::{BufReader, Write},
  path::Path,
  time::Instant,
};

use halo2_proofs::{
  halo2curves::pasta::{EqAffine, Fp},
  plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
  poly::{
    commitment::{Params, ParamsProver},
    ipa::{
      commitment::{IPACommitmentScheme, ParamsIPA},
      multiopen::ProverIPA,
      strategy::SingleStrategy,
    },
    VerificationStrategy,
  },
  transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
  },
};

use crate::model::ModelCircuit;

use super::loader::ModelMsgpack;

pub fn get_ipa_params(params_dir: &str, degree: u32) -> ParamsIPA<EqAffine> {
  let path = format!("{}/{}.params", params_dir, degree);
  let params_path = Path::new(&path);
  if File::open(&params_path).is_err() {
    let params: ParamsIPA<EqAffine> = ParamsIPA::new(degree);
    let mut buf = Vec::new();

    params.write(&mut buf).expect("Failed to write params");
    let mut file = File::create(&params_path).expect("Failed to create params file");
    file
      .write_all(&buf[..])
      .expect("Failed to write params to file");
  }

  let params_fs = File::open(&params_path).expect("couldn't load params");
  let params: ParamsIPA<EqAffine> =
    Params::read::<_>(&mut BufReader::new(params_fs)).expect("Failed to read params");
  params
}

pub fn time_circuit_ipa(circuit: ModelCircuit<Fp>, config: ModelMsgpack) {
  let rng = rand::thread_rng();
  let start = Instant::now();

  // TODO: generate empty circuits

  let degree = config.k.try_into().unwrap();
  let params = get_ipa_params("./params_ipa", degree);

  let circuit_duration = start.elapsed();
  println!(
    "Time elapsed in params construction: {:?}",
    circuit_duration
  );

  let vk = keygen_vk(&params, &circuit).unwrap();
  let vk_duration = start.elapsed();
  println!(
    "Time elapsed in generating vkey: {:?}",
    vk_duration - circuit_duration
  );

  let pk = keygen_pk(&params, vk, &circuit).unwrap();
  let pk_duration = start.elapsed();
  println!(
    "Time elapsed in generating pkey: {:?}",
    pk_duration - vk_duration
  );

  let fill_duration = start.elapsed();
  // drop(empty_circuit);
  // let proof_circuit = load_model();
  let proof_circuit = circuit;
  println!(
    "Time elapsed in filling circuit: {:?}",
    fill_duration - pk_duration
  );

  let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
  create_proof::<IPACommitmentScheme<EqAffine>, ProverIPA<EqAffine>, _, _, _, _>(
    &params,
    &pk,
    &[proof_circuit],
    &[&[&[]]],
    rng,
    &mut transcript,
  )
  .unwrap();
  let proof = transcript.finalize();
  let proof_duration = start.elapsed();
  println!("Proving time: {:?}", proof_duration - fill_duration);

  let proof_size = {
    let mut folder = std::path::PathBuf::new();
    folder.push("proof_size_check");
    let mut fd = std::fs::File::create(folder.as_path()).unwrap();
    folder.pop();
    fd.write_all(&proof).unwrap();
    fd.metadata().unwrap().len()
  };
  println!("Proof size: {} bytes", proof_size);

  let strategy = SingleStrategy::new(&params);
  let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
  assert!(
    verify_proof(&params, pk.get_vk(), strategy, &[&[&[]]], &mut transcript).is_ok(),
    "proof did not verify"
  );
  let verify_duration = start.elapsed();
  println!("Verifying time: {:?}", verify_duration - proof_duration);
}
