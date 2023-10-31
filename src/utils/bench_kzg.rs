use std::{
    fs::File,
    io::BufReader,
  };
  
use halo2_proofs::{
    dev::MockProver,
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    plonk::{create_proof, keygen_pk, keygen_vk, VerifyingKey, ProvingKey},
    poly::kzg::{
        commitment::KZGCommitmentScheme,
        multiopen::ProverSHPLONK,
        strategy::SingleStrategy,
      },
    transcript::{
      Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
    SerdeFormat,
};

use crate::{model::ModelCircuit, utils::helpers::get_public_values};
use crate::utils::proving_kzg::{serialize, get_kzg_params, verify_kzg};
use serde_derive::{Serialize, Deserialize};
use serde_json;

#[derive(Serialize, Deserialize)]
pub struct PublicVal {
    pub vals: Vec<[u8; 32]>
}


pub fn bench_kzg(step: String, circuit: ModelCircuit<Fr>) {

    if step == "setup" {
        let degree = circuit.k as u32;
        let params = get_kzg_params("./params_kzg", degree);
        let vk_circuit = circuit.clone();
        let vk = keygen_vk(&params, &vk_circuit).unwrap();
        drop(vk_circuit);
        let _ = serialize(&vk.to_bytes(SerdeFormat::RawBytes), "vkey");

        let pk_circuit = circuit.clone();
        let pk = keygen_pk(&params, vk, &pk_circuit).unwrap();
        drop(pk_circuit);        
        let _ = serialize(&pk.to_bytes(SerdeFormat::RawBytes), "pkey");

        let proof_circuit = circuit.clone();
        let _prover = MockProver::run(degree, &proof_circuit, vec![vec![]]).unwrap();
        let public_vals = get_public_values();
        let public_vals_u8_32: Vec<[u8; 32]> = public_vals
        .iter()
        .map(|v: &Fr| v.to_bytes())
        .collect();
        serde_json::to_writer(
            File::create("public_vals").unwrap(), &PublicVal{
                vals: public_vals_u8_32
            }).unwrap();
    } else if step == "prove" {
        let rng = rand::thread_rng();
        let degree = circuit.k as u32;
        let params = get_kzg_params("./params_kzg", degree);
        let pk = ProvingKey::read::<BufReader<File>, ModelCircuit<Fr>>(
            &mut BufReader::new(File::open("pkey").unwrap()),
            SerdeFormat::RawBytes,
            ()
          )
          .unwrap();
        let public_val_raw: PublicVal = serde_json::from_reader(
            File::open("public_vals").unwrap()
            ).unwrap();
        let public_vals = public_val_raw.vals.iter().map(|x| Fr::from_bytes(x).unwrap()).collect::<Vec<Fr>>();

        let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
        create_proof::<
          KZGCommitmentScheme<Bn256>,
          ProverSHPLONK<'_, Bn256>,
          Challenge255<G1Affine>,
          _,
          Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
          ModelCircuit<Fr>,
        >(
          &params,
          &pk,
          &[circuit],
          &[&[&public_vals]],
          rng,
          &mut transcript,
        )
        .unwrap();
        let proof = transcript.finalize();
        let _ = serialize(&proof, "proof");
    } else if step == "verify" {
        let degree = circuit.k as u32;
        let params = get_kzg_params("./params_kzg", degree);
        let proof = std::fs::read("proof").unwrap();
        let strategy = SingleStrategy::new(&params);
        let vk = VerifyingKey::read::<BufReader<File>, ModelCircuit<Fr>>(
            &mut BufReader::new(File::open("vkey").unwrap()),
            SerdeFormat::RawBytes,
            ()
          )
        .unwrap();
        let public_val_raw: PublicVal = serde_json::from_reader(
            File::open("public_vals").unwrap()
            ).unwrap();
        let public_vals = public_val_raw.vals.iter().map(|x| Fr::from_bytes(x).unwrap()).collect::<Vec<Fr>>();
        let transcript_read = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        verify_kzg(
            &params,
            &vk,
            strategy,
            &public_vals,
            transcript_read,
        );

    }
}