use ark_std::{end_timer, start_timer};
use halo2_proofs::{
    dev::MockProver,
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    plonk::{create_proof, keygen_pk, keygen_vk},
    poly::kzg::{
        commitment::KZGCommitmentScheme,
        multiopen::ProverSHPLONK,
        strategy::SingleStrategy,
    },
    transcript::{
      Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
  
use zkml::{
    model::ModelCircuit,
    utils::{
        proving_kzg::{get_kzg_params, verify_kzg},
        helpers::get_public_values,
    }
    
};

fn main() {
    let config_fname = std::env::args().nth(1).expect("config file path");
    let inp_fname = std::env::args().nth(2).expect("input file path");
    let circuit = ModelCircuit::<Fr>::generate_from_file(&config_fname, &inp_fname);

    
    let rng = rand::thread_rng();
  
    let timer = start_timer!(|| "Setup");
    let degree = circuit.k as u32;
    let params = get_kzg_params("./params_kzg", degree);
    end_timer!(timer);

    let timer = start_timer!(|| "Preprocess");
    let vk_circuit = circuit.clone();
    let vk = keygen_vk(&params, &vk_circuit).unwrap();
    drop(vk_circuit);
    
    let pk_circuit = circuit.clone();
    let pk = keygen_pk(&params, vk, &pk_circuit).unwrap();
    drop(pk_circuit);
    end_timer!(timer);

    let proof_circuit = circuit.clone();
    let _prover = MockProver::run(degree, &proof_circuit, vec![vec![]]).unwrap();
    let public_vals = get_public_values();
  
    let timer = start_timer!(|| "Prove");
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
      &[proof_circuit],
      &[&[&public_vals]],
      rng,
      &mut transcript,
    )
    .unwrap();
    let proof = transcript.finalize();
    end_timer!(timer);

    let timer = start_timer!(|| "Verify");
    let strategy = SingleStrategy::new(&params);
    let transcript_read = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    verify_kzg(
      &params,
      &pk.get_vk(),
      strategy,
      &public_vals,
      transcript_read,
    );
    end_timer!(timer);

}
