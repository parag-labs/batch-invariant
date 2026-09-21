mod common;

use batch_invariant::flip_case;
use batch_invariant::server::{
    argmax, logits_alone, logits_in_batch, token_depends_on_batch, Model,
};
use common::Lcg;

#[test]
fn model_vocab_counts_rows() {
    let m = Model::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    assert_eq!(m.vocab(), 3);
}

#[test]
fn argmax_first_wins_on_ties() {
    assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
}

#[test]
fn argmax_single_element() {
    assert_eq!(argmax(&[5.0]), 0);
}

#[test]
fn argmax_all_negative() {
    assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
}

#[test]
fn logits_alone_equals_invariant_batched() {
    let mut rng = Lcg::new(7);
    let model = Model::new(rng.mixed_rows(3, 16));
    let x = rng.mixed_row(16);
    let others = rng.mixed_rows(4, 16);
    assert_eq!(
        logits_alone(&model, &x, true),
        logits_in_batch(&model, &x, &others, true)
    );
}

#[test]
fn token_stable_under_invariant_flip_case() {
    let (model, x, others) = flip_case();
    assert!(!token_depends_on_batch(&model, &x, &others, true));
}

#[test]
fn token_moves_under_variant_flip_case() {
    let (model, x, others) = flip_case();
    assert!(token_depends_on_batch(&model, &x, &others, false));
}
