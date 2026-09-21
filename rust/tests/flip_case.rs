use batch_invariant::flip_case::{flip_case, D_IN};
use batch_invariant::server::{argmax, logits_alone, logits_in_batch};

#[test]
fn flip_case_shape() {
    let (model, x, others) = flip_case();
    assert_eq!(model.vocab(), 2);
    assert_eq!(x.len(), D_IN);
    assert_eq!(others.len(), 3);
    assert!(x.iter().all(|&v| v == 1.0));
}

#[test]
fn variant_flips_the_decoded_token() {
    let (model, x, others) = flip_case();
    assert_eq!(argmax(&logits_alone(&model, &x, false)), 1);
    assert_eq!(argmax(&logits_in_batch(&model, &x, &others, false)), 0);
}

#[test]
fn invariant_keeps_the_token_stable() {
    let (model, x, others) = flip_case();
    assert_eq!(argmax(&logits_alone(&model, &x, true)), 1);
    assert_eq!(argmax(&logits_in_batch(&model, &x, &others, true)), 1);
}
