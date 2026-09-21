mod common;

use batch_invariant::kernels::{
    matmul_invariant, splits_for_batch, sum_flat, sum_split, FIXED_SPLITS,
};
use batch_invariant::server::{logits_alone, logits_in_batch};
use batch_invariant::{flip_case, Model};
use common::Lcg;

#[test]
fn sum_flat_empty_is_zero() {
    assert_eq!(sum_flat(&[]), 0.0);
}

#[test]
fn sum_flat_single_element() {
    assert_eq!(sum_flat(&[42.5]), 42.5);
}

#[test]
fn sum_flat_is_left_to_right() {
    assert_eq!(sum_flat(&[0.1, 0.2, 0.3, 0.4]), ((0.1 + 0.2) + 0.3) + 0.4);
}

#[test]
fn split_differs_from_flat_in_float() {
    let vals = [1e16, 1.0, -1e16, 1.0];
    assert_ne!(sum_split(&vals, 2), sum_flat(&vals));
}

#[test]
fn single_split_is_flat() {
    let vals = [0.1, 0.2, 0.3, 0.4];
    assert_eq!(sum_split(&vals, 1), sum_flat(&vals));
}

#[test]
fn zero_or_negative_splits_is_flat() {
    let vals = [0.1, 0.2, 0.3];
    assert_eq!(sum_split(&vals, 0), sum_flat(&vals));
    assert_eq!(sum_split(&vals, -3), sum_flat(&vals));
}

#[test]
fn sum_split_empty_is_zero() {
    assert_eq!(sum_split(&[], 4), 0.0);
}

#[test]
fn sum_split_more_splits_than_elements() {
    // size = ceil(3/8) = 1, so three singleton partials and empties are skipped.
    assert_eq!(sum_split(&[1.0, 2.0, 3.0], 8), 6.0);
}

#[test]
fn splits_for_batch_known_values() {
    assert_eq!(splits_for_batch(0), 8);
    assert_eq!(splits_for_batch(1), 8);
    assert_eq!(splits_for_batch(2), 4);
    assert_eq!(splits_for_batch(4), 2);
    assert_eq!(splits_for_batch(8), 1);
    assert_eq!(splits_for_batch(9), 1);
}

#[test]
fn splits_for_batch_varies_with_batch_size() {
    assert_ne!(splits_for_batch(1), splits_for_batch(4));
}

#[test]
fn fixed_splits_is_four() {
    assert_eq!(FIXED_SPLITS, 4);
}

#[test]
fn matmul_products_and_shape() {
    let w = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let out = matmul_invariant(&[vec![1.0, 1.0], vec![2.0, 2.0]], &w);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], vec![3.0, 7.0, 11.0]);
}

#[test]
fn invariant_row_identical_alone_and_batched() {
    let mut rng = Lcg::new(1);
    let model = Model::new(rng.mixed_rows(5, 16));
    for trial in 0..200 {
        let x = rng.mixed_row(16);
        let others = rng.mixed_rows(trial % 7, 16);
        let alone = logits_alone(&model, &x, true);
        let batched = logits_in_batch(&model, &x, &others, true);
        assert_eq!(alone, batched, "invariant kernel differed on trial {trial}");
    }
}

#[test]
fn invariant_position_in_batch_does_not_matter() {
    let mut rng = Lcg::new(2);
    let w = rng.mixed_rows(4, 16);
    let x = rng.mixed_row(16);
    let neighbours = rng.mixed_rows(2, 16);
    let mut batch = vec![x.clone()];
    batch.extend(neighbours);
    let first = matmul_invariant(&batch, &w).swap_remove(0);
    let alone = matmul_invariant(&[x], &w).swap_remove(0);
    assert_eq!(first, alone);
}

#[test]
fn variant_row_changes_with_batch() {
    let (model, x, others) = flip_case();
    assert_ne!(
        logits_alone(&model, &x, false),
        logits_in_batch(&model, &x, &others, false)
    );
}
