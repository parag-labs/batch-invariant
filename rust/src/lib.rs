//! Batch-invariant reduction kernels (Rust port).
//!
//! A matmul output is a dot product: a sum over the K dimension. In exact
//! arithmetic the order of that sum does not matter. In floating point it does —
//! `(a + b) + c` can differ from `a + (b + c)` in the last bit, and far more
//! when large terms nearly cancel. Production kernels split the K reduction into
//! chunks and combine partial sums, and how many chunks is often chosen from the
//! batch shape. So a row's result can change depending on who else is in the
//! batch — a real source of non-reproducible LLM inference.
//!
//! [`kernels`] provides the reductions and the variant/invariant matmuls;
//! [`server`] wraps them in a toy decode step; [`flip_case`] is the deterministic
//! fixture where the variant kernel flips a decoded token.

pub mod flip_case;
pub mod kernels;
pub mod server;

pub use flip_case::flip_case;
pub use kernels::{
    matmul_invariant, matmul_variant, splits_for_batch, sum_flat, sum_split, FIXED_SPLITS,
};
pub use server::{argmax, logits_alone, logits_in_batch, token_depends_on_batch, Model};
