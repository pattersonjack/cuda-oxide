/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Minimal performance repros for addressable aggregate lowering.
//!
//! Build only, keeping the generated IR for inspection:
//!
//! ```text
//! cargo oxide build aggregate_place_perf_repro --emit-nvvm-ir --arch sm_80
//! ```
//!
//! Then inspect:
//!
//! ```text
//! grep -n "alloca \\[4 x \\[2 x double\\]\\|load \\[4 x \\[2 x double\\]\\|store \\[4 x \\[2 x double\\]\\|insertvalue" \
//!   crates/rustc-codegen-cuda/examples/aggregate_place_perf_repro/aggregate_place_perf_repro.ll
//! ```
//!
//! These kernels are intentionally tiny. The same patterns become expensive
//! when the local array is large, e.g. a DOPR54 scratch buffer like
//! `[[f64; 6]; 1024]`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const ROWS: usize = 4;
const COLS: usize = 2;

#[cuda_module]
mod kernels {
    use super::*;

    /// Repro 1: projected read from an addressable local array.
    ///
    /// Rust source:
    ///
    /// ```text
    /// let value = scratch[row][col];
    /// ```
    ///
    /// Expected lowering:
    ///
    /// ```text
    /// row_ptr  = array_element_addr &scratch, row
    /// elem_ptr = array_element_addr row_ptr, col
    /// value    = load double, elem_ptr
    /// ```
    ///
    /// Current lowering on main:
    ///
    /// ```text
    /// load  [4 x [2 x double]], ptr %scratch
    /// alloca [4 x [2 x double]]
    /// store [4 x [2 x double]] <whole loaded aggregate>, ...
    /// ... then index the temporary copy ...
    /// ```
    ///
    /// That whole-array load/store is semantically unnecessary. It is small
    /// here, but with `[[f64; 6]; 1024]` it creates a second 49 KiB stack
    /// object and large local-memory traffic.
    #[kernel]
    pub fn projected_read_copies_whole_array(
        row_arg: usize,
        col_arg: usize,
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }

        let mut scratch = [[0.0_f64; COLS]; ROWS];
        scratch[0] = [1.0, 2.0];
        scratch[1] = [3.0, 4.0];
        scratch[2] = [5.0, 6.0];
        scratch[3] = [7.0, 8.0];

        let row = row_arg & (ROWS - 1);
        let col = col_arg & (COLS - 1);
        let value = scratch[row][col];

        if let Some(slot) = out.get_mut(idx) {
            *slot = value;
        }
    }

    /// Repro 2: aggregate initialization is materialized as one large SSA
    /// value before storing into the stack slot.
    ///
    /// Rust source:
    ///
    /// ```text
    /// let mut scratch = [[0.0_f64; 2]; 4];
    /// ```
    ///
    /// Expected lowering for an addressable local:
    ///
    /// ```text
    /// scratch = alloca [4 x [2 x double]]
    /// zero-fill scratch, or store scalar zeros/elements into scratch
    /// ```
    ///
    /// Current lowering on main:
    ///
    /// ```text
    /// insertvalue ...                 ; build a complete [4 x [2 x double]]
    /// store [4 x [2 x double]] ..., ptr %scratch
    /// ```
    ///
    /// The source needs a zeroed destination, not a by-value aggregate.
    /// This becomes a resource issue for large scratch arrays even before
    /// any projected read copies the array again.
    #[kernel]
    pub fn aggregate_init_builds_whole_value(mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }

        let mut scratch = [[0.0_f64; COLS]; ROWS];
        scratch[2] = [0.0, 42.0];

        if let Some(slot) = out.get_mut(idx) {
            *slot = scratch[2][1];
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let mut out = DeviceBuffer::<f64>::zeroed(&stream, 1).expect("allocate output");

    module
        .projected_read_copies_whole_array(&stream, LaunchConfig::for_num_elems(1), 3, 1, &mut out)
        .expect("launch projected_read_copies_whole_array");
    let got = out
        .to_host_vec(&stream)
        .expect("copy projected read output");
    assert_eq!(got[0], 8.0);

    module
        .aggregate_init_builds_whole_value(&stream, LaunchConfig::for_num_elems(1), &mut out)
        .expect("launch aggregate_init_builds_whole_value");
    let got = out.to_host_vec(&stream).expect("copy init output");
    assert_eq!(got[0], 42.0);

    println!("SUCCESS");
}
