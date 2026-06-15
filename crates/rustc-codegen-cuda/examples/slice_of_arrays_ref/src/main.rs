// Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread};

// Repro: indexing the outer slice of a `&mut [[u32; COLS]]` should step by one
// whole row. A bug in codegen instead treats the slice data pointer like a
// pointer to the first row and writes across row 0.
//
// Kernel statement:
//     rows[i][0] = i + 1
//
// Expected output, one thread per row:
//     [1, 0, 0]
//     [2, 0, 0]
//     [3, 0, 0]
//
// Actual output on affected codegen:
//     [1, 2, 3]
//     [0, 0, 0]
//     [0, 0, 0]
const ROWS: usize = 3;
const COLS: usize = 3;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn write_first_col(rows: &mut [[u32; COLS]]) {
        let i = thread::index_1d().get();
        if i < rows.len() {
            // `rows` is a slice of fixed-size rows. This should write:
            //   thread 0 -> rows[0][0]
            //   thread 1 -> rows[1][0]
            //   thread 2 -> rows[2][0]
            rows[i][0] = i as u32 + 1;
        }
    }
}

fn expected_rows() -> Vec<[u32; COLS]> {
    (0..ROWS)
        .map(|r| {
            let mut row = [0; COLS];
            row[0] = r as u32 + 1;
            row
        })
        .collect()
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let mut rows_dev = DeviceBuffer::<[u32; COLS]>::zeroed(&stream, ROWS).unwrap();
    module
        .write_first_col(
            &stream,
            LaunchConfig::for_num_elems(ROWS as u32),
            &mut rows_dev,
        )
        .expect("launch");

    let actual = rows_dev.to_host_vec(&stream).unwrap();
    let expected = expected_rows();

    assert_eq!(actual, expected);
}
