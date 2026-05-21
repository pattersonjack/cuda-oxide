// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Minimal reproduction for a rustc-codegen-cuda miscompile: reading from
 * `*const E` where `E` is a fieldless `#[repr(u32)] enum` strides by
 * 1 byte instead of the expected 4 bytes.
 *
 * Symptom:
 *   The kernel buffer at offset 4 reads as `1` via `*const u32`, but the
 *   same bytes read through a pointer to a fieldless `#[repr(u32)]` enum do
 *   not produce the slot-1 discriminant. The failure pattern matches 1-byte
 *   pointer stride instead of the expected 4-byte stride.
 *
 * Test design:
 *   Host buffer = [0u32, 1, 2, 3] - chosen so each slot's `as u32` value
 *   *is* a valid `Tag` discriminant. The kernel receives a single pointer
 *   and reads slot 0..=3 two ways:
 *
 *     a) `*const u32`  - control. Always reads stride 4. Should give 0..=3.
 *     b) `*const Tag`  - under test. If stride is correctly 4, gives
 *                        Tag::Foo, Bar, Baz, Qux (= 0,1,2,3
 *                        when cast back to u32). If stride is buggy (1),
 *                        slots 1..=3 read zero bytes inside the first u32.
 *
 * Output (current cuda-oxide spike/applied, expected):
 *   control_u32  [0, 1, 2, 3]  PASS
 *   enum_ptr     [0, 0, 0, 0]  FAIL (1-byte stride)
 *
 * After fix:
 *   control_u32  [0, 1, 2, 3]  PASS
 *   enum_ptr     [0, 1, 2, 3]  PASS
 *
 * Build:
 *   cargo oxide run repr_u32_enum_stride
 */

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Fieldless `#[repr(u32)]` enum used as a tag-like device-buffer element.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Tag {
    Foo = 0,
    Bar = 1,
    Baz = 2,
    Qux = 3,
}

// SAFETY: trivial repr(u32) POD, safe to copy device↔host.
unsafe impl cuda_core::DeviceCopy for Tag {}

const N: usize = 4;

#[cuda_module]
mod kernels {
    use super::*;

    /// Control: read via `*const u32` with `add(i)`. Stride should be 4.
    /// Writes input[i] for i in 0..N.
    #[kernel]
    pub fn read_via_u32(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        // Read through a raw u32 pointer with arithmetic.
        let base: *const u32 = input.as_ptr();
        let v = unsafe { *base.add(i) };
        if let Some(slot) = out.get_mut(idx) {
            *slot = v;
        }
    }

    /// Test: read via `*const Tag` with `add(i)`, then cast the discriminant
    /// back to u32. If stride is correctly 4 the output matches the u32
    /// control. If stride is buggy (8) the output skips slots.
    #[kernel]
    pub fn read_via_enum(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        // Reinterpret the input buffer as `*const Tag`. The bytes are
        // identical (both u32-sized, repr(u32)); only pointer-arithmetic
        // stride is under test.
        let base: *const Tag = input.as_ptr() as *const Tag;
        let tag = unsafe { *base.add(i) };
        if let Some(slot) = out.get_mut(idx) {
            *slot = tag as u32;
        }
    }
}

fn run_and_report<F>(name: &str, stream: &Arc<CudaStream>, launch: F) -> Vec<u32>
where
    F: FnOnce(&Arc<CudaStream>, LaunchConfig, &DeviceBuffer<u32>, &mut DeviceBuffer<u32>),
{
    // Input designed so that any reasonable misstride produces a visible
    // wrong discriminant: 0,1,2,3 all map to distinct Tag variants.
    let input: Vec<u32> = (0..N as u32).collect();
    let dev_in = DeviceBuffer::from_host(stream, &input).unwrap();
    let mut dev_out = DeviceBuffer::<u32>::zeroed(stream, N).unwrap();

    launch(
        stream,
        LaunchConfig::for_num_elems(N as u32),
        &dev_in,
        &mut dev_out,
    );

    let host_out = dev_out.to_host_vec(stream).unwrap();
    let expected: Vec<u32> = (0..N as u32).collect();
    let verdict = if host_out == expected { "PASS" } else { "FAIL" };
    println!("  {name:<14}  {host_out:?}   {verdict}");
    host_out
}

fn main() {
    println!("=== repr(u32) enum pointer-stride miscompile minimal repro ===\n");
    println!("Input host buffer: [0, 1, 2, 3]");
    println!("Expected output (both kernels, if codegen is correct): [0, 1, 2, 3]");
    println!("Buggy expected output (enum_ptr, stride-1): slots 1..=3 read zero bytes\n");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");

    let control = run_and_report("control_u32", &stream, |s, cfg, i, o| {
        module.read_via_u32(s, cfg, i, o).expect("launch")
    });
    let enum_path = run_and_report("enum_ptr", &stream, |s, cfg, i, o| {
        module.read_via_enum(s, cfg, i, o).expect("launch")
    });

    println!();
    if control == enum_path {
        println!("RESULT: control and enum-path agree — bug not reproduced (good!).");
        std::process::exit(0);
    } else {
        println!("RESULT: control and enum-path DISAGREE — bug reproduced.");
        println!("        control_u32: {control:?}");
        println!("        enum_ptr:    {enum_path:?}");
        std::process::exit(1);
    }
}
