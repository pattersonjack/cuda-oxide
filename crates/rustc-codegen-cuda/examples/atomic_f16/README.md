# atomic_f16

Correctness and microbenchmark coverage for scalar `f16` CUDA atomics.

## Correctness

```bash
cargo oxide run atomic_f16 --arch sm_103
```

The default binary checks:

- `DeviceAtomicF16::fetch_add`
- `DeviceAtomicF16::fetch_sub`
- `BlockAtomicF16::fetch_add`
- returned old values and final bin counts

## Benchmark

```bash
./crates/rustc-codegen-cuda/examples/atomic_f16/run-bench.sh --arch sm_103 \
  > crates/rustc-codegen-cuda/examples/atomic_f16/traces/atomic_f16_bench.csv
```

The benchmark emits CSV:

```text
n,mode,type,bins,avg_ms,mops
```

`mode=unused` ignores the returned old value. `mode=return` stores every old
value and forces an atomic form that returns the previous value. Inspect the
generated PTX (`bench.ptx`) to see whether the backend selected `atom` or
`red` for a given toolchain.

Scalar f16 atomics are a correctness feature, not a guaranteed speedup over f32
atomics. The checked-in trace plot shows the exact-integer accumulation limit
for repeated f16 `+1`; use the benchmark on target hardware for throughput.

The checked-in benchmark trace was collected with `--arch sm_103`. Scalar f16
global atomics were slower than f32 in every measured case:

```text
bins  f32 unused Mops  f16 unused Mops  slowdown
1     657              36               18x
16    10416            60               173x
256   30555            103              298x
4096  259703           859              302x
```

The benchmark does not cover packed `f16x2` atomics. Those require a separate
packed-half API and are not a drop-in replacement for scalar `DeviceAtomicF16`.

## Traces

The `traces/` directory contains the checked-in `sm_103` benchmark CSV and PNG
artifacts used to document the scalar f16 vs f32 behavior.
