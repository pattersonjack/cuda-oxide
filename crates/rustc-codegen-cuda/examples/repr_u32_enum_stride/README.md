# `repr_u32_enum_stride` — minimal repro

Reads from `*const E` where `E` is a fieldless `#[repr(u32)] enum` are
miscompiled by `rustc-codegen-cuda`: pointer arithmetic strides by
**1 byte** instead of the expected 4.

The example uses a generic `Tag` enum with four valid discriminants:
`Foo = 0`, `Bar = 1`, `Baz = 2`, and `Qux = 3`. The kernel buffer at
offset 4 reads as `1` via `*const u32`, but a buggy enum pointer path does
not produce the slot-1 discriminant. The failure pattern matches 1-byte
stride instead of the expected 4-byte stride.

## Run

```bash
cargo oxide run repr_u32_enum_stride
```

## Expected output before the fix

```
control_u32   [0, 1, 2, 3]   PASS
enum_ptr      [0, 0, 0, 0]   FAIL
RESULT: control and enum-path DISAGREE — bug reproduced.
```

## Expected output after the fix

```
control_u32   [0, 1, 2, 3]   PASS
enum_ptr      [0, 1, 2, 3]   PASS
RESULT: control and enum-path agree — bug not reproduced (good!).
```

## Root cause and fix

The MIR importer lowered fieldless enum discriminants from the number of
variants alone. For this four-variant `#[repr(u32)]` enum, that selected an
8-bit discriminant type even though Rust's explicit representation requires a
32-bit layout.

Pointer arithmetic over `*const Tag` then used the lowered enum element size,
so `base.add(1)` advanced by 1 byte instead of 4 bytes. The fix in
`crates/mir-importer/src/translator/types.rs` consults `AdtDef::repr().int`
first and uses the requested integer width whenever the enum has an explicit
integer representation. The existing variant-count heuristic remains the
fallback for enums without an explicit integer repr.

`Tag` in this example is exactly the shape under test: a fieldless
`#[repr(u32)]` enum with four variants (`Foo = 0`, `Bar = 1`, `Baz = 2`,
`Qux = 3`) stored in a device buffer.
