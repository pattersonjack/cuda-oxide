/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Side data for exporting legacy NVVM typed-pointer IR.
//!
//! The internal LLVM dialect intentionally uses opaque pointers. Older NVVM
//! dialects still require typed pointer syntax, so lowering passes can seed
//! value-level facts here and the exporter can complete them with a dataflow
//! pass before printing textual LLVM IR.

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{
    builtin::types::{FP32Type, FP64Type, IntegerType},
    context::{Context, Ptr},
    identifier::Identifier,
    r#type::TypeObj,
    value::Value,
};

use crate::types::{ArrayType, HalfType, PointerType, StructType, VectorType, VoidType};

const FACTS_KEY: &str = "cuda_oxide_legacy_typed_pointer_facts";

/// Value-level type fact used only for legacy typed-pointer export.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LegacyType {
    /// A non-aggregate, non-pointer LLVM type.
    Scalar(Ptr<TypeObj>),
    /// A typed pointer. `None` means the analysis knows the address space but
    /// not the pointee yet; export falls back to `i8*` in that case.
    Pointer {
        pointee: Option<Box<LegacyType>>,
        addrspace: u32,
    },
    Struct(Vec<LegacyType>),
    Array {
        elem: Box<LegacyType>,
        size: u64,
    },
    Vector {
        elem: Box<LegacyType>,
        num_elements: u32,
        scalable: bool,
    },
}

impl LegacyType {
    pub fn from_llvm_type(ctx: &Context, ty: Ptr<TypeObj>) -> Self {
        let ty_ref = ty.deref(ctx);
        if let Some(ptr_ty) = ty_ref.downcast_ref::<PointerType>() {
            LegacyType::Pointer {
                pointee: None,
                addrspace: ptr_ty.address_space(),
            }
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
            LegacyType::Struct(
                struct_ty
                    .fields()
                    .map(|field| LegacyType::from_llvm_type(ctx, field))
                    .collect(),
            )
        } else if let Some(array_ty) = ty_ref.downcast_ref::<ArrayType>() {
            LegacyType::Array {
                elem: Box::new(LegacyType::from_llvm_type(ctx, array_ty.elem_type())),
                size: array_ty.size(),
            }
        } else if let Some(vector_ty) = ty_ref.downcast_ref::<VectorType>() {
            LegacyType::Vector {
                elem: Box::new(LegacyType::from_llvm_type(ctx, vector_ty.elem_type())),
                num_elements: vector_ty.num_elements(),
                scalable: vector_ty.is_scalable(),
            }
        } else {
            LegacyType::Scalar(ty)
        }
    }

    pub fn pointer_to_llvm(ctx: &Context, pointee: Ptr<TypeObj>, addrspace: u32) -> Self {
        LegacyType::Pointer {
            pointee: Some(Box::new(LegacyType::from_llvm_type(ctx, pointee))),
            addrspace,
        }
    }

    pub fn pointer_to(pointee: LegacyType, addrspace: u32) -> Self {
        LegacyType::Pointer {
            pointee: Some(Box::new(pointee)),
            addrspace,
        }
    }

    pub fn unknown_pointer(addrspace: u32) -> Self {
        LegacyType::Pointer {
            pointee: None,
            addrspace,
        }
    }

    pub fn as_pointer(&self) -> Option<(u32, Option<&LegacyType>)> {
        match self {
            LegacyType::Pointer { pointee, addrspace } => Some((*addrspace, pointee.as_deref())),
            _ => None,
        }
    }

    pub fn with_addrspace(&self, addrspace: u32) -> Self {
        match self {
            LegacyType::Pointer { pointee, .. } => LegacyType::Pointer {
                pointee: pointee.clone(),
                addrspace,
            },
            other => other.clone(),
        }
    }

    pub fn merge(&self, other: &LegacyType) -> LegacyType {
        use LegacyType::*;
        match (self, other) {
            (
                Pointer {
                    pointee: a,
                    addrspace: aspace,
                },
                Pointer {
                    pointee: b,
                    addrspace: bspace,
                },
            ) if aspace == bspace => {
                let pointee = match (a.as_deref(), b.as_deref()) {
                    (Some(a), Some(b)) => Some(Box::new(a.merge(b))),
                    (Some(a), None) => Some(Box::new(a.clone())),
                    (None, Some(b)) => Some(Box::new(b.clone())),
                    (None, None) => None,
                };
                Pointer {
                    pointee,
                    addrspace: *aspace,
                }
            }
            (Struct(a), Struct(b)) if a.len() == b.len() => {
                Struct(a.iter().zip(b).map(|(a, b)| a.merge(b)).collect())
            }
            (
                Array { elem: a, size },
                Array {
                    elem: b,
                    size: bsize,
                },
            ) if size == bsize => Array {
                elem: Box::new(a.merge(b)),
                size: *size,
            },
            (
                Vector {
                    elem: a,
                    num_elements,
                    scalable,
                },
                Vector {
                    elem: b,
                    num_elements: bnum,
                    scalable: bscalable,
                },
            ) if num_elements == bnum && scalable == bscalable => Vector {
                elem: Box::new(a.merge(b)),
                num_elements: *num_elements,
                scalable: *scalable,
            },
            _ if self == other => self.clone(),
            // Conflicts usually indicate a bitcast-like use. Keep the older
            // fact rather than inventing a textual bitcast during export.
            _ => self.clone(),
        }
    }

    pub fn indexed(&self, indices: &[u32]) -> Option<LegacyType> {
        let mut cur = self;
        for idx in indices {
            match cur {
                LegacyType::Struct(fields) => cur = fields.get(*idx as usize)?,
                LegacyType::Array { elem, size } => {
                    if u64::from(*idx) >= *size {
                        return None;
                    }
                    cur = elem;
                }
                _ => return None,
            }
        }
        Some(cur.clone())
    }

    pub fn with_indexed(&self, indices: &[u32], value: LegacyType) -> Option<LegacyType> {
        let Some((&head, tail)) = indices.split_first() else {
            return Some(value);
        };

        match self {
            LegacyType::Struct(fields) => {
                let idx = head as usize;
                let field = fields.get(idx)?;
                let mut next = fields.clone();
                next[idx] = field.with_indexed(tail, value)?;
                Some(LegacyType::Struct(next))
            }
            LegacyType::Array { elem, size } => {
                if u64::from(head) >= *size {
                    return None;
                }
                Some(LegacyType::Array {
                    elem: Box::new(elem.with_indexed(tail, value)?),
                    size: *size,
                })
            }
            _ => None,
        }
    }

    /// Result pointee type for a GEP with this type as the source element.
    ///
    /// LLVM's first GEP index steps through the pointer itself, so indexing
    /// into aggregates starts at the second index.
    pub fn gep_indexed_pointee(&self, indices: usize, const_indices: &[Option<u32>]) -> LegacyType {
        let mut cur = self;
        for idx_pos in 1..indices {
            match cur {
                LegacyType::Struct(fields) => {
                    let Some(Some(idx)) = const_indices.get(idx_pos) else {
                        return cur.clone();
                    };
                    let Some(field) = fields.get(*idx as usize) else {
                        return cur.clone();
                    };
                    cur = field;
                }
                LegacyType::Array { elem, .. } | LegacyType::Vector { elem, .. } => {
                    cur = elem;
                }
                _ => return cur.clone(),
            }
        }
        cur.clone()
    }

    pub fn to_llvm_string(&self, ctx: &Context) -> String {
        let mut out = String::new();
        self.write_llvm(ctx, &mut out);
        out
    }

    pub fn write_llvm(&self, ctx: &Context, out: &mut String) {
        match self {
            LegacyType::Scalar(ty) => write_scalar_type(ctx, *ty, out),
            LegacyType::Pointer { pointee, addrspace } => {
                if let Some(pointee) = pointee {
                    pointee.write_llvm(ctx, out);
                } else {
                    write!(out, "i8").unwrap();
                }
                if *addrspace != 0 {
                    write!(out, " addrspace({addrspace})").unwrap();
                }
                write!(out, "*").unwrap();
            }
            LegacyType::Struct(fields) => {
                write!(out, "{{ ").unwrap();
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(out, ", ").unwrap();
                    }
                    field.write_llvm(ctx, out);
                }
                write!(out, " }}").unwrap();
            }
            LegacyType::Array { elem, size } => {
                write!(out, "[{size} x ").unwrap();
                elem.write_llvm(ctx, out);
                write!(out, "]").unwrap();
            }
            LegacyType::Vector {
                elem,
                num_elements,
                scalable,
            } => {
                if *scalable {
                    write!(out, "<vscale x {num_elements} x ").unwrap();
                } else {
                    write!(out, "<{num_elements} x ").unwrap();
                }
                elem.write_llvm(ctx, out);
                write!(out, ">").unwrap();
            }
        }
    }
}

pub type ValueFacts = HashMap<Value, LegacyType>;

pub fn set_value_type_fact(ctx: &mut Context, value: Value, fact: LegacyType) {
    let facts = facts_mut(ctx);
    facts
        .entry(value)
        .and_modify(|existing| *existing = existing.merge(&fact))
        .or_insert(fact);
}

pub fn value_type_facts(ctx: &Context) -> ValueFacts {
    let key = facts_key();
    let Some(idx) = ctx.aux_data_map.get(&key).copied() else {
        return HashMap::new();
    };
    ctx.aux_data
        .get(idx)
        .and_then(|boxed| boxed.downcast_ref::<ValueFacts>())
        .cloned()
        .unwrap_or_default()
}

fn facts_mut(ctx: &mut Context) -> &mut ValueFacts {
    let key = facts_key();
    let idx = if let Some(idx) = ctx.aux_data_map.get(&key).copied() {
        idx
    } else {
        let idx = ctx.aux_data.insert(Box::new(ValueFacts::new()));
        ctx.aux_data_map.insert(key, idx);
        idx
    };

    ctx.aux_data
        .get_mut(idx)
        .and_then(|boxed| boxed.downcast_mut::<ValueFacts>())
        .expect("legacy typed pointer facts have the wrong aux data type")
}

fn facts_key() -> Identifier {
    Identifier::try_new(FACTS_KEY.to_string()).expect("valid identifier")
}

fn write_scalar_type(ctx: &Context, ty: Ptr<TypeObj>, output: &mut String) {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        write!(output, "i{}", int_ty.width()).unwrap();
    } else if ty_ref.is::<VoidType>() {
        write!(output, "void").unwrap();
    } else if ty_ref.is::<HalfType>() {
        write!(output, "half").unwrap();
    } else if ty_ref.is::<FP32Type>() {
        write!(output, "float").unwrap();
    } else if ty_ref.is::<FP64Type>() {
        write!(output, "double").unwrap();
    } else {
        write!(output, "void /* unknown: {} */", ty_ref.disp(ctx)).unwrap();
    }
}
