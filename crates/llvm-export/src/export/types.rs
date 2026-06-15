/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM type printing.

use std::fmt::Write;

use pliron::{
    builtin::types::{FP32Type, FP64Type, IntegerType},
    context::Ptr,
    r#type::TypeObj,
    r#type::Typed,
};

use crate::types::{HalfType, PointerType, StructType, VoidType};

use super::state::ModuleExportState;

pub(super) fn addrspace_of_type(ty: Ptr<TypeObj>, ctx: &pliron::context::Context) -> u32 {
    ty.deref(ctx)
        .downcast_ref::<PointerType>()
        .map_or(0, PointerType::address_space)
}

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_type(&self, ty: Ptr<TypeObj>, output: &mut String) -> Result<(), String> {
        let ty_ref = ty.deref(self.ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            write!(output, "i{}", int_ty.width()).unwrap();
        } else if let Some(ptr_ty) = ty_ref.downcast_ref::<PointerType>() {
            let addrspace = ptr_ty.address_space();
            if addrspace != 0 {
                write!(output, "ptr addrspace({addrspace})").unwrap();
            } else {
                write!(output, "ptr").unwrap();
            }
        } else if ty_ref.is::<VoidType>() {
            write!(output, "void").unwrap();
        } else if ty_ref.is::<HalfType>() {
            write!(output, "half").unwrap();
        } else if ty_ref.is::<FP32Type>() {
            write!(output, "float").unwrap();
        } else if ty_ref.is::<FP64Type>() {
            write!(output, "double").unwrap();
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
            write!(output, "{{ ").unwrap();
            for (i, elem_ty) in struct_ty.fields().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                self.export_type(elem_ty, output)?;
            }
            write!(output, " }}").unwrap();
        } else if let Some(array_ty) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
            write!(output, "[{} x ", array_ty.size()).unwrap();
            self.export_type(array_ty.elem_type(), output)?;
            write!(output, "]").unwrap();
        } else if let Some(vec_ty) = ty_ref.downcast_ref::<crate::types::VectorType>() {
            write!(output, "<{} x ", vec_ty.num_elements()).unwrap();
            self.export_type(vec_ty.elem_type(), output)?;
            write!(output, ">").unwrap();
        } else {
            write!(output, "void /* unknown: {} */", ty_ref.disp(self.ctx)).unwrap();
        }
        Ok(())
    }

    pub(super) fn export_type_without_value(
        &self,
        ty: Ptr<TypeObj>,
        output: &mut String,
    ) -> Result<(), String> {
        if self.uses_legacy_typed_pointers() {
            crate::pointer_facts::LegacyType::from_llvm_type(self.ctx, ty)
                .write_llvm(self.ctx, output);
            Ok(())
        } else {
            self.export_type(ty, output)
        }
    }

    pub(super) fn export_value_type(
        &self,
        val: pliron::value::Value,
        output: &mut String,
    ) -> Result<(), String> {
        if self.uses_legacy_typed_pointers()
            && let Some(fact) = self.legacy_fact(val)
        {
            fact.write_llvm(self.ctx, output);
            Ok(())
        } else {
            self.export_type_without_value(val.get_type(self.ctx), output)
        }
    }

    pub(super) fn legacy_function_ref(
        &self,
        name: &str,
        ret_ty: Ptr<TypeObj>,
        args: &[pliron::value::Value],
    ) -> String {
        let mut output = String::new();
        self.export_type_without_value(ret_ty, &mut output)
            .expect("type printing cannot fail");
        output.push_str(" (");
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            self.export_value_type(*arg, &mut output)
                .expect("type printing cannot fail");
        }
        write!(output, ")* @{name}").unwrap();
        output
    }

    pub(super) fn legacy_decl_function_ref(
        &self,
        name: &str,
        ret_ty: Ptr<TypeObj>,
        arg_tys: &[Ptr<TypeObj>],
    ) -> String {
        let mut output = String::new();
        self.export_type_without_value(ret_ty, &mut output)
            .expect("type printing cannot fail");
        output.push_str(" (");
        for (i, arg_ty) in arg_tys.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            self.export_type_without_value(*arg_ty, &mut output)
                .expect("type printing cannot fail");
        }
        write!(output, ")* @{name}").unwrap();
        output
    }

    /// Compute conservative ABI alignment (bytes) for a type.
    ///
    /// Used as the fallback when no explicit alignment is stamped on a
    /// load/store/alloca op. Required for atomic loads/stores (LLVM IR
    /// mandates explicit alignment) and for vectorization hints.
    pub(super) fn natural_alignment(&self, ty: Ptr<TypeObj>) -> u32 {
        let ty_ref = ty.deref(self.ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            // ceil(width / 8), minimum 1.
            std::cmp::max(1, int_ty.width() / 8)
        } else if ty_ref.is::<FP32Type>() {
            4
        } else if ty_ref.is::<FP64Type>() {
            8
        } else if ty_ref.is::<HalfType>() {
            2
        } else if ty_ref.is::<PointerType>() {
            8
        } else if let Some(array_ty) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
            // ABI alignment of `[N x T]` matches elem alignment.
            self.natural_alignment(array_ty.elem_type())
        } else if let Some(vec_ty) = ty_ref.downcast_ref::<crate::types::VectorType>() {
            // ABI alignment of an LLVM vector: power-of-2-rounded total width.
            let elem = self.natural_alignment(vec_ty.elem_type());
            let total = elem.saturating_mul(vec_ty.num_elements());
            let mut a = 1u32;
            while a.saturating_mul(2) <= total && a < 128 {
                a *= 2;
            }
            a
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
            // Max field alignment (1 if empty). May under-state a repr(align)
            // raise; the true alignment is carried on the op, not the type.
            struct_ty
                .fields()
                .map(|f| self.natural_alignment(f))
                .max()
                .unwrap_or(1)
        } else {
            // Conservative fallback for pointers and unknown types.
            8
        }
    }
}
