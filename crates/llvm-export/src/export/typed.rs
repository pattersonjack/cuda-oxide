/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Legacy typed-pointer dataflow for textual LLVM 7 / NVVM 1.x export.

use pliron::{
    basic_block::BasicBlock, builtin::op_interfaces::CallOpInterface, context::Ptr,
    linked_list::ContainsLinkedList, op::Op, operation::Operation, r#type::Typed, value::Value,
};

use crate::{
    ops,
    pointer_facts::{self, LegacyType, ValueFacts},
    types::{FuncType, PointerType, VoidType},
};

use super::{
    state::{ModuleExportState, PredecessorMap},
    types::addrspace_of_type,
};

impl<'a> ModuleExportState<'a> {
    pub(super) fn analyze_legacy_type_facts(
        &self,
        func: &ops::FuncOp,
        pred_map: &PredecessorMap,
    ) -> ValueFacts {
        if !self.uses_legacy_typed_pointers() {
            return ValueFacts::new();
        }

        let mut facts = pointer_facts::value_type_facts(self.ctx);
        seed_function_values(self.ctx, func, &mut facts);

        for _ in 0..64 {
            let before = facts.len();
            let mut changed = false;
            changed |= propagate_phi_facts(self.ctx, func, pred_map, &mut facts);
            changed |= propagate_op_facts(self.ctx, func, &mut facts);
            if !changed && facts.len() == before {
                break;
            }
        }

        facts
    }
}

fn seed_function_values(
    ctx: &pliron::context::Context,
    func: &ops::FuncOp,
    facts: &mut ValueFacts,
) {
    if func.get_operation().deref(ctx).regions().count() == 0 {
        return;
    }

    let region = func.get_operation().deref(ctx).get_region(0);
    for block in region.deref(ctx).iter(ctx) {
        for arg in block.deref(ctx).arguments() {
            set_fact(
                facts,
                arg,
                LegacyType::from_llvm_type(ctx, arg.get_type(ctx)),
            );
        }
        for op in block.deref(ctx).iter(ctx) {
            for res in op.deref(ctx).results() {
                set_fact(
                    facts,
                    res,
                    LegacyType::from_llvm_type(ctx, res.get_type(ctx)),
                );
            }
        }
    }
}

fn propagate_phi_facts(
    _ctx: &pliron::context::Context,
    func: &ops::FuncOp,
    pred_map: &PredecessorMap,
    facts: &mut ValueFacts,
) -> bool {
    let mut changed = false;
    let region = func.get_operation().deref(_ctx).get_region(0);
    for block in region.deref(_ctx).iter(_ctx) {
        let args: Vec<_> = block.deref(_ctx).arguments().collect();
        if args.is_empty() {
            continue;
        }
        let Some(preds) = pred_map.get(&block) else {
            continue;
        };

        for (arg_idx, arg) in args.iter().enumerate() {
            for (_, pred_args) in preds {
                let Some(incoming) = pred_args.get(arg_idx).copied() else {
                    continue;
                };
                if let Some(fact) = facts.get(&incoming).cloned() {
                    changed |= set_fact(facts, *arg, fact);
                }
                if let Some(fact) = facts.get(arg).cloned() {
                    changed |= set_fact(facts, incoming, fact);
                }
            }
        }
    }
    changed
}

fn propagate_op_facts(
    ctx: &pliron::context::Context,
    func: &ops::FuncOp,
    facts: &mut ValueFacts,
) -> bool {
    let mut changed = false;
    let region = func.get_operation().deref(ctx).get_region(0);
    for block in region.deref(ctx).iter(ctx) {
        for op in block.deref(ctx).iter(ctx) {
            let op_obj = Operation::get_op_dyn(op, ctx);
            let op_dyn = op_obj.as_ref();
            let op_ref = op.deref(ctx);

            if op_dyn.downcast_ref::<ops::LoadOp>().is_some() {
                let ptr = op_ref.get_operand(0);
                let res = op_ref.get_result(0);
                changed |= propagate_load_like(ctx, ptr, res, facts);
            } else if op_dyn.downcast_ref::<ops::StoreOp>().is_some() {
                let val = op_ref.get_operand(0);
                let ptr = op_ref.get_operand(1);
                changed |= propagate_store_like(ctx, ptr, val, facts);
            } else if let Some(alloca) = op_dyn.downcast_ref::<ops::AllocaOp>() {
                let res = op_ref.get_result(0);
                let elem_ty = alloca
                    .get_attr_alloca_element_type(ctx)
                    .expect("Missing alloca_element_type")
                    .get_type(ctx);
                changed |= set_fact(facts, res, LegacyType::pointer_to_llvm(ctx, elem_ty, 0));
            } else if let Some(gep) = op_dyn.downcast_ref::<ops::GetElementPtrOp>() {
                changed |= propagate_gep(ctx, gep, facts);
            } else if let Some(load) = op_dyn.downcast_ref::<ops::AtomicLoadOp>() {
                let ptr = op_ref.get_operand(0);
                let res = op_ref.get_result(0);
                changed |= propagate_load_like(ctx, ptr, res, facts);
                let _ = load;
            } else if op_dyn.downcast_ref::<ops::AtomicStoreOp>().is_some() {
                let val = op_ref.get_operand(0);
                let ptr = op_ref.get_operand(1);
                changed |= propagate_store_like(ctx, ptr, val, facts);
            } else if op_dyn.downcast_ref::<ops::AtomicRmwOp>().is_some() {
                let res = op_ref.get_result(0);
                let ptr = op_ref.get_operand(0);
                let val = op_ref.get_operand(1);
                changed |= propagate_store_like(ctx, ptr, val, facts);
                changed |= propagate_load_like(ctx, ptr, res, facts);
            } else if op_dyn.downcast_ref::<ops::AtomicCmpxchgOp>().is_some() {
                let ptr = op_ref.get_operand(0);
                let cmp = op_ref.get_operand(1);
                changed |= propagate_store_like(ctx, ptr, cmp, facts);
            } else if op_dyn.downcast_ref::<ops::BitcastOp>().is_some() {
                changed |= propagate_pointer_cast(ctx, op, false, facts);
            } else if op_dyn.downcast_ref::<ops::AddrSpaceCastOp>().is_some() {
                changed |= propagate_pointer_cast(ctx, op, true, facts);
            } else if op_dyn.downcast_ref::<ops::IntToPtrOp>().is_some() {
                let res = op_ref.get_result(0);
                let addrspace = addrspace_of_type(res.get_type(ctx), ctx);
                changed |= set_fact(facts, res, LegacyType::unknown_pointer(addrspace));
            } else if let Some(select) = op_dyn.downcast_ref::<ops::SelectOp>() {
                let _ = select;
                let res = op_ref.get_result(0);
                let true_val = op_ref.get_operand(1);
                let false_val = op_ref.get_operand(2);
                changed |= unify_values(res, true_val, facts);
                changed |= unify_values(res, false_val, facts);
            } else if let Some(extract) = op_dyn.downcast_ref::<ops::ExtractValueOp>() {
                let agg = op_ref.get_operand(0);
                let res = op_ref.get_result(0);
                if let Some(agg_fact) = facts.get(&agg)
                    && let Some(field_fact) = agg_fact.indexed(&extract.indices(ctx))
                {
                    changed |= set_fact(facts, res, field_fact);
                }
            } else if let Some(insert) = op_dyn.downcast_ref::<ops::InsertValueOp>() {
                let agg = op_ref.get_operand(0);
                let val = op_ref.get_operand(1);
                let res = op_ref.get_result(0);
                let agg_fact = facts
                    .get(&agg)
                    .cloned()
                    .unwrap_or_else(|| LegacyType::from_llvm_type(ctx, agg.get_type(ctx)));
                let val_fact = facts
                    .get(&val)
                    .cloned()
                    .unwrap_or_else(|| LegacyType::from_llvm_type(ctx, val.get_type(ctx)));
                if let Some(result_fact) = agg_fact.with_indexed(&insert.indices(ctx), val_fact) {
                    changed |= set_fact(facts, res, result_fact);
                }
            } else if let Some(call) = op_dyn.downcast_ref::<ops::CallOp>() {
                let func_ty = call.callee_type(ctx);
                let ret_ty = func_ty
                    .deref(ctx)
                    .downcast_ref::<FuncType>()
                    .expect("CallOp callee type must be FuncType")
                    .result_type();
                if !ret_ty.deref(ctx).is::<VoidType>() {
                    let res = op_ref.get_result(0);
                    changed |= set_fact(facts, res, LegacyType::from_llvm_type(ctx, ret_ty));
                }
            }
        }
    }
    changed
}

fn propagate_load_like(
    ctx: &pliron::context::Context,
    ptr: Value,
    res: Value,
    facts: &mut ValueFacts,
) -> bool {
    let mut changed = false;
    let result_fact = facts
        .get(&res)
        .cloned()
        .unwrap_or_else(|| LegacyType::from_llvm_type(ctx, res.get_type(ctx)));
    let addrspace = addrspace_of_type(ptr.get_type(ctx), ctx);
    changed |= set_fact(
        facts,
        ptr,
        LegacyType::pointer_to(result_fact.clone(), addrspace),
    );

    if let Some((_, Some(pointee))) = facts.get(&ptr).and_then(LegacyType::as_pointer) {
        changed |= set_fact(facts, res, pointee.clone());
    }
    changed
}

fn propagate_store_like(
    ctx: &pliron::context::Context,
    ptr: Value,
    val: Value,
    facts: &mut ValueFacts,
) -> bool {
    let val_fact = facts
        .get(&val)
        .cloned()
        .unwrap_or_else(|| LegacyType::from_llvm_type(ctx, val.get_type(ctx)));
    let addrspace = addrspace_of_type(ptr.get_type(ctx), ctx);
    set_fact(facts, ptr, LegacyType::pointer_to(val_fact, addrspace))
}

fn propagate_gep(
    ctx: &pliron::context::Context,
    gep: &ops::GetElementPtrOp,
    facts: &mut ValueFacts,
) -> bool {
    let op_ref = gep.get_operation().deref(ctx);
    let ptr = op_ref.get_operand(0);
    let res = op_ref.get_result(0);
    let src_elem_ty = gep
        .get_attr_gep_src_elem_type(ctx)
        .expect("Missing gep_src_elem_type")
        .get_type(ctx);
    let src_elem = facts
        .get(&ptr)
        .and_then(|fact| fact.as_pointer().and_then(|(_, pointee)| pointee.cloned()))
        .unwrap_or_else(|| LegacyType::from_llvm_type(ctx, src_elem_ty));

    let addrspace = addrspace_of_type(ptr.get_type(ctx), ctx);
    let mut changed = set_fact(
        facts,
        ptr,
        LegacyType::pointer_to(src_elem.clone(), addrspace),
    );

    let indices = gep.get_attr_gep_indices(ctx).unwrap();
    let const_indices: Vec<_> = indices
        .0
        .iter()
        .map(|idx| match idx {
            crate::attributes::GepIndexAttr::Constant(c) => Some(*c),
            crate::attributes::GepIndexAttr::OperandIdx(_) => None,
        })
        .collect();
    let result_pointee = src_elem.gep_indexed_pointee(indices.0.len(), &const_indices);
    changed |= set_fact(
        facts,
        res,
        LegacyType::pointer_to(result_pointee, addrspace),
    );
    changed
}

fn propagate_pointer_cast(
    ctx: &pliron::context::Context,
    op: Ptr<Operation>,
    update_addrspace: bool,
    facts: &mut ValueFacts,
) -> bool {
    let op_ref = op.deref(ctx);
    let val = op_ref.get_operand(0);
    let res = op_ref.get_result(0);

    let src_as = addrspace_of_type(val.get_type(ctx), ctx);
    let dst_as = addrspace_of_type(res.get_type(ctx), ctx);
    if !val.get_type(ctx).deref(ctx).is::<PointerType>()
        || !res.get_type(ctx).deref(ctx).is::<PointerType>()
    {
        return false;
    }

    let mut changed = false;
    if let Some(src) = facts.get(&val).cloned() {
        let fact = if update_addrspace {
            src.with_addrspace(dst_as)
        } else {
            src
        };
        changed |= set_fact(facts, res, fact);
    }
    if let Some(dst) = facts.get(&res).cloned() {
        let fact = if update_addrspace {
            dst.with_addrspace(src_as)
        } else {
            dst
        };
        changed |= set_fact(facts, val, fact);
    }
    changed
}

fn unify_values(a: Value, b: Value, facts: &mut ValueFacts) -> bool {
    let mut changed = false;
    if let Some(fact) = facts.get(&a).cloned() {
        changed |= set_fact(facts, b, fact);
    }
    if let Some(fact) = facts.get(&b).cloned() {
        changed |= set_fact(facts, a, fact);
    }
    changed
}

fn set_fact(facts: &mut ValueFacts, value: Value, fact: LegacyType) -> bool {
    match facts.get_mut(&value) {
        Some(existing) => {
            let merged = existing.merge(&fact);
            if *existing != merged {
                *existing = merged;
                true
            } else {
                false
            }
        }
        None => {
            facts.insert(value, fact);
            true
        }
    }
}

pub(super) fn build_predecessor_map(
    ctx: &pliron::context::Context,
    func: &ops::FuncOp,
) -> PredecessorMap {
    let mut pred_map: PredecessorMap = std::collections::HashMap::new();
    for block in func
        .get_operation()
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
    {
        let block_ref = block.deref(ctx);
        if let Some(term) = block_ref.iter(ctx).last() {
            let term_obj = Operation::get_op_dyn(term, ctx);
            let term_dyn = term_obj.as_ref();

            if term_dyn.downcast_ref::<ops::BrOp>().is_some() {
                let dest = term.deref(ctx).successors().next().unwrap();
                let args: Vec<_> = term.deref(ctx).operands().collect();
                pred_map.entry(dest).or_default().push((block, args));
            } else if term_dyn.downcast_ref::<ops::CondBrOp>().is_some() {
                let succs: Vec<_> = term.deref(ctx).successors().collect();
                let true_dest = succs[0];
                let false_dest = succs[1];

                let num_true = true_dest.deref(ctx).arguments().count();
                let num_false = false_dest.deref(ctx).arguments().count();

                let all_ops: Vec<_> = term.deref(ctx).operands().collect();
                if all_ops.len() >= 1 + num_true + num_false {
                    let true_args = all_ops[1..=num_true].to_vec();
                    let false_args = all_ops[1 + num_true..1 + num_true + num_false].to_vec();

                    pred_map
                        .entry(true_dest)
                        .or_default()
                        .push((block, true_args));
                    pred_map
                        .entry(false_dest)
                        .or_default()
                        .push((block, false_args));
                }
            }
        }
    }
    pred_map
}

pub(super) fn build_block_labels(
    ctx: &pliron::context::Context,
    func: &ops::FuncOp,
    entry_block: Ptr<BasicBlock>,
) -> std::collections::HashMap<Ptr<BasicBlock>, String> {
    let mut block_labels = std::collections::HashMap::new();
    let mut next_label_id = 0;
    for (i, block_node) in func
        .get_operation()
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .enumerate()
    {
        if i == 0 {
            debug_assert_eq!(block_node, entry_block);
            block_labels.insert(block_node, "entry".to_string());
        } else {
            let label = format!("bb{next_label_id}");
            next_label_id += 1;
            block_labels.insert(block_node, label);
        }
    }
    block_labels
}
