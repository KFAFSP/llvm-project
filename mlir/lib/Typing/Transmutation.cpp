//===- Transmutation.h - Value type transmutation ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Typing/Transmutation.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Typing/Context.h"
#include "mlir/Typing/TypeCheckOpInterface.h"

using namespace mlir;
using namespace mlir::Typing;

bool mlir::Typing::mayTransmute(const OpOperand &use) {
  // - builtin.unrealized_conversion_cast supports transmutation.
  // - Ops that opt-in to type checking must support operand transmutation.
  auto *owner = use.getOwner();
  return llvm::isa<UnrealizedConversionCastOp, TypeCheckOpInterface>(owner);
}

Operation *mlir::Typing::promote(
    const Context &context,
    OpBuilder &builder,
    Location loc,
    Value value,
    Type superType) {
  assert(value && superType);
  const auto &typeSystem = context.getTypeSystem(value);
  assert(typeSystem.isSubtype(value.getType(), superType));

  return typeSystem.promote(builder, loc, value, superType);
}

LogicalResult mlir::Typing::transmute(
    const Context &context,
    RewriterBase &rewriter,
    Value value,
    Type subType,
    bool allowPromote) {
  assert(value && subType);
  const auto &typeSystem = context.getTypeSystem(value);
  assert(typeSystem.isSubtype(subType, value.getType()));

  // The owner must opt-in to transmutation.
  auto iface = llvm::dyn_cast<TypeCheckOpInterface>(getOwner(value));
  if (!iface) return failure();

  // Find all uses of the value that would be broken by transmutation.
  SmallVector<OpOperand *> brokenUses;
  for (auto &use : value.getUses()) {
    if (mayTransmute(use)) continue;
    brokenUses.push_back(&use);
  }

  // Fix all broken uses by inserting a promoting cast.
  Operation *promotion = nullptr;
  if (!brokenUses.empty()) {
    if (!allowPromote) return failure();

    // Setup the rewriter to insert the casts at the right location.
    OpBuilder::InsertionGuard guard(rewriter);
    const auto loc = (*iface).getLoc();
    if (llvm::isa<OpResult>(value))
      rewriter.setInsertionPointAfter(iface);
    else
      rewriter.setInsertionPointToStart(llvm::cast<BlockArgument>(value).getOwner());

    // For the purposes of instantiating the promoting cast, we "preview" the
    // transmutation by setting the type on the value, which we undo afterwards.
    const auto superType = value.getType();
    value.setType(subType);
    promotion = promote(context, rewriter, loc, value, value.getType());
    value.setType(superType);
    if (!promotion) return failure();
  }

  // Attempt to perform the actual transmutation.
  rewriter.startOpModification(iface);
  if (failed(iface.transmute(value, subType))) {
    rewriter.cancelOpModification(iface);
    rewriter.eraseOp(promotion);
    return failure();
  }
  rewriter.finalizeOpModification(iface);

  // Fix all broken uses by going through the promotion instead.
  for (auto *use : brokenUses)
    use->set(promotion->getResult(0));
  return success();
}
