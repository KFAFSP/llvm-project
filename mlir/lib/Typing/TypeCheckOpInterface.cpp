//===- TypeCheckOpInterface.cpp - Type checking interface -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Typing/TypeCheckOpInterface.h"

#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Typing/TypeChecker.h"

using namespace mlir;

[[nodiscard]]
static bool canTransmuteBranchOpSuccessorEdges(BlockArgument arg, Type into) {
  assert(arg && into);

  // Find all successor uses of the owning block.
  for (auto &use : arg.getOwner()->getUses()) {
    auto iface = llvm::dyn_cast<BranchOpInterface>(use.getOwner());
    if (!iface) {
      // We don't understand this type of successor.
      return false;
    }

    // Determine the least restrictive type for this control flow edge.
    const auto ops = iface.getSuccessorOperands(use.getOperandNumber());
    const auto inTy = ops.isOperandProduced(arg.getArgNumber())
      ? arg.getType()
      : use.getOwner()->getOperand(ops.getOperandIndex(arg.getArgNumber())).getType();

    if (!iface.areTypesCompatible(inTy, into)) {
      // According to the provided interface, this transmutation is illegal.
      return false;
    }
  }

  // We found no evidence that this transmutation is illegal, assuming the
  // parent has initiated it.
  return true;
}

[[nodiscard]]
static bool canTransmuteRegionBranchSuccessorEdges(BlockArgument arg, Type into) {
  assert(arg && into);

  auto iface = llvm::dyn_cast<RegionBranchOpInterface>(arg.getOwner()->getParentOp());
  if (!iface) {
    // Assuming the parent has initiated this transmutation, it is legal.
    return true;
  }

  if (!arg.getOwner()->isEntryBlock()) {
    // Region branches only apply to entry blocks in regions.
    return true;
  }

  // We could now traverse all predecessors of the current region and check
  // their edges. However, since transmutation is triggered by the parent op,
  // we simply check transitive correctness.
  return iface.areTypesCompatible(arg.getType(), into);
}

//===- Generated implementation -------------------------------------------===//

#include "mlir/Typing/TypeCheckOpInterface.cpp.inc"

//===----------------------------------------------------------------------===//

LogicalResult mlir::detail::transmuteArgument(BlockArgument arg, Type into) {
  assert(arg && into);

  // Check the built-in control flow interface for legality.
  if (!canTransmuteBranchOpSuccessorEdges(arg, into)) return failure();
  if (!canTransmuteRegionBranchSuccessorEdges(arg, into)) return failure();

  // Perform the transmutation.
  arg.setType(into);
  return success();
}

LogicalResult mlir::detail::verifyTypeCheckOpInterface(Operation *op) {
  auto iface = llvm::cast<TypeCheckOpInterface>(op);

  Typing::MLIRTypeChecker typeChecker;
  if (auto maybeContra = iface.typeCheck(typeChecker)) {
    maybeContra->becomesEffect(op) << "invalid types specified";
    return maybeContra->handle(op->getLoc());
  }

  assert(typeChecker.isValid() && "broken TypeCheckOpInterface::typeCheck implementation");
  return success();
}
