//===- FixPointTypeChecker.cpp - Fix point type checker ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Typing/FixPointTypeChecker.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Typing/TypeCheckOpInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE  "fix-point-type-checker"

using namespace mlir;
using namespace mlir::Typing;

//===----------------------------------------------------------------------===//
// FixPointTypeChecker implementation
//===----------------------------------------------------------------------===//

void FixPointTypeChecker::invalidate(Operation *op) {
  assert(op);
  if (!llvm::isa<TypeCheckOpInterface>(op)) return;

  LLVM_DEBUG({
    OpPrintingFlags flags{};
    flags.skipRegions();
    flags.elideLargeElementsAttrs();
    flags.elideLargeResourceString();

    llvm::dbgs() << "[FixPointTypeChecker] invalidating ";
    op->print(llvm::dbgs(), flags);
  });

  invalid.insert(op);
}

void FixPointTypeChecker::invalidateAll(Operation *root) {
  assert(root);

  LLVM_DEBUG({
    OpPrintingFlags flags{};
    flags.skipRegions();
    flags.elideLargeElementsAttrs();
    flags.elideLargeResourceString();

    llvm::dbgs() << "[FixPointTypeChecker] invalidating all below ";
    root->print(llvm::dbgs(), flags);
  });

  const auto hasIface = [](Operation *op) {
    return llvm::isa<TypeCheckOpInterface>(op);
  };
  invalid.insert(LexicalOpQueue(root, hasIface));
}

std::optional<Contradiction> FixPointTypeChecker::trySolve() {
  const auto beginIt = [&](Operation *op) {
    if (iterationLimit == kNoLimit) return true;
    const auto [it, _] = iterations.try_emplace(op, 0U);
    return ++it->second <= iterationLimit;
  };

  while (auto *next = invalid.dequeue()) {
    LLVM_DEBUG({
      OpPrintingFlags flags{};
      flags.skipRegions();
      flags.elideLargeElementsAttrs();
      flags.elideLargeResourceString();

      llvm::dbgs() << "[FixPointTypeChecker] applying rule for ";
      next->print(llvm::dbgs(), flags);
      llvm::dbgs() << "\n";
    });

    // Attempt to start another iteration for the next invalid op.
    if (!beginIt(next)) {
      auto contra = this->fatal(next);
      contra << "iteration limit reached";
      return contra;
    }

    // Type check this operation and apply its deductions.
    auto iface = llvm::cast<TypeCheckOpInterface>(*next);
    auto maybeContra = iface.typeCheck(*this);
    if (!maybeContra || !maybeContra->isFatal()) continue;

    // Type checking is stuck.
    maybeContra->becomesEffect(this) << "failed to type check";
    return maybeContra;
  }

  if (!isValid()) {
    // For some unreported reason, the context is/has become invalid.
    auto contra = this->fatal(this);
    contra << "typing is invalid";
    LLVM_DEBUG(print(llvm::dbgs()););
    return contra;
  }

  return std::nullopt;
}

void FixPointTypeChecker::print(llvm::raw_ostream &os) const {
  OpPrintingFlags flags{};
  flags.skipRegions();
  flags.elideLargeElementsAttrs();
  flags.elideLargeResourceString();

  MLIRTypeChecker::print(os);
  os << "invalid = [\n";
  for (auto *op : invalid) {
    os << "  ";
    op->print(os, flags);
    os << "\n";
  }
  os << "]\n";
}

void FixPointTypeChecker::dump() const {
  print(llvm::errs());
}
