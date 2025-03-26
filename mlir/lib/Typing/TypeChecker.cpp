//===- TypeChecker.cpp - MLIR type checker ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Typing/TypeChecker.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Typing/Contradiction.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE  "mlir-type-checker"

using namespace mlir;
using namespace mlir::Typing;

//===----------------------------------------------------------------------===//
// AbstractTypeChecker implementation
//===----------------------------------------------------------------------===//

LogicalResult AbstractTypeChecker::handle(Location loc, Contradiction &&contra) {
  // Simply emit the diagnostic and abort type checking.
  emit(loc, contra);
  return failure();
}

//===----------------------------------------------------------------------===//
// MLIRTypeChecker implementation
//===----------------------------------------------------------------------===//

bool MLIRTypeChecker::assign(Value value, Bound bound) {
  assert(value);

  if (!bound) ++numUnattainable;

  const auto [it, added] = impl.try_emplace(value, bound);
  if (!added) {
    if (it->second == bound) return false;
    if (!it->second) --numUnattainable;
    it->second = bound;
  }

  for (auto *user : value.getUsers()) invalidate(user);
  return true;
}

MeetResult MLIRTypeChecker::meet(Value value, Bound bound) {
  assert(value);

  // Ensure that there is an entry for value in the map. If not, initialize it
  // with the IR-carried constraint via the Context::get method.
  auto &current = impl.try_emplace(value, Context::get(value)).first->second;

  auto *dialect = getOwner(value)->getDialect();
  const auto result = Bound::meet(getTypeSystem(dialect), current, bound);

  LLVM_DEBUG({
    llvm::dbgs() << "[MLIRTypeChecker] ";
    value.printAsOperand(llvm::dbgs(), OpPrintingFlags{});
    llvm::dbgs() << " : " << current << " (*) " << bound << " = " << result << "\n";
  });

  if (!result) {
    // The bound is unattainable and thus the context becomes invalid.
    auto contra = this->fatal(value);
    contra << bound << " does not meet requirement " << current;
    contra.attachNote(dialect) << "under '" << dialect->getNamespace() << "' dialect";
    current = {};
    ++numUnattainable;
    return contra;
  }

  if (current == result) {
    // But no progress was made.
    return Contradiction{};
  }

  current = result;

  // Progress was made on the value, so all its users are invalidated.
  for (auto *user : value.getUsers()) invalidate(user);
  return current;
}

void MLIRTypeChecker::print(llvm::raw_ostream &os) const {
  OpPrintingFlags flags{};
  flags.skipRegions();
  flags.elideLargeElementsAttrs();
  flags.elideLargeResourceString();

  os << "MLIRTypeChecker {\n";
	for (auto [value, bound] : impl) {
    os << "  ";
		value.printAsOperand(os, flags);
		os << " : " << bound << "\n";
	}
  os << "}\n";
}

void MLIRTypeChecker::dump() const {
  print(llvm::errs());
}
