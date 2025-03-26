//===- TypeCheck.cpp - Type checking pass ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Typing/FixPointTypeChecker.h"
#include "mlir/Typing/Transmutation.h"
#include "mlir/Typing/TypeCheckOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir{

namespace {

#define GEN_PASS_DEF_TYPECHECK
#include "mlir/Transforms/Passes.h.inc"

struct TypeCheckPass : public impl::TypeCheckBase<TypeCheckPass> {
  using impl::TypeCheckBase<TypeCheckPass>::TypeCheckBase;

  void runOnOperation() override
  {
    if (maxIterations == 0) maxIterations = Typing::FixPointTypeChecker::kNoLimit;
    Typing::FixPointTypeChecker typeChecker(getOperation(), maxIterations);

    if (failed(typeChecker.solve(getOperation()->getLoc()))) {
      signalPassFailure();
      return;
    }

    if (!allowTransmute) return;

    IRRewriter rewriter(getOperation());
    for (auto [value, bound] : typeChecker) {
      std::ignore = Typing::transmute(
        typeChecker,
        rewriter,
        value,
        bound,
        allowPromote);
    }
  }
};

} // namespace

} // namespace mlir

std::unique_ptr<Pass> mlir::createTypeCheckPass()
{
  return std::make_unique<TypeCheckPass>();
}
