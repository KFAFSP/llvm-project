//===- FixPointTypeChecker.h - Fix point type checker -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Typing/TypeChecker.h"
#include "mlir/Typing/LexicalOpQueue.h"

#include <limits>

#ifndef MLIR_TYPING_FIXPOINTTYPECHECKER_H
#define MLIR_TYPING_FIXPOINTTYPECHECKER_H

namespace mlir::Typing {

/// Implements an MLIRTypeChecker that finds a fix point of the ruleset.
///
/// Using a naive worklist algorithm, this type checker continues to apply rules
/// defined by TypeCheckOpInterface ops until a fix point is reached. This is
/// detected by the worklist of invalid operations becoming empty. To initialize
/// the worklist, `invalidateAll` or the appropriate constructor should be used.
///
/// Bounded execution is guaranteed if the advisories given on the declaration
/// of the TypeCheckOpInterface::typeCheck method are observed. Users can also
/// specify an iteration limit that terminates execution if any operation is
/// visited more times than the limit allows.
class FixPointTypeChecker : public MLIRTypeChecker {
public:
  /// Constant that indicates no iteration limit.
  static constexpr auto kNoLimit = std::numeric_limits<std::size_t>::max();

  /// Initializes a FixPointTypeChecker with @p iterationLimit .
  ///
  /// @param              iterationLimit  Limit for operation rule applications.
  ///
  /// @pre    `iterationLimit > 0U`
  explicit FixPointTypeChecker(std::size_t iterationLimit = kNoLimit)
      : invalid(), iterationLimit(iterationLimit), iterations() {
    assert(iterationLimit > 0U);
  }
  /// Initializes a FixPointTypeChecker with @p iterationLimit below @p root .
  ///
  /// Invalidates all `TypeCheckOpInterface` ops below and including @p root .
  ///
  /// @param              root            Root Operation.
  /// @param              iterationLimit  Limit for operation rule applications.
  ///
  /// @pre    `root`
  /// @pre    `iterationLimit > 0U`
  explicit FixPointTypeChecker(
    Operation *root,
    std::size_t iterationLimit = kNoLimit)
      : FixPointTypeChecker(iterationLimit) {
    assert(root);

    invalidateAll(root);
  }

  /// Obtains a value indicating whether the typing is solved.
  [[nodiscard]] bool isSolved() const { return invalid.empty(); }

  /// @copydoc AbstractTypeChecker::invalidate(Operation *)
  void invalidate(Operation *op) override;
  /// Invalidates all ops below and including @p root .
  ///
  /// @param              root            Root Operation.
  ///
  /// @pre    `root`
  void invalidateAll(Operation *root);

  /// Attempts to solve the typing for all invalid operations.
  std::optional<Contradiction> trySolve();
  /// Solves the typing for all invalid operations and reports errors.
  ///
  /// @param              defaultLoc  Default Location.
  ///
  /// @retval `success` A valid typing was found and no errors reported.
  /// @retval `failure` No valid typing was found, all errors were reported.
  LogicalResult solve(Location defaultLoc) {
    if (auto maybeContra = trySolve())
      return maybeContra->handle(defaultLoc);
    return success();
  }

  /// Writes a textual representation of the state to @p os .
  void print(llvm::raw_ostream &os) const;
  /// Writes a textual representation of the state to `llvm::errs()`.
  void dump() const;

private:
  LexicalOpQueue invalid;
  std::size_t iterationLimit;
  llvm::DenseMap<Operation *, std::size_t> iterations;
};

} // namespace mlir::Typing

#endif // MLIR_TYPING_FIXPOINTTYPECHECKER_H
