//===- TypeChecker.h - MLIR type checker ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TYPING_TYPECHECKER_H
#define MLIR_TYPING_TYPECHECKER_H

#include "mlir/Typing/Context.h"
#include "mlir/Typing/Contradiction.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::Typing {

inline Evidence &operator<<(Evidence &evidence, const Bound &bound)
{
  return evidence << "{" << bound.getValue() << "}";
}

/// Holds either the resulting Bound of a meet operation, or a Contradiction.
///
/// Since trivially unattainable bounds have a default-constructed distinct
/// state, and are valid interpretations of all failed meets, Bound is used as
/// the base and the discriminator instead of as an alternative.
///
/// FIXME: This should be an std::expected<Bound, Contradiction> in C++23.
class MeetResult : public Bound {
public:
  /// Initializes a successful MeetResult from @p bound .
  /*implicit*/ MeetResult(Bound bound) : Bound(bound), success(0) {}
  /// Initializes a failed MeetResult from @p contra .
  /*implicit*/ MeetResult(Contradiction contra)
      : Bound(), contra(std::move(contra)) {}

  /*implicit*/ MeetResult(MeetResult &&move) : Bound(move) {
    if (move)
      success = 0;
    else
      contra = std::move(move.contra);
  }
  MeetResult &operator=(MeetResult &&move) {
    if (!static_cast<bool>(*this)) contra.~Contradiction();
    static_cast<Bound &>(*this) = move;
    if (!move) contra = std::move(move.contra);
    return *this;
  }

  ~MeetResult() {
    if (static_cast<bool>(*this)) return;
    contra.~Contradiction();
  }

  /// Obtains the contained Contradiction, if any.
  ///
  /// This operation leaves the MeetResult in an moved-from state if it had a
  /// Contradiction, otherwise it remains valid. The intended use is as follows:
  ///
  /// ```c++
  /// if (auto maybeContra = meet.toContra()) {
  ///   return maybeContra->reportIfFatal(here);
  /// }
  /// Type sample = meet.getValue();
  /// ```
  [[nodiscard]] std::optional<Contradiction> toContra() {
    if (static_cast<bool>(*this)) return std::nullopt;
    return std::optional<Contradiction>(std::in_place, std::move(contra));
  }

private:
  union {
    int success;
    Contradiction contra;
  };
};

/// Interface for implementing an MLIR type checker.
///
/// The type checker provides the API that typing rules implemented by the
/// TypeCheckOpInterface use to make their deductions. In particular, a type
/// checker provides a Context, which contains the most restrictive inferred
/// bounds on all IR values.
class AbstractTypeChecker : public Context {
public:
  /// Meets the bound for @p value with @p bound.
  ///
  /// Using the type system of the owner of @p value , meets the current type
  /// bound with @p bound , updates it (also in the context), and returns a
  /// Contradiction that indicates how the context has changed:
  ///
  ///   - If the resulting bound is unattainable, the context is now invalid,
  ///     and the result indicates a fatal error.
  ///   - If the resulting bound is equal to the current bound, the result has
  ///     an indeterminate Contradiction that indicates no progress was made.
  ///   - Otherwise, the result indicates success.
  ///
  /// @param              value Value.
  /// @param              bound Incoming Bound.
  ///
  /// @pre    `value`
  ///
  /// @return A MeetResult instance that is in one of the 3 result states.
  virtual MeetResult meet(Value value, Bound bound) = 0;

  /// Invalidates the deductions of @p op .
  ///
  /// If @p op implements the TypeCheckOpInterface, this method marks its
  /// deductions as outdated. This means that its typing rule will have to be
  /// visited again before a valid typing is found.
  ///
  /// @param  [in]      op  Operation.
  ///
  /// @pre    `op`
  virtual void invalidate(Operation *op) = 0;

  /// Creates a fatal Contradiction that will be handled by this type checker.
  ///
  /// @param              at  Source.
  ///
  /// @pre    `at`
  Contradiction fatal(Source at) {
    assert(at && "provide source or use default-constructor Contradiction{}");
    return Contradiction(this, at);
  }

private:
  /// Attempts to handle a Contradiction detected during type checking.
  ///
  /// @param              loc     Default Location.
  /// @param  [in]      contra  Contradiction.
  ///
  /// @return LogicalResult
  virtual LogicalResult handle(Location loc, Contradiction &&contra);

  friend class Contradiction;
};

/// Base class for implemeting an MLIR type checker.
///
/// This default implementation uses an llvm::DenseMap to store the typing, and
/// thus ensures that meet operations are properly tracked. It stores a counter
/// that tracks the validity of the resulting context, and invalidates all users
/// of updated values automatically.
class MLIRTypeChecker : public AbstractTypeChecker {
public:
  using storage_type = llvm::DenseMap<Value, Bound>;
  using value_type = storage_type::value_type;
  using size_type = storage_type::size_type;
  using iterator = storage_type::const_iterator;

  /// @copydoc Context::get(Value)
  [[nodiscard]] Bound get(Value value) const final {
    const auto it = impl.find(value);
    if (it != impl.end()) return it->second;
    return Context::get(value);
  }

  /// @copydoc Context::isValid()
  [[nodiscard]] bool isValid() const override { return numUnattainable == 0U; }

  /// Assigns @p bound to @p value .
  ///
  /// Updates the internal unattainable bounds counter, and thus allows healing
  /// an invalid context state. Automatically invalidates all users of @p value
  /// if the context changed.
  ///
  /// @param              value Value.
  /// @param              bound Bound for @p value .
  ///
  /// @pre    `value`
  ///
  /// @return Whether the context has changed.
  bool assign(Value value, Bound bound);

  /// @copydoc AbstractTypeChecker::meet(Value, Bound)
  ///
  /// Automatically invalidates all users of @p value if the context changed.
  MeetResult meet(Value value, Bound bound) override;

  /// @copydoc AbstractTypeChecker::invalidate(Operation *)
  void invalidate(Operation *op) override {}

  /// Writes a textual representation of the context to @p os .
  void print(llvm::raw_ostream &os) const;
  /// Writes a textual representation of the context to `llvm::errs()`.
  void dump() const;

  [[nodiscard]] bool empty() const { return impl.empty(); }
  [[nodiscard]] size_type size() const { return impl.size(); }
  [[nodiscard]] iterator begin() const { return impl.begin(); }
  [[nodiscard]] iterator end() const { return impl.end(); }

public:
  storage_type impl;
  unsigned numUnattainable = 0U;
};

} // namespace mlir::Typing

#endif // MLIR_TYPING_TYPECHECKER_H
