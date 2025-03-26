//===- Bound.h - Subtype type bound -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TYPING_BOUND_H
#define MLIR_TYPING_BOUND_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/PointerIntPair.h"

#include <cstdint>

namespace mlir {

class Diagnostic;
class AbstractTypeSystem;

} // namespace mlir

namespace mlir::Typing {

/// Represents a subtyping bound on a type.
///
/// A bound is placed on some type `x` by defining its relation to some concrete
/// reference type `v` called the bound value. Bounds can have the following
/// representations, in descending order of strictness:
///
///   - Identity:     x == v    "x must be definitionally equal to v"
///   - Equivalence:  x ~= v    "x must be equivalent to v"
///   - Lower:        x :> v    "x must be a supertype of v"
///   - Upper:        x <: v    "x must be a subtype of v"
///
/// A bound constructed without a bound value is unattainable. In particular, a
/// default-constructed bound is unattainable. A bound constructed with a bound
/// value is always trivially attained by that value.
///
/// The Bound type is layout-compatible to mlir::Type, and due to the above
/// properties, can be used as its own sample.
///
/// A bound can only be tested under some subtype relation defined by an
/// AbstractTypeSystem.
struct Bound {
  // This type must be layout compatible to `mlir::Type`.
  using Impl = llvm::PointerIntPair<Type, 2, std::intptr_t>;

  /// Distingiushes the different kinds of type bounds.
  ///
  /// We use bit fields to indicate the type of test we need to perform. To
  /// ensure that the default-constructed instance is an Identity bound on
  /// the nullptr, 00 must enable all tests.
  ///
  /// To determine whether a test should take place, bitwise AND can be used.
  enum class Kind : std::intptr_t {
    /// Matches definitionally equal types.
    Identity    = 0b00,
    /// Matches supertypes of the bound value.
    Lower       = 0b01,
    /// Matches subtypes of the bound value.
    Upper       = 0b10,
    /// Matches types equivalent to the bound value.
    Equivalence = Lower | Upper
  };

  /// Obtains an equivalence bound.
  [[nodiscard]] static Bound equivalent(Type value) {
    return Bound(Kind::Equivalence, value);
  }
  /// Obtains a lower bound.
  [[nodiscard]] static Bound lower(Type value) {
    return Bound(Kind::Lower, value);
  }
  /// Obtains an upper bound.
  [[nodiscard]] static Bound upper(Type value) {
    return Bound(Kind::Upper, value);
  }

  /// Initializes an unattainable bound.
  ///
  /// @post   `!static_cast<bool>(result)`
  /*implicit*/ Bound() : impl() {}
  /// Initializes a bound that matches only @p value .
  ///
  /// @post   `static_cast<bool>(result) == !!value`
  /*implicit*/ Bound(Type value)
    : Bound(Kind::Identity, value) {}

  /// Gets the bound value.
  ///
  /// If the value is present, it trivially matches the bound. If it is not
  /// present, then the bound is known to be unattainable.
  [[nodiscard]] Type getValue() const { return impl.getPointer(); }
  /// Gets the bound Kind.
  [[nodiscard]] Kind getKind() const
  {
    return static_cast<Kind>(impl.getInt());
  }

  /// Determines whether this is an identity constraint.
  [[nodiscard]] bool isIdentity() const
  {
    return getKind() == Kind::Identity;
  }
  /// Determines whether this is an equivalence constraint.
  [[nodiscard]] bool isEquivalence() const
  {
    return getKind() == Kind::Equivalence;
  }
  /// Determines whether this includes an upper bound.
  [[nodiscard]] bool isUpper() const
  {
    constexpr auto mask = static_cast<std::intptr_t>(Kind::Upper);
    return (static_cast<std::intptr_t>(getKind()) | mask) == mask;
  }
  /// Determines whether this includes a lower bound.
  [[nodiscard]] bool isLower() const
  {
    constexpr auto mask = static_cast<std::intptr_t>(Kind::Lower);
    return (static_cast<std::intptr_t>(getKind()) | mask) == mask;
  }

  /// Determines whether @p type matches the bound under the @p typeSystem .
  ///
  /// @param  [in]      typeSystem  AbstractTypeSystem.
  /// @param              type        Type to test against the bound.
  ///
  /// @return Whether @p type matches the bound under the @p typeSystem .
  ///
  /// @post   `type || !result`
  [[nodiscard]] bool test(
    const AbstractTypeSystem &typeSystem,
    Type type) const;

  /// Meets the bounds @p lhs and @p rhs to produce a more refined bound.
  ///
  /// This method uses the subtype relation of the @p typeSystem to produce a
  /// new bound from two given bounds that is more restrictive. The meet
  /// operator "a (*) b = c" is uniquely defined by its properties:
  ///
  ///   - Idempotency:    a (*) a = a
  ///   - Commutativity:  a (*) b = b (*) a
  ///   - Subsumption:    a (*) b = c -> x in c -> x in a, x in b
  ///
  /// and the following implications:
  ///
  /// ```
  /// x = v_a, rel(x, v_b)  -> c = { v_a            v_a = v_b
  ///                              { unattainable   otherwise
  /// x ~= v_a, rel(x, v_b) -> c = { ~= v_a         rel(v_a, v_b)
  ///                              { unattainable   otherwise
  /// x :> v_a, x :> v_b    -> c = { :> v_a         v_a :> v_b
  ///                              { :> v_b         v_b :> v_a
  ///                              { unattainable   otherwise
  /// x :> v_a, x <: v_b    -> c = { ~= v_a         v_a <: v_b, v_b <: v_a
  ///                              { :> v_a         v_a <: v_b, v_b </: v_a
  ///                              { unattainable   otherwise
  /// x <: v_a, x <: v_b    -> c = { <: v_a         v_a <: v_b
  ///                              { <: v_b         v_b <: v_a
  ///                              { unattainable   otherwise
  /// ```
  ///
  /// For ease of use, the result will always be the default-constructed bound
  /// if it is known to be unattainable, allowing `if (auto c = meet(ts, a, b))`
  /// constructs.
  ///
  /// @param  [in]      typeSystem  AbstractTypeSystem.
  /// @param              lhs         Left operand.
  /// @param              rhs         Right operand.
  ///
  /// @return The most restrictive intersection of @p lhs and @p rhs .
  [[nodiscard]] static Bound meet(
    const AbstractTypeSystem &typeSystem,
    Bound lhs,
    Bound rhs);

  /// Writes a textual description of the bound to @p os .
  void print(llvm::raw_ostream &os) const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Bound &bound) {
    bound.print(os);
    return os;
  }

  /// @copydoc getValue()
  /*implicit*/ operator Type() const { return getValue(); }
  /// Determines whether this bound is not the default-constructed bound.
  explicit operator bool() const { return impl.getOpaqueValue() != nullptr; }

  /// Determines whether two bounds are definitionally equal.
  [[nodiscard]] bool operator==(const Bound &) const = default;

private:
  Bound(Kind kind, Type value)
    : impl(value, static_cast<std::intptr_t>(kind)) {}

  Impl impl;
};

} // namespace mlir::Typing

#endif // MLIR_TYPING_BOUND_H
