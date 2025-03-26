//===- TypeSystem.h - MLIR Type System --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_TYPE_SYSTEM_H
#define MLIR_IR_TYPE_SYSTEM_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <compare>
#include <memory>

namespace mlir {

class OpBuilder;

/// Base class for an MLIR type system.
///
/// Provides methods that define relations and operators on types used for the
/// purposes of type checking. Implementations must take care to satisfy all the
/// requirements as documented on the virtual functions.
class AbstractTypeSystem {
public:
  /// Disambiguates between two equivalent types @p lhs and @p rhs .
  ///
  /// To stably order two equivalent types @p lhs and @p rhs , we compare the
  /// pointers to their TypeIDs and pick the smaller one. This ordering is not
  /// necessarily stable across MLIR invocations. However, it follows the same
  /// line of reasoning that mlir::OpTrait::Commutative uses.
  ///
  /// @param              lhs First type.
  /// @param              rhs Second type.
  ///
  /// @pre    `lhs && rhs`
  ///
  /// @return Type
  ///
  /// @post   `result == disambiguate(rhs, lhs)`
  [[nodiscard]] static Type disambiguate(Type lhs, Type rhs) {
    assert(lhs && rhs);

    if (lhs.getTypeID().getAsOpaquePointer()
        > rhs.getTypeID().getAsOpaquePointer())
      return rhs;
    return lhs;
  }

  virtual ~AbstractTypeSystem();

  //===------------------------------------------------------------------===//
  // Subtype relation
  //===------------------------------------------------------------------===//

  /// Determines whether @p sub is a subtype of @p super .
  ///
  /// Implements the subtype relation "T <: U" (read as: "T is a subtype of U" /
  /// "U subsumes T"). It has the following properties:
  ///
  ///   - Reflexivity:  T <: T
  ///   - Transitivity: X <: Y, Y <: Z -> X <: Z
  ///
  /// @param              sub   Subtype.
  /// @param              super Supertype.
  ///
  /// @pre    `sub && super`
  ///
  /// @return Whether @p sub is a subtype of @p super .
  ///
  /// @post   `sub != super || result`
  [[nodiscard]] virtual bool isSubtype(Type sub, Type super) const = 0;

  /// Compares two types @p lhs and @p rhs using the subtype relation.
  ///
  /// This method performs a 4-way comparison of @p lhs and @p rhs using the
  /// isSubtype(Type, Type) method. Comparing T and U, the possible results
  /// (represented by std::partial_ordering) are:
  ///
  ///   - T </: U and U </: T -> T and U are unrelated (unordered)
  ///   - T  <: U and U </: T -> T is a proper subtype of U (less)
  ///   - T </: U and U  <: T -> T is a proper supertype of U (greater)
  ///   - T  <: U and U  <: T -> T and U are equivalent (equivalent)
  ///
  /// @param              lhs First type.
  /// @param              rhs Second type.
  ///
  /// @pre    `lhs && rhs`
  ///
  /// @retval unordered   @p lhs and @p rhs are unrelated.
  /// @retval less        @p lhs is a proper subtype of @p rhs .
  /// @retval greater     @p lhs is a proper supertype of @p rhs .
  /// @retval equivalent  @p lhs and @p rhs are subtypes of each other.
  ///
  /// @post   `!unordered || (!isSubtype(lhs, rhs) && !isSubtype(rhs, lhs))`
  /// @post   `!less || (isSubtype(lhs, rhs) && !isSubtype(rhs, lhs))`
  /// @post   `!greater || (!isSubtype(lhs, rhs) && isSubtype(rhs, lhs))`
  /// @post   `!equivalent || (isSubtype(lhs, rhs) && isSubtype(rhs, lhs))`
  [[nodiscard]] std::partial_ordering compare(Type lhs, Type rhs) const;

  /// Determines the "smallest" type between @p lhs and @p rhs .
  ///
  /// This method obtains the type that is the subtype of the other, or the
  /// absent type @c nullptr if they are unrelated. In addition, it satisfies:
  ///
  ///   - Idempotency:    min(a, a) = a
  ///   - Commutativity:  min(a, b) = min(b, a)
  ///   - Associativity:  min(min(a, b), c) = min(a, min(b, c))
  ///
  /// In particular, if a <: b and b <: a, a type is chosen deterministically
  /// such that the result does not depend on the order of operands.
  ///
  /// @param              lhs First type.
  /// @param              rhs Second type.
  ///
  /// @pre    `lhs && rhs`
  ///
  /// @post   `lhs != rhs || result == lhs`
  /// @post   `result == min(rhs, lhs)`
  /// @post   `!result || (isSubtype(result, lhs) && isSubtype(result, rhs))`
  [[nodiscard]] Type min(Type lhs, Type rhs) const;

  /// Determines the "largest" type between @p lhs and @p rhs .
  ///
  /// This function is analogous to min(Type, Type), see that method for more
  /// information.
  ///
  /// @param              lhs First type.
  /// @param              rhs Second type.
  ///
  /// @pre    `lhs && rhs`
  ///
  /// @post   `lhs != rhs || result == lhs`
  /// @post   `result == max(rhs, lhs)`
  /// @post   `!result || (isSubtype(lhs, result) && isSubtype(rhs, result))`
  [[nodiscard]] Type max(Type lhs, Type rhs) const;

  //===------------------------------------------------------------------===//
  // Promotion operator
  //===------------------------------------------------------------------===//

  /// Obtains the common supertype of @p lhs and @p rhs .
  ///
  /// Implements the type promotion operator "T |_| U". This function may be
  /// partial, i.e., it may return @c nullptr to indicate that it is not defined
  /// for this pair. It has the following properties:
  ///
  ///   - Idempotency:   T |_| T = T
  ///   - Commutativity: T |_| U = U |_| T
  ///   - Associativity: (T |_| U) |_| V = T |_| (U |_| V)
  ///                    provided T, U, V promote
  ///   - Subsumption:   T |_| U = V -> T <: V, U <: V
  ///
  /// @param              lhs First type.
  /// @param              rhs Second type.
  ///
  /// @pre    `lhs && rhs`
  ///
  /// @return Type
  ///
  /// @post   `!result || compare(lhs, rhs) == std::partial_ordering::unordered`
  /// @post   `!result || (isSubtype(lhs, result) && isSubtype(rhs, result))`
  [[nodiscard]] virtual Type promote(Type lhs, Type rhs) const = 0;

  /// Attempts to create a promoting cast operation.
  ///
  /// @param              builder   OpBuilder.
  /// @param              loc       Location.
  /// @param              input     Value to promote.
  /// @param              superType Cast result type.
  ///
  /// @pre    `input && superType`
  /// @pre    `isSubtype(input.getType(), superType)`
  ///
  /// @retval `nullptr` Can't promote @p input to @p resultType .
  /// @retval Operation Cast that promotes @p input to @p resultType .
  ///
  /// @post   `!result || result->getNumResults() == 1`
  /// @post   `!result || result->getResult(0).getType() == superType`
  [[nodiscard]] virtual Operation *promote(
    OpBuilder &builder,
    Location loc,
    Value input,
    Type superType) const = 0;

  /// Attempts to reduce @p types by promotion as much as possible.
  ///
  /// Using the promote(Type, Type) function, the @p types will be reduced
  /// pairwise until no further reductions are possible. The order in which
  /// these reductions are attempted is deterministic and lexicographical.
  ///
  /// The following postconditions are met:
  ///
  ///   - The size of @p types can never increase.
  ///   - If @p types contains exactly one type, it is a supertype of all
  ///     original input types.
  ///   - No pair in @p types is related (the @p types are irreducible).
  ///
  /// @param  [in,out]    types Work list.
  ///
  /// @pre    `llvm::count(types, Type{}) == 0`
  ///
  /// @post   `llvm::count(types, Type{}) == 0`
  /// @post   `types.size() <= typesPre.size()`
  /// @post   `types.size() != 1 || types[0] :> typesPre[i]`
  /// @post   `compare(types[i], types[j]) == std::partial_ordering::unordered`
  void promote(SmallVectorImpl<Type> &types) const;

  /// Attempts to promote @p types to a single type.
  ///
  /// Using the promote(SmallVectorImpl<Type>) function, @p types will be
  /// promoted to a single common supertype, if possible.
  ///
  /// @param              types Types.
  ///
  /// @pre    `llvm::count(types, Type{}) == 0`
  ///
  /// @retval failure     @p types can't be promoted.
  /// @retval Type        Common supertype of @p types .
  ///
  /// @post   `!types.empty() || failed(result)`
  /// @post   `failed(result) || *result :> types[i]`
  /// @post   `*result != nullptr`
  FailureOr<Type> promote(ArrayRef<Type> types) const;
};

/// Implements an AbstractTypeSystem based only on definitional equality.
class EqualityTypeSystem final : public AbstractTypeSystem {
public:
  /// Gets the canonical singleton instance.
  static EqualityTypeSystem &getInstance();

  /// @copydoc AbstractTypeSystem::isSubtype(Type, Type)
  [[nodiscard]] bool isSubtype(Type sub, Type super) const final {
    return sub == super;
  }
  /// @copydoc AbstractTypeSystem::promote(Type, Type)
  [[nodiscard]] Type promote(Type lhs, Type rhs) const final {
    return lhs == rhs ? lhs : Type{};
  }
  /// @copydoc AbstractTypeSystem::createPromotion(OpBuilder &, Location, Value, Type)
  Operation *promote(OpBuilder &, Location, Value, Type) const final {
    return nullptr;
  }

private:
  // Ensure nobody accidentally creates or moves instances instead of using the
  // singleton. This should help with debugging.
  EqualityTypeSystem() = default;
  EqualityTypeSystem(const EqualityTypeSystem &) = delete;
  EqualityTypeSystem(EqualityTypeSystem &&) = delete;
};

/// Memoizes an AbstractTypeSystem.
class CachedTypeSystem final : public AbstractTypeSystem {
public:
  explicit CachedTypeSystem(const AbstractTypeSystem &base);
  ~CachedTypeSystem();

  /// @copydoc AbstractTypeSystem::isSubtype(Type, Type)
  [[nodiscard]] bool isSubtype(Type sub, Type super) const final;
  /// @copydoc AbstractTypeSystem::promote(Type, Type)
  [[nodiscard]] Type promote(Type lhs, Type rhs) const final;
  /// @copydoc AbstractTypeSystem::createPromotion(OpBuilder &, Location, Value, Type)
  Operation *promote(
    OpBuilder &builder,
    Location loc,
    Value value,
    Type superType) const final {
    return base.promote(builder, loc, value, superType);
  }

  /// Gets the underlying AbstractTypeSystem.
  const AbstractTypeSystem &getBase() const { return base; }

private:
  struct Impl;
  const AbstractTypeSystem &base;
  std::unique_ptr<Impl> impl;
};

/// Interface for a dialect that defines a type system.
///
/// Provides an interface for dialects to implement an AbstractTypeSystem by
/// overriding a minimal number of methods. Provides default implementations
/// that use only definitional equality.
class DialectTypeSystemInterface
    : public DialectInterface::Base<DialectTypeSystemInterface>,
      public AbstractTypeSystem {
public:
  /// Initializes an interface model for @p dialect .
  ///
  /// @param  [in]        dialect Dialect.
  explicit DialectTypeSystemInterface(Dialect *dialect) : Base(dialect) {}

  virtual ~DialectTypeSystemInterface();

  //===------------------------------------------------------------------===//
  // Subtype relation
  //===------------------------------------------------------------------===//

  /// Determines whether @p sub is a subtype of @p super .
  ///
  /// Implements the dialect-specific subtype relation. See
  /// AbstractTypeSystem::isSubtype(Type, Type) for more information.
  ///
  /// When overriding this function, the implementation may assume that both
  /// @p sub and @p super are non-null. The type system may memoize this
  /// relation, and may exploit its properties.
  ///
  /// @param              sub   Subtype.
  /// @param              super Supertype.
  ///
  /// @pre    `sub && super`
  ///
  /// @return Whether @p sub is a subtype of @p super .
  ///
  /// @post   `sub != super || result`
  [[nodiscard]] bool isSubtype(Type sub, Type super) const override {
    // Type identity fulfills our requirements and preserves the status quo.
    return sub == super;
  }

  //===------------------------------------------------------------------===//
  // Promotion operator
  //===------------------------------------------------------------------===//

  /// Obtains the common supertype of @p lhs and @p rhs .
  ///
  /// Implements the dialect-specific type promotion operator. See
  /// AbstractTypeSystem::promote(Type, Type) for more information.
  ///
  /// When overriding this function, the implementation may assume that both
  /// @p lhs and @p rhs are non-null. The type checker may memoize this
  /// function, and may exploit its properties.
  ///
  /// @param              lhs First type.
  /// @param              rhs Second type.
  ///
  /// @pre    `lhs && rhs`
  ///
  /// @return Type
  ///
  /// @post   `!result || compare(lhs, rhs) == std::partial_ordering::unordered`
  /// @post   `!result || (isSubtype(lhs, result) && isSubtype(rhs, result))`
  [[nodiscard]] Type promote(Type lhs, Type rhs) const override {
    // This trivial implementation fulfills our requirements.
    if (isSubtype(lhs, rhs)) return rhs;
    if (isSubtype(rhs, lhs)) return lhs;
    return Type{};
  }

  /// Attempts to create a promoting cast operation.
  ///
  /// Implements the dialect-specific cast op builder. See
  /// AbstractTypeSystem::promote(OpBuilder &, Location, Value, Type) for more
  /// information.
  ///
  /// When overriding this function, the implementation may assume that @p input
  /// and @p superType are non-null, and that @p superType is a proper supertype
  /// of the current type of @p input .
  ///
  /// @param              builder   OpBuilder.
  /// @param              loc       Location.
  /// @param              input     Value to promote.
  /// @param              superType Cast result type.
  ///
  /// @pre    `input && superType`
  /// @pre    `isSubtype(input.getType(), superType)`
  ///
  /// @retval `nullptr` Can't promote @p input to @p resultType .
  /// @retval Operation Cast that promotes @p input to @p resultType .
  ///
  /// @post   `!result || result->getNumResults() == 1`
  /// @post   `!result || result->getResult(0).getType() == superType`
  [[nodiscard]] Operation *promote(
    OpBuilder &builder,
    Location loc,
    Value input,
    Type superType) const override;
};

/// Provides a type system for arbitrary MLIR programs.
///
/// Dialects and their operations can participate in type checking by way of
/// implementing the appropriate interfaces, the DialectTypeSystemInterface and
/// the TypeCheckOpInterface. Together, they form the type system of their
/// respective dialect.
///
/// Since MLIR programs are represented as a mix of dialects, the type checker
/// needs to combine the constraints of multiple dialects. The main objective
/// of the MLIRTypeSystem is to determine which IR mutations are legal under
/// these constraints.
class MLIRTypeSystem : public DialectInterfaceCollection<DialectTypeSystemInterface> {
public:
  //===------------------------------------------------------------------===//
  // Subtype relation
  //===------------------------------------------------------------------===//

  /// Determines whether @p dialect considers @p sub a subtype of @p super .
  ///
  /// See DialectTypeSystemInterface::isSubtype(Type, Type) for more
  /// information on the subtype relation.
  ///
  /// @param  [in]        dialect Dialect.
  /// @param              sub     Subtype.
  /// @param              super   Supertype.
  ///
  /// @pre    `sub && super`
  ///
  /// @return Whether @p sub is a subtype of @p super in the @p dialect .
  ///
  /// @post   `sub != super || result`
  [[nodiscard]] bool isSubtype(Dialect *dialect, Type sub, Type super) const;

  /// Compares two types @p lhs and @p rhs using the subtype relation of
  /// @p dialect .
  ///
  /// See DialectTypeSystemInterface::compare(Type, Type) for more information
  /// on the comparison.
  ///
  /// @param  [in]        dialect Dialect.
  /// @param              lhs     First type.
  /// @param              rhs     Second type.
  ///
  /// @pre    `lhs && rhs`
  ///
  /// @return The relation between @p lhs and @p rhs according to @p dialect .
  ///
  /// @post   `lhs != rhs || std::is_eq(result)`
  [[nodiscard]] std::partial_ordering
  compare(Dialect *dialect, Type lhs, Type rhs) const;

  //===------------------------------------------------------------------===//
  // Value substitutability
  //===------------------------------------------------------------------===//

  /// Decides whether @p use can be substituted by a value of type @p with .
  ///
  /// Examines the subtype relation to determine whether @p with is a subtype
  /// of the current type of @p use under the dialect that defines the user
  /// operation.
  ///
  /// @param  [in]        use   Use to substitute.
  /// @param              with  Type to substitute with.
  ///
  /// @pre    `with`
  ///
  /// @return Whether @p use is substituable with a value of type @p with .
  ///
  /// @post   `result || use.get().getType() != with`
  [[nodiscard]] bool canSubstitute(const OpOperand &use, Type with) const;

  /// Decides whether @p value can be substituted by a value of type @p with .
  ///
  /// Checks the substitutability of every use of @p value using @p with . See
  /// canSubstitute(const OpOperand &, Type) for more information.
  ///
  /// @param              value Value to substitute.
  /// @param              with  Type to substitute with.
  ///
  /// @pre    `value && with`
  ///
  /// @return Whether @p value is substituable with a value of type @p with .
  ///
  /// @post   `result || use.get().getType() != with`
  [[nodiscard]] bool canSubstitute(Value value, Type with) const;
};

} // namespace mlir

#endif
