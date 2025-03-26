//===- Context.h - Type checking context ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TYPING_CONTEXT_H
#define MLIR_TYPING_CONTEXT_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeSystem.h"
#include "mlir/IR/Value.h"
#include "mlir/Typing/Bound.h"

namespace mlir::Typing {

//===----------------------------------------------------------------------===//
// getOwner
//===----------------------------------------------------------------------===//
//
// The context introduces the concept of value ownership for the purposes of
// type checking. A value's owner is the operation that is responsible for its
// definition, and thus the only operation that is allowed to modify it.
//
// The type system associated with the owner of a value is that value's
// definition type system, which determines if a definition can be replaced.
// Replacement can happen in the form of casting, cloning or transmutation.

/// Gets the operation that owns @p argument .
///
/// @param              argument  BlockArgument.
///
/// @pre    `argument`
///
/// @return Operation owning the block owning @p argument .
///
/// @post   `result`
[[nodiscard]] inline Operation *getOwner(BlockArgument argument) {
  assert(argument);

  return argument.getOwner()->getParentOp();
}
/// Gets the operation that owns @p result .
///
/// @param              result  OpResult.
///
/// @pre    `result`
///
/// @return Operation owning the @p result .
///
/// @post   `result`
[[nodiscard]] inline Operation *getOwner(OpResult result) {
  assert(result);

  return result.getOwner();
}
/// Gets the operation that owns @p value .
///
/// The owner of @p value is the operation that canonically defines its
/// semantics, and thus provides the associated definition type system. Results
/// are owned by their defining operations, and block arguments are owned by
/// the immediate parent of the region that contains the parent block.
///
/// @param              value Value
///
/// @pre    `value`
///
/// @return Operation owning the @p value .
///
/// @post   `result`
[[nodiscard]] inline Operation *getOwner(Value value) {
  assert(value);

  if (const auto result = llvm::dyn_cast<OpResult>(value))
    return getOwner(result);
  return getOwner(llvm::cast<BlockArgument>(value));
}

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

/// Base class for a type checking context.
///
/// An MLIR type checking context stores a mapping of SSA values to types. Since
/// type checking in MLIR is focused on gradual refinement of subtype bounds, a
/// context stores the currently deduced Bound instead. Since Bound instances
/// are self-sampling, they can be used interchangably with MLIR types.
///
/// A context that assings no value an unattainable bound is considered valid.
/// In particular, the default implementation that reads bounds from verified IR
/// always forms a valid context by definition.
///
/// For convenience, the Context also provides access to the definition type
/// system associated with any value. The definition type system of a value is
/// the AbstractTypeSystem associated with its owner. It is used to meet bounds
/// placed on the value.
class Context {
public:
  virtual ~Context() = default;

  /// Obtains the current Bound on the type of @p value .
  ///
  /// @param              value Value.
  ///
  /// @pre    `value`
  ///
  /// @return Bound on @p value .
  ///
  /// @post   incorrect or `result.test(getTypeSystem(value), value.getType())`
  [[nodiscard]] virtual Bound get(Value value) const;

  /// Indicates whether the context is free from unattainable bounds.
  [[nodiscard]] virtual bool isValid() const { return true; }

  /// Obtains the AbstractTypeSystem for @p owner .
  ///
  /// @param              owner Owning IR element.
  ///
  /// @pre    `owner`
  ///
  /// @return AbstractTypeSystem for @p owner .
  [[nodiscard]] virtual const AbstractTypeSystem &getTypeSystem(Dialect *owner) const;

  /// @copydoc getTypeSystem(Dialect *)
  [[nodiscard]] const AbstractTypeSystem &getTypeSystem(Operation *owner) const
  {
    assert(owner);
    return getTypeSystem(owner->getDialect());
  }
  /// @copydoc getTypeSystem(Dialect *)
  [[nodiscard]] const AbstractTypeSystem &getTypeSystem(OpResult owner) const
  {
    assert(owner);
    return getTypeSystem(owner.getOwner());
  }
  /// @copydoc getTypeSystem(Dialect *)
  [[nodiscard]] const AbstractTypeSystem &getTypeSystem(BlockArgument owner) const
  {
    assert(owner);
    return getTypeSystem(owner.getParentRegion()->getParentOp());
  }
  /// @copydoc getTypeSystem(Dialect *)
  [[nodiscard]] const AbstractTypeSystem &getTypeSystem(Value owner) const
  {
    assert(owner);
    if (auto result = llvm::dyn_cast<OpResult>(owner))
      return getTypeSystem(result);
    return getTypeSystem(llvm::cast<BlockArgument>(owner));
  }
};

} // namespace mlir::Typing

#endif // MLIR_TYPING_CONTEXT_H
