//===- Transmutation.h - Value type transmutation ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TYPING_TRANSMUTATION_H
#define MLIR_TYPING_TRANSMUTATION_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {

class OpBuilder;
class RewriterBase;

} // namespace mlir

namespace mlir::Typing {

  class Context;

/// Determines whether @p use may be transmuted to a subtype.
///
/// @param  [in]        use OpOperand.
///
/// @returns  Whether the owner of @p use permits transmutation.
[[nodiscard]] bool mayTransmute(const OpOperand &use);

/// Attempts to insert a promoting cast for @p value to @p superType .
///
/// This method uses the AbstractTypeSystem belonging to the owner of @p value
/// under @p context to try and insert a promoting cast for @p value to the
/// @p superType . This operation is fallible.
///
/// @param  [in]        context   Context.
/// @param              builder   OpBuilder.
/// @param              loc       Location.
/// @param              value     Value to promote.
/// @param              superType Supertype to promote to.
///
/// @pre    `value && superType`
/// @pre    `context.getTypeSystem(value).isSubtype(value.getType(), superType)`
///
/// @retval `nullptr` Unable to promote @p value .
/// @retval Operation Inserted promotion cast.
Operation *promote(
  const Context &context,
  OpBuilder &builder,
  Location loc,
  Value value,
  Type superType);

/// Attempts to transmute the type of @p value to @p into .
///
/// This method uses the TypeCheckOpInterface implementation of the owner of
/// @p value to attempt an in-place transmutation of its type to @p subType .
/// In cases where uses of @p value do not support transmutation (see
/// mlir::Typing::mayTransmute(const OpOperand &)), this requires promotion.
///
/// If @p allowPromote is @c true , the AbstractTypeSystem belonging to the
/// owner of @p value is used to attempt a promoting cast to derive a value of
/// the pre-transmuted type. If this succeeds, transmutation may still succeed,
/// leaving all non-transmutable uses attached to the promoting cast.
///
/// @param  [in]        context       Context.
/// @param              rewriter      RewriterBase.
/// @param              value         Value to transmute the type of.
/// @param              subType       Subtype to transmute to.
/// @param              allowPromote  Whether promoting casts may be inserted.
///
/// @pre    `value && subType`
/// @pre    `context.getTypeSystem(value).isSubtype(subType, value.getType())`
///
/// @returns  Whether the IR was modified.
///
/// @post   `failure(result) || value.getType() == into`
LogicalResult transmute(
  const Context &context,
  RewriterBase &rewriter,
  Value value,
  Type subType,
  bool allowPromote = true);

} // namespace mlir::Typing

#endif // MLIR_TYPING_TRANSMUTATION_H
