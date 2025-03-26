/// Declares the TypeCheckOpInterface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#ifndef MLIR_TYPING_TYPECHECKOPINTERFACE_H
#define MLIR_TYPING_TYPECHECKOPINTERFACE_H

#include "mlir/Typing/Contradiction.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

namespace mlir::Typing {

class AbstractTypeChecker;

} // namespace mlir::Typing

namespace mlir::detail {

/// Performs default argument transmutation.
///
/// @param              arg     BlockArgument.
/// @param              into    Target type.
///
/// @pre    `arg && into`
///
/// @retval success @p arg was transmuted to @p into .
/// @retval failure The IR remains unchanged.
///
/// @post   `failed(result) || arg.getType() == into`
LogicalResult transmuteArgument(BlockArgument arg, Type into);

/// Uses a local MLIRTypeChecker to determine whether @p op is valid.
///
/// @param  [in]        op  Operation.
///
/// @pre    `llvm::isa_and_present<TypeCheckOpInterface>(op)`
///
/// @return Whether verification succeeded.
LogicalResult verifyTypeCheckOpInterface(Operation *op);

} // namespace mlir::detail

//===- Generated includes -------------------------------------------------===//

#include "mlir/Typing/TypeCheckOpInterface.h.inc"

//===----------------------------------------------------------------------===//

#endif // MLIR_TYPING_TYPECHECKOPINTERFACE_H
