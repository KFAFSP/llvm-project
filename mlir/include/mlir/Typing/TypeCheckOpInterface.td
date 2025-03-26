//===- TypeCheckOpInterface.td - Type checking interface ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an interface to allow operations to participate in
// unified type checking.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TYPECHECKOPINTERFACE
#define MLIR_TYPECHECKOPINTERFACE

include "mlir/IR/OpBase.td"

def TypeCheckOpInterface : OpInterface<"TypeCheckOpInterface"> {
  let cppNamespace = "::mlir";

  let description = [{
    This interface allows operations to participate in type inference.

    Operations implementing this interface advertise a typing rule through it,
    which is evaluated via the `typeCheck` method. This method can examine the
    current typing of all IR values through an `AbstractTypeChecker`, and then
    make deductions or indicate some `Contradiction`.

    Note that implementing this interface opts-in to type checking, meaning that
    all IR-declared result types are considered upper bounds instead of exact
    constraints for the purposes of type checking.

    In addition, this interface allows ops to customize transmutation of types
    of their owned values, if supported.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Evaluates the typing rule for this operation.

        The operation can observe the entire context through the `typeChecker`
        argument. It can then make deductions based on the context and report
        them to the `typeChecker`. In case typing is stuck, the method should
        return a contradiction that describes the issue.

        Implementations of this method must uphold the following invariants:
          - If the result indicates a contradiction, no deductions should have
            been reported to the `typeChecker`.

          - Deductions can only be made on owned values.

          - The typing rule should guarantee progress and preservation locally:
            - If a typing is correct, replacing some dependent value with a
              value of a subtype must lead to a correct typing. In particular,
              no supertype may be deduced for any owned value of this op.
              (Subsequent typing iterations can't become less restrictive.)

            - Evaluating the rule on a correct typing may not invalidate the
              operation unless some owned value was deduced to a proper subtype.
              (Iterations repeat until the typing is the most restrictive.).

            - Invalidating parent operations is necessary to type check the MLIR
              RegionBranchTerminator / yield idiom, but may lead to unbounded
              recursion. This is avoided by never invalidating child ops, and
              observing the first two rules.
              (Iterations must carry restricted yielded types to their parents.)

        Additionally, this op must return any contradiction that was produced by
        the type checker through methods invoked by the implementation.
      }],
      /*retTy=*/"::std::optional<::mlir::Typing::Contradiction>",
      /*methodName=*/"typeCheck",
      /*args=*/(ins "::mlir::Typing::AbstractTypeChecker &":$typeChecker),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return ::std::nullopt; }]>,
    InterfaceMethod<
      /*desc=*/[{
        Controls type transmutation of owned block arguments.

        Attempts to transmute the owned `argument` to type `into`. If `success`
        is returned, the type of the argument must now be `into`. If `failure`
        is returned, the IR must remain unchanged.

        The default implementation performs trivial transmutation of the
        argument if the following are all true:
          - There are no non-BranchOpInterface uses of the block.
          - All BranchOpInterface uses of the block remain legal after.
          - Any surrounding RegionBranchOpInterface remains legal after.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"transmuteArgument",
      /*args=*/(ins
        "::mlir::BlockArgument":$argument,
        "::mlir::Type":$into),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(argument && into);
        assert(argument.getOwner()->getParentOp() == $_op);
        return ::mlir::detail::transmuteArgument(argument, into);
      }]>,
    InterfaceMethod<
      /*desc=*/[{
        Controls type transmutation of owned result values.

        Attempts to transmute the owned `result` to type `into`. If `success`
        is returned, the type of the result must now be `into`. If `failure` is
        returned, the IR must remain unchanged.

        The default implementation performs trivial transmutation of the result.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"transmuteResult",
      /*args=*/(ins "::mlir::OpResult":$result, "::mlir::Type":$into),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(result && into);
        assert(result.getOwner() == $_op);
        result.setType(into);
        return success();
      }]>];

  // NOTE: Verify the op and its descendants first, then verify this interface.
  let verifyWithRegions = 1;
  let verify = [{ return ::mlir::detail::verifyTypeCheckOpInterface($_op); }];

  code extraClassDeclaration = [{
    ::mlir::LogicalResult transmute(::mlir::Value value, ::mlir::Type into)
    {
      assert(value && into);
      if (auto result = llvm::dyn_cast<OpResult>(value))
        return this->transmuteResult(result, into);
      else
        return this->transmuteArgument(llvm::cast<BlockArgument>(value), into);
    }
  }];
}

#endif // MLIR_TYPECHECKOPINTERFACE
