//===- Context.cpp - Type checking context --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Typing/Context.h"

#include "mlir/Typing/TypeCheckOpInterface.h"

using namespace mlir;
using namespace mlir::Typing;

//===----------------------------------------------------------------------===//
// Context implementation
//===----------------------------------------------------------------------===//

Bound Context::get(Value value) const {
	assert(value);

	// If the owner of value advertises a type checking rule, we assume the type
	// in the current IR to be the current upper bound. This allows deductions
	// to refine the type later.
	if (llvm::isa<TypeCheckOpInterface>(getOwner(value)))
		return Bound::upper(value.getType());

	// Otherwise, the value is owned by an operation that does not opt-in to the
	// type checking mechanism, and therefore is bounded exactly to its current
	// IR type.
	return value.getType();
}

const AbstractTypeSystem &Context::getTypeSystem(Dialect *owner) const {
  assert(owner);

  if (auto *iface = owner->getRegisteredInterface<DialectTypeSystemInterface>())
	  return *iface;

  return EqualityTypeSystem::getInstance();
}
