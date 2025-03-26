//===- TypeSystem.cpp - MLIR Type System ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeSystem.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <compare>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// AbstractTypeSystem implementation
//===----------------------------------------------------------------------===//

AbstractTypeSystem::~AbstractTypeSystem() = default;

std::partial_ordering
AbstractTypeSystem::compare(Type lhs, Type rhs) const {
  assert(lhs && rhs);

  // Short-circuit trivial equality.
  if (lhs == rhs) return std::partial_ordering::equivalent;

  // Compare in both directions and return most restrictive result.
  const auto lessOrEqual    = isSubtype(lhs, rhs);
  const auto greaterOrEqual = isSubtype(rhs, lhs);
  if (lessOrEqual && greaterOrEqual) return std::partial_ordering::equivalent;
  if (lessOrEqual) return std::partial_ordering::less;
  if (greaterOrEqual) return std::partial_ordering::greater;
  return std::partial_ordering::unordered;
}

Type AbstractTypeSystem::min(Type lhs, Type rhs) const {
  assert(lhs && rhs);

  const auto relation = compare(lhs, rhs);
  if (std::is_eq(relation)) {
    // The types are equivalent, but not necessarily definitionally equal. We
    // must ensure that an order-independent solution is found.
    return disambiguate(lhs, rhs);
  }

  if (std::is_lt(relation)) return lhs;
  if (std::is_gt(relation)) return rhs;

  // The types are unrelated.
  return {};
}

Type AbstractTypeSystem::max(Type lhs, Type rhs) const {
  assert(lhs && rhs);

  const auto relation = compare(lhs, rhs);
  if (std::is_eq(relation)) {
    // The types are equivalent, but not necessarily definitionally equal. We
    // must ensure that an order-independent solution is found.
    return disambiguate(lhs, rhs);
  }

  if (std::is_gt(relation)) return lhs;
  if (std::is_lt(relation)) return rhs;

  // The types are unrelated.
  return {};
}

/// Concept for an invocable that reduces Types.
template<class Fn>
concept TypeReduction = std::is_invocable_r_v<Type, Fn, Type, Type>;

/// Tries to reduce @p types to a single value via fallible reduction @p fn .
///
/// @param  [in,out]    types   The types to reduce.
/// @param              fn      The fallible reduction function.
///
/// @pre    `llvm::count(types, Type{}) == 0`
///
/// @post   `llvm::count(types, Type{}) == 0`
/// @post   `types_post.size() <= types_pre.size()`
static void reducePairwise(
    llvm::SmallVectorImpl<Type> &types,
    TypeReduction auto fn) {
  assert(llvm::count(types, Type{}) == 0);

  using std::swap;

  if (types.empty()) {
    // The empty list can't be reduced.
    return;
  }

  // Reduce pairs in the list until it is irreducible.
  auto *head = types.begin();
  while (types.size() > 1) {
    // Find some element to combine with the head.
    auto *it = std::next(head);
    for (; it != types.end(); ++it) {
      if (const auto reduced = fn(*head, *it)) {
        // Update the head element and stop searching.
        *head = reduced;
        break;
      }
    }

    if (it == types.end()) {
      // No pair involving the current head could be reduced. Advance the
      // head to check all other pairs.
      if (++head == types.end()) {
        // All pairs have been visited, the list is irreducible.
        return;
      }

      // Continue searching from the new head.
      continue;
    }

    // The element at `it` was reduced into the head, erase it.
    types.erase(it);
    if (head != types.begin()) {
      // Swap the head to the front of the list, to avoid searching over
      // all the known-irreducible pairs first.
      swap(types.front(), *head);
      // Set the head to the front of the list to ensure all pairs will be
      // visited in future iterations.
      head = types.begin();
    }
  }
}

void AbstractTypeSystem::promote(SmallVectorImpl<Type> &types) const {
#ifndef NDEBUG
  assert(llvm::count(types, Type{}) == 0);
  SmallVector<Type> typesPre{ArrayRef<Type>(types)};
#endif

  // NOTE: The top type could be short-circuited to the result if it is known,
  //       but there is no significant benefit at the expected sizes.

  // Reduce as many pairs using the promote function as possible.
  reducePairwise(types, [&](Type lhs, Type rhs) -> Type {
    return promote(lhs, rhs);
  });

#ifndef NDEBUG
  // NOTE: While the algorithm is trusted, the interface implementation isn't.
  //       Asserting the invariants here protects downstream users.

  // The promoted type must be a supertype of all input types.
  assert(types.size() != 1 || llvm::all_of(typesPre, [&](Type type) -> bool {
         return isSubtype(type, types.front());
       }));
  // types must be irreducible (at least) according to the subtype relation.
  for (auto *lhs = types.begin(); lhs != types.end(); ++lhs)
    for (auto *rhs = std::next(lhs); rhs != types.end(); ++rhs)
      assert(compare(*lhs, *rhs) == std::partial_ordering::unordered);
#endif
}

FailureOr<Type> AbstractTypeSystem::promote(ArrayRef<Type> types) const {
  SmallVector<Type> worklist(types);
  promote(worklist);
  if (worklist.size() == 1) return worklist.front();
  return failure();
}

//===----------------------------------------------------------------------===//
// EqualityTypeSystem implementation
//===----------------------------------------------------------------------===//

EqualityTypeSystem &EqualityTypeSystem::getInstance() {
  static EqualityTypeSystem instance;
  return instance;
}

//===----------------------------------------------------------------------===//
// CachedTypeSystem implementation
//===----------------------------------------------------------------------===//

struct CachedTypeSystem::Impl {
  llvm::DenseMap<std::pair<Type, Type>, bool> subtypeRelation;
  llvm::DenseMap<std::pair<Type, Type>, Type> promotionOperator;
};

CachedTypeSystem::CachedTypeSystem(const AbstractTypeSystem &base)
    : base(base), impl(std::make_unique<Impl>()) {}

CachedTypeSystem::~CachedTypeSystem() = default;

bool CachedTypeSystem::isSubtype(Type sub, Type super) const {
  auto [it, added] = impl->subtypeRelation
    .try_emplace(std::make_pair(sub, super));
  if (added) it->getSecond() = base.isSubtype(sub, super);
  return it->getSecond();
}

Type CachedTypeSystem::promote(Type lhs, Type rhs) const {
  // Derive an order-independent key for (lhs, rhs).
  const auto makeKey = [](Type lhs, Type rhs) {
    if (lhs.getAsOpaquePointer() > rhs.getAsOpaquePointer())
      return std::make_pair(rhs, lhs);
    return std::make_pair(lhs, rhs);
  };

  auto [it, added] = impl->promotionOperator
    .try_emplace(makeKey(lhs, rhs));
  if (added) it->getSecond() = base.promote(lhs, rhs);
  return it->getSecond();
}

//===----------------------------------------------------------------------===//
// DialectTypeSystemInterface implementation
//===----------------------------------------------------------------------===//

DialectTypeSystemInterface::~DialectTypeSystemInterface() = default;

Operation *DialectTypeSystemInterface::promote(
    OpBuilder &builder,
    Location loc,
    Value input,
    Type superType) const {
  assert(input && superType);
  assert(isSubtype(input.getType(), superType));

  return builder.create<UnrealizedConversionCastOp>(
    loc,
    TypeRange{superType},
    ValueRange{input});
}

//===----------------------------------------------------------------------===//
// MLIRTypeSystem implementation
//===----------------------------------------------------------------------===//

bool MLIRTypeSystem::isSubtype(Dialect *dialect, Type sub, Type super) const {
  assert(sub && super);

  if (sub == super) {
    // Ensure that reflexivity is always preserved.
    return true;
  }

  const auto *iface = this->getInterfaceFor(dialect);
  return iface && iface->isSubtype(sub, super);
}

std::partial_ordering
MLIRTypeSystem::compare(Dialect *dialect, Type lhs, Type rhs) const {
  assert(lhs && rhs);

  if (lhs == rhs) {
    // Ensure that trivial equivalence is always preserved.
    return std::partial_ordering::equivalent;
  }

  const auto *iface = this->getInterfaceFor(dialect);
  return iface ? iface->compare(lhs, rhs) : std::partial_ordering::unordered;
}

bool MLIRTypeSystem::canSubstitute(const OpOperand &use, Type with) const {
  assert(with);

  // Substitutability is decided by the subtype relation under the dialect
  // of the operation that has the operand use.
  return isSubtype(use.getOwner()->getDialect(), with, use.get().getType());
}

bool MLIRTypeSystem::canSubstitute(Value value, Type with) const {
  assert(value && with);

  // All uses of the value must be substitutable under their respective users'
  // dialects, as implemented above.
  return llvm::all_of(
    value.getUses(),
    [=,this](const OpOperand &use) {
      return canSubstitute(use, with);
    });
}
