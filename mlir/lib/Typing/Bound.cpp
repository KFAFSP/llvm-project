//===- Bound.cpp - Subtype type bound -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Typing/Bound.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeSystem.h"

#include <cassert>
#include <compare>

using namespace mlir;
using namespace mlir::Typing;

//===----------------------------------------------------------------------===//
// Bound implementation
//===----------------------------------------------------------------------===//

/// Performs the actual bounds test.
///
/// @param              kind  Bound::Kind.
/// @param              cmp   Comparison of type <=> Bound::getValue()
///
/// @return Whether the compared type is within the bound.
[[nodiscard]] static bool test(Bound::Kind kind, std::partial_ordering cmp) {
  constexpr auto lowerMask = static_cast<std::intptr_t>(Bound::Kind::Lower);
  constexpr auto upperMask = static_cast<std::intptr_t>(Bound::Kind::Upper);
  const auto flags = static_cast<std::intptr_t>(kind);

  if ((flags & lowerMask) == lowerMask && !std::is_gteq(cmp)) return false;
  if ((flags & upperMask) == upperMask && !std::is_lteq(cmp)) return false;
  return true;
}

bool Bound::test(const AbstractTypeSystem &typeSystem, Type type) const {
  // Perform the trivial comparisons.
  if (getValue() == type) return true;
  if (!type || !getValue()) return false;

  return ::test(getKind(), typeSystem.compare(type, getValue()));
}

Bound Bound::meet(const AbstractTypeSystem &typeSystem, Bound lhs, Bound rhs) {
  using std::swap;

  // Handle absent bound value cases.
  if (!lhs.getValue() || !rhs.getValue()) {
    return {};
  }

  // Handle identity bounds without invoking the type system.
  if (lhs.isIdentity() || rhs.isIdentity()) {
    if (lhs.getValue() != rhs.getValue()) return {};
    return lhs.getValue();
  }

  // Compare the two bound values.
  auto relation = typeSystem.compare(lhs, rhs);

  // Ensure that lhs is the more restrictive bound of the two, and use that.
  // The order is: (unattainable, identity,) equivalence, lower, upper

  if (rhs.isEquivalence()) { swap(lhs, rhs); relation = 0 <=> relation; }
  if (lhs.isEquivalence()) {
    // The result is at least an equivalence constraint.

    if (!::test(rhs.getKind(), relation)) {
      // The value of lhs is not matched by rhs.
      return {};
    }
    if (std::is_eq(relation)) {
      // Both types are equivalent, so we must disambiguate.
      return AbstractTypeSystem::disambiguate(lhs, rhs);
    }

    // Lhs is still the strictest bound.
    return lhs;
  }

  if (rhs.isLower()) { swap(lhs, rhs); relation = 0 <=> relation; }
  if (lhs.isLower()) {
    // The result is at least a lower bound.

    if (rhs.isUpper()) {
      if (std::is_gt(relation)) {
        // The upper bound is less than the lower bound, so no intersection.
        return {};
      }
      if (std::is_eq(relation)) {
        // The values are equivalent, so we derive an equivalence constraint
        // from the disambiguated types.
        return AbstractTypeSystem::disambiguate(lhs, rhs);
      }
    } else if (rhs.isLower()) {
      if (std::is_lt(relation)) {
        // The rhs bound is greater, so it is a better lower bound.
        return rhs;
      }
      if (std::is_eq(relation)) {
        // The values are equivalent, so we must disambiguate.
        return Bound::lower(AbstractTypeSystem::disambiguate(lhs, rhs));
      }
    }

    // Lhs is still the strictest bound.
    return lhs;
  }

  // The result is at least an upper bound.
  assert(lhs.isUpper() && rhs.isUpper());

  if (std::is_gt(relation)) {
    // The rhs bound is smaller, so it is a better upper bound.
    return rhs;
  }
  if (std::is_eq(relation)) {
    // The values are equivalent, so we must disambiguate.
    return Bound::upper(AbstractTypeSystem::disambiguate(lhs, rhs));
  }

  // Lhs is still the strictest bound.
  return lhs;
}

void Bound::print(llvm::raw_ostream &os) const {
  os << "{";
  if (auto value = getValue()) {
    switch (getKind()) {
      case Kind::Equivalence: os << "~= "; break;
      case Kind::Upper:       os << "<: "; break;
      case Kind::Lower:       os << ":> "; break;
      default: break;
    }
    value.print(os);
  }
  os << "}";
}
