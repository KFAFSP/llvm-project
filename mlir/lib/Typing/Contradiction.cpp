//===- Contradiction.cpp - Contradiction result helper ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Typing/Contradiction.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Typing/TypeChecker.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>

using namespace mlir;
using namespace mlir::Typing;

//===----------------------------------------------------------------------===//
// Source implementation
//===----------------------------------------------------------------------===//

std::optional<Location> Source::getLoc() const {
  if (getPointer().isNull()) {
    // Saves us having to write 'dyn_cast_if_present'.
    return std::nullopt;
  }

  if (auto value = llvm::dyn_cast<Value>(getPointer()))
    return value.getLoc();
  if (auto *use = llvm::dyn_cast<OpOperand *>(getPointer()))
    return use->getOwner()->getLoc();
  if (auto *op = llvm::dyn_cast<Operation *>(getPointer()))
    return op->getLoc();
  if (llvm::isa<Location>(getPointer()))
    return llvm::cast<Location>(getPointer());

  // Everything else does not have an associated location.
  return std::nullopt;
}

void Source::print(llvm::raw_ostream &os) const {
  if (getPointer().isNull()) {
    os << "<unknown>";
    return;
  }

  OpPrintingFlags flags{};
  flags.skipRegions();
  flags.elideLargeElementsAttrs();
  flags.elideLargeResourceString();

  if (llvm::isa<Context *>(getPointer())) {
    os << "from context";
    return ;
  }
  if (auto *dialect = llvm::dyn_cast<Dialect *>(getPointer())) {
    os << "dialect '" << dialect->getNamespace() << "'";
    return;
  }
  if (auto value = llvm::dyn_cast<Value>(getPointer())) {
    if (auto arg = llvm::dyn_cast<BlockArgument>(value)) {
      os << "argument #" << arg.getArgNumber() << " of ";
      if (arg.getOwner()->isEntryBlock()) {
        os << "region #" << arg.getParentRegion()->getRegionNumber() << " of ";
        arg.getOwner()->getParentOp()->print(os, flags);
        return;
      }

      arg.getOwner()->printAsOperand(os);
      return;
    }

    auto result = llvm::cast<OpResult>(value);
    os << "result #" << result.getResultNumber() << " of ";
    result.getOwner()->print(os, flags);
    return;
  }
  if (auto *use = llvm::dyn_cast<OpOperand *>(getPointer())) {
    os << "operand #" << use->getOperandNumber() << " of ";
    use->getOwner()->print(os, flags);
    return;
  }
  if (auto *op = llvm::dyn_cast<Operation *>(getPointer())) {
    op->print(os, flags);
    return;
  }
  if (llvm::isa<Location>(getPointer())) {
    os << llvm::cast<Location>(getPointer());
    return;
  }

  llvm_unreachable("unhandled PtrUnion alternative");
}

//===----------------------------------------------------------------------===//
// Evidence implementation
//===----------------------------------------------------------------------===//

InFlightDiagnostic Evidence::toDiagnostic(
    Location defaultLoc,
    DiagnosticSeverity severity) const {
  defaultLoc = getLoc().value_or(defaultLoc);
  const auto attachArgs = [&](Diagnostic &diag) {
    for (auto &arg : getArgs()) diag.append(arg);
  };

  switch (severity) {
    case DiagnosticSeverity::Error:
    {
      auto result = emitError(defaultLoc);
      attachArgs(*result.getUnderlyingDiagnostic());
      return result;
    }

    case DiagnosticSeverity::Warning:
    {
      auto result = emitWarning(defaultLoc);
      attachArgs(*result.getUnderlyingDiagnostic());
      return result;
    }

    case DiagnosticSeverity::Remark:
    {
      auto result = emitRemark(defaultLoc);
      attachArgs(*result.getUnderlyingDiagnostic());
      return result;
    }

    default:
      llvm_unreachable("can't raise note");
  }
}

Diagnostic &Evidence::toNote(Diagnostic &diag) const {
  auto &note = diag.attachNote(getLoc());
    for (auto &arg : getArgs()) note.append(arg);
  return note;
}

Evidence &Evidence::append(const SmallVectorImpl<char> &str) {
  auto &stored = strings.emplace_back(std::make_unique<char[]>(str.size()));
  std::copy(str.begin(), str.end(), stored.get());
  args.emplace_back(StringRef(stored.get(), str.size()));
  return *this;
}

Evidence &Evidence::append(const Twine &str) {
  if (str.isSingleStringRef()) {
    // This code assumes that StringRef arguments will have sufficient lifetime.
    args.emplace_back(str.getSingleStringRef());
    return *this;
  }

  SmallString<64> buffer;
  str.toVector(buffer);
  return append(buffer);
}

Evidence &Evidence::append(
    llvm::function_ref<void(llvm::raw_ostream &)> printer) {
  llvm::SmallString<64> buffer;
  {
    llvm::raw_svector_ostream os(buffer);
    printer(os);
  }
  return append(buffer);
}

void Evidence::print(llvm::raw_ostream &os) const {
  os << "from ";
  getSource().print(os);
  os << ": ";

  for (auto &arg : getArgs()) arg.print(os);
}

void Evidence::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

//===----------------------------------------------------------------------===//
// Contradiction implementation
//===----------------------------------------------------------------------===//

Contradiction &Contradiction::becomesEffect(Source at) {
  // Move the data stored in this Contradiction into a new instance, which then
  // re-assigns the cause pointer of this instance.
  this->cause = std::make_unique<Contradiction>(std::move(*this));
  // Assign new Evidence to the internal storage.
  static_cast<Evidence &>(*this) = Evidence(at);
  return *this;
}

LogicalResult Contradiction::handle(Location loc) {
  if (!isFatal()) return success();
  return parent->handle(loc, std::move(*this));
}

void Contradiction::print(llvm::raw_ostream &os) const {
  for (auto *contra = this; contra; contra = contra->getCause()) {
    static_cast<const Evidence &>(*contra).print(os);
    for (auto &note: contra->getNotes()) {
      os << "  note: ";
      note.print(os);
    }
  }
}

void Contradiction::dump() const {
  print(llvm::errs());
}

void mlir::Typing::emit(
    Location loc,
    const Contradiction &contra,
    DiagnosticSeverity severity) {
  // Follow the chain of causes.
  for (auto *it = &contra; it; it = it->getCause()) {
    if (!it->isFatal() && severity == DiagnosticSeverity::Error) {
      // All causes leading up to this point are downgraded to warnings.
      severity = DiagnosticSeverity::Warning;
    }

    auto diag = it->toDiagnostic(loc, severity);
    for (auto &note : it->getNotes())
      note.toNote(*diag.getUnderlyingDiagnostic());
    loc = diag.getUnderlyingDiagnostic()->getLocation();
    diag.report();
  }
}
