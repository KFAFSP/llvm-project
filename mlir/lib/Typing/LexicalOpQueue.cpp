//===- LexicalOpQueue.cpp - Lexically-ordered Op queue ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Typing/LexicalOpQueue.h"

#include "mlir/IR/Visitors.h"
#include <algorithm>
#include <iterator>

using namespace mlir;
using namespace mlir::Typing;

std::partial_ordering mlir::Typing::compareLexically(
    Operation *lhs,
    Operation *rhs) {
  assert(lhs && rhs);

  if (lhs == rhs) return std::partial_ordering::equivalent;

  // Find the least common ancestor of lhs and rhs, i.e., the deepest node in
  // the IR which is the ancestor of both. We ascend from after, setting it to
  // the first grand-sibling of the before node.
  auto *top = lhs;
  while (!top->isAncestor(rhs)) {
    lhs = top;
    top = top->getParentOp();
    if (!top) {
      // The two nodes share no common ancestor, which means they are not nested
      // within the same root IR operation, and thus incomparable.
      return std::partial_ordering::unordered;
    }
  }

  if (top == lhs) {
    // lhs is an ancestor of rhs, meaning it is lexically less than rhs.
    return std::partial_ordering::less;
  }
  if (top == rhs) {
    // rhs is an ancestor of lhs, meaning lhs is lexically greater than rhs.
    return std::partial_ordering::greater;
  }

  // Find the sibling of lhs that is the proper ancestor of rhs.
  while (rhs->getParentOp() != top) {
    rhs = rhs->getParentOp();
    assert(rhs);
  }

  // We're now at the "branch point" below top at which the paths to lhs and rhs
  // diverge, and we know that they are siblings on the op level. This means we
  // can perform a comparison at Region and Block level to order them.

  // Compare the regions.
  {
    auto *lhsRegion = lhs->getParentRegion();
    auto *rhsRegion = rhs->getParentRegion();
    assert(lhsRegion->getParentOp() == top && rhsRegion->getParentOp() == top);

    // Regions are canonically ordered in iteration order, which is given by the
    // region number.
    const auto cmp = lhsRegion->getRegionNumber() <=> rhsRegion->getRegionNumber();
    if (!std::is_eq(cmp)) return cmp;
  }

  // Compare the blocks.
  {
    auto *lhsBlock = lhs->getBlock();
    auto *rhsBlock = rhs->getBlock();

    if (lhsBlock != rhsBlock) {
      // Blocks are canonically ordered in iteration order, which we can only
      // observe by following the linked list. By following the list from one
      // side, and seeing if the other appears, we can distinguish both orders.
      lhsBlock = lhsBlock->getNextNode();
      while (lhsBlock && lhsBlock != rhsBlock)
        lhsBlock = lhsBlock->getNextNode();

      // Iff lhsBlock is not null, rhsBlock was found, and rhs is after lhs.
      return lhsBlock
        ? std::partial_ordering::less
        : std::partial_ordering::greater;
    }
  }

  // We can delegate the comparison to the monotonous operation indices.
  return lhs->isBeforeInBlock(rhs)
    ? std::partial_ordering::less
    : std::partial_ordering::greater;
}

[[nodiscard]]
static Operation *lexicalLast(Block &block) {
  return block.empty() ? nullptr : &block.back();
}

[[nodiscard]]
static Operation *lexicalLast(Region &region) {
  for (auto &block : llvm::reverse(region.getBlocks()))
    if (auto *last = lexicalLast(block)) return last;

  return nullptr;
}

Operation *mlir::Typing::lexicalLast(Operation *root) {
  assert(root);

  for (auto &region : llvm::reverse(root->getRegions()))
    if (auto *last = ::lexicalLast(region)) return last;

  return root;
}

//===----------------------------------------------------------------------===//
// LexicalOpQueue implementation
//===----------------------------------------------------------------------===//

bool LexicalOpQueue::compare(Operation *lhs, Operation *rhs) {
  const auto cmp = compareLexically(lhs, rhs);
  if (cmp == std::partial_ordering::unordered) {
    // Compare pointers so that there is some ordering. It does not need to be
    // consistent with Operation::walk, because that can not produce this case.
    return lhs > rhs;
  }

  return std::is_gt(cmp);
}

LexicalOpQueue::LexicalOpQueue(
    Operation *root,
    llvm::function_ref<bool(Operation *)> pred) {
  assert(root);

  root->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (!pred || pred(op)) impl.push_back(op);
  });
}

LexicalOpQueue::iterator LexicalOpQueue::find(Operation *op) const {
  auto *it = std::lower_bound(impl.begin(), impl.end(), op, compare);
  if (it == impl.end() || *it != op) return end();
  return std::make_reverse_iterator(++it);
}

bool LexicalOpQueue::insert(Operation *op) {
  // Perform sorted insertion into the worklist of invalid ops.
  auto *it = std::lower_bound(impl.begin(), impl.end(), op, compare);
  if (it != impl.end() && *it == op) return false;
  impl.insert(it, op);
  return true;
}

LexicalOpQueue::size_type LexicalOpQueue::insert(const LexicalOpQueue &rhs) {
  if (rhs.empty()) return 0U;

  // Find the range in this set that overlaps with the other.
  auto *first = std::lower_bound(impl.begin(), impl.end(), rhs.impl.front(), compare);
  auto *last = std::upper_bound(first, impl.end(), rhs.impl.back(), compare);
  if (first == last) {
    // The sets are trivially disjoint.
    impl.insert(last, rhs.impl.begin(), rhs.impl.end());
    return rhs.size();
  }

  // Move everything past the overlap point to a temporary buffer.
  storage_type lhs(first, impl.end());
  const auto oldSize = impl.size();
  impl.erase(first, impl.end());

  std::set_union(
    lhs.begin(),
    lhs.end(),
    rhs.impl.begin(),
    rhs.impl.end(),
    std::back_inserter(impl),
    compare);
  return size() - oldSize;
}
