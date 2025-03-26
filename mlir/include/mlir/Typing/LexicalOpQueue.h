//===- TypeChecker.h - Lexically-ordered Op queue ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TYPING_LEXICALOPQUEUE_H
#define MLIR_TYPING_LEXICALOPQUEUE_H

#include "mlir/IR/Operation.h"

#include <compare>

namespace mlir::Typing {

/// Implements lexical three-way comparison of @p lhs and @p rhs .
///
/// The lexical ordering of ops is given by the canonical iteration order (i.e,
/// `Operation::walk<WalkOrder::PreOrder>`). This means that Regions and Blocks
/// are visited in their order at the parent. In particular, lexical ordering is
/// reflected in the generic op format.
///
/// @param              lhs First operation.
/// @param              rhs Second operation.
///
/// @pre    `lhs && rhs`
///
/// @retval `unordered`   @p lhs and @p rhs are not in the same IR subtree.
/// @retval `equivalent`  @p lhs and @p rhs point to the same op.
/// @retval `less`        @p lhs orders lexically before @p rhs .
/// @retval `greater`     @p lhs orders lexically after @p rhs .
[[nodiscard]]
std::partial_ordering compareLexically(Operation *lhs, Operation *rhs);

/// Finds the lexically last operation under @p root .
///
/// @param              root  Root operation.
///
/// @pre    `root`
///
/// @retval The last operation under @p root in lexical order.
///
/// @post   `std::is_gteq(result, root)`
[[nodiscard]]
Operation *lexicalLast(Operation *root);

/// Implements a priority queue of operations in lexical order.
///
/// Acts as an ordered set of operations such that each operation can only be
/// enqueued once, and operations are iterated / dequeued in lexical order as
/// established by compareLexically(Operation *, Operation *).
class LexicalOpQueue {
  [[nodiscard]] static bool compare(Operation *lhs, Operation *rhs);

public:
  using storage_type = SmallVector<Operation *, 16>;
  using value_type = Operation *;
  using size_type = storage_type::size_type;
  using iterator = storage_type::const_reverse_iterator;

  /// Initializes an empty LexicalOpQueue.
  /*implicit*/ LexicalOpQueue() = default;

  /// Initializes a LexicalOpQueue from @p root and all its descendants.
  ///
  /// @param              root  Root Operation.
  /// @param              pred  Optional predicate to match.
  ///
  /// @pre    `root`
  explicit LexicalOpQueue(
    Operation *root,
    function_ref<bool(Operation *)> pred = {});

  /// Finds @p op is in the queue.
  ///
  /// @param              op  Operation to find.
  ///
  /// @pre    `op`
  [[nodiscard]] iterator find(Operation *op) const;
  /// Determines whether @p op is in the queue.
  ///
  /// @param              op  Operation to find.
  ///
  /// @pre    `op`
  [[nodiscard]] bool contains(Operation *op) const {
    return find(op) != end();
  }

  /// Inserts @p op into the queue.
  ///
  /// @param              op  Operation to insert.
  ///
  /// @pre    `op`
  ///
  /// @return Whether @p op was inserted.
  bool insert(Operation *op);

  /// Inserts @p rhs into the queue.
  ///
  /// @param  [in]      rhs LexicalOpQueue to insert.
  ///
  /// @return Number of ops inserted.
  size_type insert(const LexicalOpQueue &rhs);

  /// Dequeues the next operation in lexical order.
  ///
  /// @retval Operation Next operation in lexical order.
  /// @retval `nullptr` Queue is empty.
  Operation *dequeue() {
    if (empty()) return nullptr;
    return impl.pop_back_val();
  }

  [[nodiscard]] bool empty() const { return impl.empty(); }
  [[nodiscard]] size_type size() const { return impl.size(); }
  [[nodiscard]] iterator begin() const { return impl.rbegin(); }
  [[nodiscard]] iterator end() const { return impl.rend(); }

private:
  storage_type impl;
};

} // namespace mlir::Typing

#endif // MLIR_TYPING_LEXICALOPQUEUE_H
