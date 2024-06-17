//===--- CIRClauseProcessor - Interface to OpenMP Runtimes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class provides helper methods for generating MLIR code for OpenMP
// clauses within the MLIR::OMP dialect.
//
// Each method named process<Clause>() handles the MLIR code generation
// for a specific OpenMP clause type. These functions return `false` if the
// corresponding clause is not present. Otherwise, they return `true` and update
// the referenced parameters with the generated MLIR code.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRCLAUSEPROCESSOR_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRCLAUSEPROCESSOR_H
#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenOpenMPRuntime.h"
#include "clang/AST/ASTFwd.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/Basic/OpenMPKinds.h"

// #include <functional>
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Support/ErrorHandling.h"
#include <list>
#include <string>
#include <variant>

#include "mlir/Dialect/OpenMP/OpenMPClauseOperands.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

class CIRClauseProcessor {

private:
  cir::CIRGenFunction &CGF;
  cir::CIRGenBuilderTy &builder;
  const clang::OMPExecutableDirective &dirCtx;
  std::list<const clang::OMPClause *> clauses;
  using ClauseIterator = std::list<const clang::OMPClause *>::const_iterator;

  static llvm::StringRef getClauseName(const clang::OMPClause *clause);
  static mlir::omp::ClauseTaskDependAttr
  getDependKindAttr(cir::CIRGenBuilderTy &builder,
                    const clang::OMPDependClause *clause);
  // Get the iterator of a clause of type C
  template <typename C>
  static ClauseIterator findClause(ClauseIterator begin, ClauseIterator end);

  // Get the first instance of a clause of type C, nullptr otherwise
  template <typename C> const C *findUniqueClause() const;

  // Treat each clause according to function "callback", returns true if one or
  // more clauses of type C are found
  template <typename C>
  bool findRepeatableClause(std::function<void(const C *)> callback) const;

  // Sets "result" attribute if the clause of type C is present
  template <typename C> bool markClauseOccurrence(mlir::UnitAttr &result) const;

public:
  CIRClauseProcessor(cir::CIRGenFunction &CGF,
                     const clang::OMPExecutableDirective &dirCtx)
      : CGF(CGF), builder(this->CGF.getBuilder()), dirCtx(dirCtx),
        clauses(std::list<const clang::OMPClause *>()) {
    for (const clang::OMPClause *clause : dirCtx.clauses()) {
      this->clauses.push_back(clause);
    }
  }

  // At most 1 ocurrence clauses
  bool processIf(mlir::omp::IfClauseOps &result) const;

  bool processFinal(mlir::omp::FinalClauseOps &result) const;

  bool processPriority(mlir::omp::PriorityClauseOps &result) const;

  // UnitAttr clauses

  bool processUntied(mlir::omp::UntiedClauseOps &result) const;

  bool processMergeable(mlir::omp::MergeableClauseOps &result) const;

  bool processNowait(mlir::omp::NowaitClauseOps &result) const;

  // Taskloop clauses
  bool processGrainSize(mlir::omp::GrainsizeClauseOps &result) const;

  bool processNumTasks(mlir::omp::NumTasksClauseOps &result) const;

  bool processNogroup(mlir::omp::NogroupClauseOps &result) const;

  // Repeatable clauses
  bool processDepend(mlir::omp::DependClauseOps &result,
                     cir::OMPTaskDataTy &data, mlir::Location &location) const;

  template <typename... Cs> void processTODO() const;
};

template <typename C>
CIRClauseProcessor::ClauseIterator
CIRClauseProcessor::findClause(ClauseIterator begin, ClauseIterator end) {
  for (ClauseIterator it = begin; it != end; ++it) {
    const clang::OMPClause *clause = *it;
    if (llvm::dyn_cast<const C>(clause)) {
      return it;
    }
  }
  return end;
}

template <typename C> const C *CIRClauseProcessor::findUniqueClause() const {
  ClauseIterator it = findClause<C>(clauses.begin(), clauses.end());
  if (it != clauses.end()) {
    return llvm::dyn_cast<const C>(*it);
  }
  return nullptr;
}

template <typename C>
bool CIRClauseProcessor::markClauseOccurrence(mlir::UnitAttr &result) const {
  if (findUniqueClause<C>()) {
    result = this->CGF.getBuilder().getUnitAttr();
    return true;
  }
  return false;
}

template <typename C>
bool CIRClauseProcessor::findRepeatableClause(
    std::function<void(const C *)> callback) const {
  bool found = false;
  ClauseIterator next, end = clauses.end();
  for (ClauseIterator it = clauses.begin(); it != end; it = next) {
    next = findClause<C>(it, end);

    if (next != end) {
      callback(llvm::dyn_cast<const C>(*next));
      found = true;
      ++next;
    }
  }
  return found;
}

template <typename... Cs> void CIRClauseProcessor::processTODO() const {
  auto checkClause = [&](const clang::OMPClause *clause) {
    if (clause) {
      std::string error_msg = "The following clause is not yet implemented: " +
                              getClauseName(clause).str();
      llvm_unreachable(error_msg.c_str());
    }
  };

  for (const clang::OMPClause *clause : clauses) {
    (checkClause(llvm::dyn_cast<const Cs>(clause)), ...);
  }
}
#endif