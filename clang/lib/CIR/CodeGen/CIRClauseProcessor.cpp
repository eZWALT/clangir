//===--- CIRGenOpenMPRuntime.h - Interface to OpenMP Runtimes -------------===//
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

#include "CIRClauseProcessor.h"
#include "CIRGenModule.h"

bool CIRClauseProcessor::processUntied(
    const clang::OMPExecutableDirective &dirCtx, mlir::UnitAttr &result) const {
  bool hasUntied = dirCtx.hasClausesOfKind<OMPUntiedClause>();
  result = hasUntied ? this->CGF.getBuilder().getUnitAttr() : nullptr;
  return hasUntied;
}

bool CIRClauseProcessor::processMergeable(
    const clang::OMPExecutableDirective &dirCtx, mlir::UnitAttr &result) const {
  bool hasMergeable = dirCtx.hasClausesOfKind<OMPMergeableClause>();
  result = hasMergeable ? this->CGF.getBuilder().getUnitAttr() : nullptr;
  return hasMergeable;
}

bool CIRClauseProcessor::processFinal(
    const clang::OMPExecutableDirective &dirCtx, mlir::Value &result) const {
  llvm_unreachable("Clause currently in development");
  /* Yet to be implemented, hint:
  bool hasFinal = dirCtx.hasClausesOfKind<OMPFinalClause>();
  if(hasFinal){
    //getSingleClause will raise an exception if multiple identical clauses
  exist const clang::OMPFinalClause* finalClause =
  dirCtx.getSingleClause<OMPFinalClause>(); const clang::Expr* finalExpr =
  finalClause->getCondition();
    //This is the issue, returns a value of type cir.bool and not mlir unsigned
  int 1 bit result = this->CGF.evaluateExprAsBool(finalExpr);
  }
  return hasFinal;
  */
}

bool CIRClauseProcessor::processIf(const clang::OMPExecutableDirective &dirCtx,
                                   mlir::Value &result) const {
  llvm_unreachable("Clause currently in development");
  /* Yet to be implemented, hint:
  bool hasIf = dirCtx.hasClausesOfKind<OMPIfClause>();
  if(hasIf){
    //getSingleClause will raise an exception if multiple identical clauses
  exist const clang::OMPIfClause* ifClause =
  dirCtx.getSingleClause<OMPIfClause>(); const clang::Expr* ifExpr =
  ifClause->getCondition();
    //This is the issue, returns a value of type cir.bool and not mlir unsigned
  int 1 bit result = this->CGF.evaluateExprAsBool(ifExpr);
  }
  return hasIf;
  */
}

bool CIRClauseProcessor::processPriority(
    const clang::OMPExecutableDirective &dirCtx, mlir::Value &result) const {
  llvm_unreachable("Clause currently in development");
  /* Yet to be implemented, hint:
  bool hasPriority = dirCtx.hasClausesOfKind<OMPPriorityClause>();
  if(hasPriority){
    //getSingleClause will raise an exception if multiple identical clauses
  exist const clang::OMPPriorityClause* priorityClause =
  dirCtx.getSingleClause<OMPPriorityClause>(); const clang::Expr* priorityExpr =
  priorityClause->getPriority();
    //This is the issue, returns a scalar of type cir.int and not mlir unsigned
  int 32 result = this->CGF.buildScalarExpr(priorityExpr);
  }
  return hasPriority;
  */
}
