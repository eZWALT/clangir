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
#include <mlir/IR/BuiltinAttributes.h>

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

  bool hasFinal = dirCtx.hasClausesOfKind<OMPFinalClause>();
  if (hasFinal) {
    auto builder = this->CGF.getBuilder();
    auto scopeLoc = this->CGF.getLoc(dirCtx.getSourceRange());
    // getSingleClause will raise an exception if multiple identical clauses
    // exist
    const clang::OMPFinalClause *finalClause =
        dirCtx.getSingleClause<OMPFinalClause>();
    const clang::Expr *finalExpr = finalClause->getCondition();
    mlir::Value finalValue = this->CGF.evaluateExprAsBool(finalExpr);
    mlir::ValueRange finalRange(finalValue);

    mlir::Type int1Ty = builder.getI1Type();
    result = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     scopeLoc, /*TypeOut*/ int1Ty, /*Inputs*/ finalRange)
                 .getResult(0);
  }
  return hasFinal;
}

bool CIRClauseProcessor::processIf(const clang::OMPExecutableDirective &dirCtx,
                                   mlir::Value &result) const {
  bool hasIf = dirCtx.hasClausesOfKind<OMPIfClause>();
  if (hasIf) {
    auto builder = this->CGF.getBuilder();
    auto scopeLoc = this->CGF.getLoc(dirCtx.getSourceRange());
    // getSingleClause will raise an exception if multiple identical clauses
    // exist
    const clang::OMPIfClause *ifClause = dirCtx.getSingleClause<OMPIfClause>();
    const clang::Expr *ifExpr = ifClause->getCondition();
    mlir::Value ifValue = this->CGF.evaluateExprAsBool(ifExpr);
    mlir::ValueRange ifRange(ifValue);

    mlir::Type int1Ty = builder.getI1Type();
    result = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     scopeLoc, /*TypeOut*/ int1Ty, /*Inputs*/ ifRange)
                 .getResult(0);
  }
  return hasIf;
}

bool CIRClauseProcessor::processPriority(
    const clang::OMPExecutableDirective &dirCtx, mlir::Value &result) const {
  bool hasPriority = dirCtx.hasClausesOfKind<OMPPriorityClause>();
  if (hasPriority) {
    auto builder = this->CGF.getBuilder();
    auto scopeLoc = this->CGF.getLoc(dirCtx.getSourceRange());
    // getSingleClause will raise an exception if multiple identical clauses
    // exist
    const clang::OMPPriorityClause *priorityClause =
        dirCtx.getSingleClause<OMPPriorityClause>();
    const clang::Expr *priorityExpr = priorityClause->getPriority();
    mlir::Value priorityValue = this->CGF.buildScalarExpr(priorityExpr);
    mlir::ValueRange priorityRange(priorityValue);

    mlir::Type uint32Ty = builder.getI32Type();
    result = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     scopeLoc, /*TypeOut*/ uint32Ty, /*Inputs*/ priorityRange)
                 .getResult(0);
  }
  return hasPriority;
}

bool CIRClauseProcessor::processDepend(
    const clang::OMPExecutableDirective &dirCtx,
    mlir::ArrayAttr &dependTypeOperands,
    llvm::SmallVector<mlir::Value> &dependOperands) const {
  bool hasDepend = dirCtx.hasClausesOfKind<OMPDependClause>();
  llvm_unreachable("Clause currently in development");

  if (hasDepend) {
    auto builder = this->CGF.getBuilder();
    auto scopeLoc = this->CGF.getLoc(dirCtx.getSourceRange());

    // XD?
  }
  return hasDepend;
}