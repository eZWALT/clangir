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
#include "CIRGenOpenMPRuntime.h"
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>

bool CIRClauseProcessor::processUntied(mlir::UnitAttr &result) const 
{
  markClauseOccurrence<clang::OMPUntiedClause>(result);
}

bool CIRClauseProcessor::processMergeable(mlir::UnitAttr &result) const
{
  markClauseOccurrence<clang::OMPMergeableClause>(result);
}

bool CIRClauseProcessor::processFinal(mlir::Value &result) const
{
  const clang::OMPFinalClause* clause = findUniqueClause<clang::OMPFinalClause>();
  if(clause){
    auto scopeLoc = this->CGF.getLoc(this->dirCtx.getSourceRange());
    const clang::Expr *finalExpr = clause->getCondition();
    mlir::Value finalValue = this->CGF.evaluateExprAsBool(finalExpr);
    mlir::ValueRange finalRange(finalValue);
    mlir::Type int1Ty = builder.getI1Type();
    result = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     scopeLoc, /*TypeOut*/ int1Ty, /*Inputs*/ finalRange)
                 .getResult(0);
    return true;
  }
  result = NULL;
  return false;
}

bool CIRClauseProcessor::processIf(mlir::Value &result) const 
{
  const clang::OMPIfClause* clause = findUniqueClause<clang::OMPIfClause>();
  if (clause) {
    auto scopeLoc = this->CGF.getLoc(this->dirCtx.getSourceRange());
    const clang::Expr *ifExpr = clause->getCondition();
    mlir::Value ifValue = this->CGF.evaluateExprAsBool(ifExpr);
    mlir::ValueRange ifRange(ifValue);
    mlir::Type int1Ty = builder.getI1Type();
    result = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     scopeLoc, /*TypeOut*/ int1Ty, /*Inputs*/ ifRange)
                 .getResult(0);
    return true;
  }
  result = NULL;
  return false;
}

bool CIRClauseProcessor::processPriority(mlir::Value &result) const {
  const clang::OMPPriorityClause* clause = findUniqueClause<clang::OMPPriorityClause>();
  if (clause) {
    auto scopeLoc = this->CGF.getLoc(dirCtx.getSourceRange());
    const clang::Expr *priorityExpr = clause->getPriority();
    mlir::Value priorityValue = this->CGF.buildScalarExpr(priorityExpr);
    mlir::ValueRange priorityRange(priorityValue);
    mlir::Type uint32Ty = builder.getI32Type();
    result = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     scopeLoc, /*TypeOut*/ uint32Ty, /*Inputs*/ priorityRange)
                 .getResult(0);
    return true;
  }
  result = NULL;
  return false;
}

bool CIRClauseProcessor::processDepend(
    mlir::omp::DependClauseOps& result, cir::OMPTaskDataTy) const
{
  return findRepeatableClause<clang::OMPDependClause>(
    [&](const clang::OMPDependClause* clause){
      mlir::omp::ClauseTaskDependAttr dependType = NULL;

      const mlir::Value variable = NULL;      
      result.dependVars.append(variable);
      result.dependTypeAttrs.append(dependType);
    }
  );

}