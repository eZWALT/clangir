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
#include "CIRGenBuilder.h"
#include "CIRGenOpenMPRuntime.h"
#include <clang/AST/ASTFwd.h>
#include <clang/Basic/OpenMPKinds.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>

bool CIRClauseProcessor::processUntied(mlir::omp::UntiedClauseOps &result) const 
{
  markClauseOccurrence<clang::OMPUntiedClause>(result.untiedAttr);
}

bool CIRClauseProcessor::processMergeable(mlir::omp::MergeableClauseOps &result) const
{
  markClauseOccurrence<clang::OMPMergeableClause>(result.mergeableAttr);
}

bool CIRClauseProcessor::processFinal(mlir::omp::FinalClauseOps& result) const
{
  const clang::OMPFinalClause* clause = findUniqueClause<clang::OMPFinalClause>();
  if(clause){
    auto scopeLoc = this->CGF.getLoc(this->dirCtx.getSourceRange());
    const clang::Expr *finalExpr = clause->getCondition();
    mlir::Value finalValue = this->CGF.evaluateExprAsBool(finalExpr);
    mlir::ValueRange finalRange(finalValue);
    mlir::Type int1Ty = builder.getI1Type();
    result.finalVar = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     scopeLoc, /*TypeOut*/ int1Ty, /*Inputs*/ finalRange)
                 .getResult(0);
    return true;
  }
  result = NULL;
  return false;
}

bool CIRClauseProcessor::processIf(mlir::omp::IfClauseOps& result) const 
{
  const clang::OMPIfClause* clause = findUniqueClause<clang::OMPIfClause>();
  if (clause) {
    auto scopeLoc = this->CGF.getLoc(this->dirCtx.getSourceRange());
    const clang::Expr *ifExpr = clause->getCondition();
    mlir::Value ifValue = this->CGF.evaluateExprAsBool(ifExpr);
    mlir::ValueRange ifRange(ifValue);
    mlir::Type int1Ty = builder.getI1Type();
    result.ifVar = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     scopeLoc, /*TypeOut*/ int1Ty, /*Inputs*/ ifRange)
                 .getResult(0);
    return true;
  }
  result = NULL;
  return false;
}

bool CIRClauseProcessor::processPriority(mlir::omp::PriorityClauseOps& result) const {
  const clang::OMPPriorityClause* clause = findUniqueClause<clang::OMPPriorityClause>();
  if (clause) {
    auto scopeLoc = this->CGF.getLoc(dirCtx.getSourceRange());
    const clang::Expr *priorityExpr = clause->getPriority();
    mlir::Value priorityValue = this->CGF.buildScalarExpr(priorityExpr);
    mlir::ValueRange priorityRange(priorityValue);
    mlir::Type uint32Ty = builder.getI32Type();
    result.priorityVar = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     scopeLoc, /*TypeOut*/ uint32Ty, /*Inputs*/ priorityRange)
                 .getResult(0);
    return true;
  }
  result = NULL;
  return false;
}

bool CIRClauseProcessor::processDepend(
    mlir::omp::DependClauseOps& result, cir::OMPTaskDataTy data) const
{
  return findRepeatableClause<clang::OMPDependClause>(
    [&](const clang::OMPDependClause* clause){
      //Get the depend type
      mlir::omp::ClauseTaskDependAttr dependType = getDependKindAttr(
        this->builder, clause
      );
      //Get an mlir value of the address of the depend variable
      const mlir::Value variable = NULL;
      result.dependVars.push_back(variable);
      result.dependTypeAttrs.push_back(dependType);
    }
  );

}


//Helper functions
static mlir::omp::ClauseTaskDependAttr getDependKindAttr(
  cir::CIRGenBuilderTy& builder,
  const clang::OMPDependClause* clause
){
  const clang::OpenMPDependClauseKind kind = clause->getDependencyKind();
  mlir::omp::ClauseTaskDepend mlirKind;
  switch(kind){
    case clang::OpenMPDependClauseKind::OMPC_DEPEND_in:
      mlirKind = mlir::omp::ClauseTaskDepend::taskdependin;
      break;
    case clang::OpenMPDependClauseKind::OMPC_DEPEND_out: 
      mlirKind = mlir::omp::ClauseTaskDepend::taskdependout;
      break;
    case clang::OpenMPDependClauseKind::OMPC_DEPEND_inout:
      mlirKind = mlir::omp::ClauseTaskDepend::taskdependinout;
      break;
    case clang::OpenMPDependClauseKind::OMPC_DEPEND_unknown:
    case clang::OpenMPDependClauseKind::OMPC_DEPEND_depobj:
    case clang::OpenMPDependClauseKind::OMPC_DEPEND_inoutallmemory:
    case clang::OpenMPDependClauseKind::OMPC_DEPEND_inoutset:
    case clang::OpenMPDependClauseKind::OMPC_DEPEND_mutexinoutset:
    case clang::OpenMPDependClauseKind::OMPC_DEPEND_outallmemory:
    case clang::OpenMPDependClauseKind::OMPC_DEPEND_sink:
    case clang::OpenMPDependClauseKind::OMPC_DEPEND_source:
      llvm_unreachable("Unhandled parser task dependency");
      break;
  }
  return mlir::omp::ClauseTaskDependAttr::get(builder.getContext(), mlirKind);
}