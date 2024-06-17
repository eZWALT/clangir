//===--- CIRClauseProcessor- Interface to OpenMP Runtimes -------------===//
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
#include "Address.h"
#include "CIRGenBuilder.h"
#include "CIRGenOpenMPRuntime.h"
#include <clang/AST/ASTFwd.h>

#include <clang/Basic/OpenMPKinds.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>

bool CIRClauseProcessor::processNowait(
    mlir::omp::NowaitClauseOps &result) const {
  return markClauseOccurrence<clang::OMPNowaitClause>(result.nowaitAttr);
}

bool CIRClauseProcessor::processUntied(
    mlir::omp::UntiedClauseOps &result) const {
  return markClauseOccurrence<clang::OMPUntiedClause>(result.untiedAttr);
}
bool CIRClauseProcessor::processMergeable(
    mlir::omp::MergeableClauseOps &result) const {
  return markClauseOccurrence<clang::OMPMergeableClause>(result.mergeableAttr);
}
bool CIRClauseProcessor::processFinal(mlir::omp::FinalClauseOps &result) const {
  const clang::OMPFinalClause *clause =
      findUniqueClause<clang::OMPFinalClause>();
  if (clause) {
    auto scopeLoc = this->CGF.getLoc(this->dirCtx.getSourceRange());
    const clang::Expr *finalExpr = clause->getCondition();
    mlir::Value finalValue = this->CGF.evaluateExprAsBool(finalExpr);
    mlir::ValueRange finalRange(finalValue);
    mlir::Type int1Ty = builder.getI1Type();
    result.finalVar =
        builder
            .create<mlir::UnrealizedConversionCastOp>(
                scopeLoc, /*TypeOut*/ int1Ty, /*Inputs*/ finalRange)
            .getResult(0);
    return true;
  }
  return false;
}
bool CIRClauseProcessor::processIf(mlir::omp::IfClauseOps &result) const {
  const clang::OMPIfClause *clause = findUniqueClause<clang::OMPIfClause>();
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
  return false;
}
bool CIRClauseProcessor::processPriority(
    mlir::omp::PriorityClauseOps &result) const {
  const clang::OMPPriorityClause *clause =
      findUniqueClause<clang::OMPPriorityClause>();
  if (clause) {
    auto scopeLoc = this->CGF.getLoc(dirCtx.getSourceRange());
    const clang::Expr *priorityExpr = clause->getPriority();
    mlir::Value priorityValue = this->CGF.buildScalarExpr(priorityExpr);
    mlir::ValueRange priorityRange(priorityValue);
    mlir::Type uint32Ty = builder.getI32Type();
    result.priorityVar =
        builder
            .create<mlir::UnrealizedConversionCastOp>(
                scopeLoc, /*TypeOut*/ uint32Ty, /*Inputs*/ priorityRange)
            .getResult(0);
    return true;
  }
  return false;
}
// Taskloop clauses
bool CIRClauseProcessor::processGrainSize(
    mlir::omp::GrainsizeClauseOps &result) const {
  // Check existence of mutally exclusive clause num_tasks
  const clang::OMPNumTasksClause *tasksClause =
      findUniqueClause<clang::OMPNumTasksClause>();
  const clang::OMPGrainsizeClause *grainClause =
      findUniqueClause<clang::OMPGrainsizeClause>();
  if (tasksClause and grainClause) {
    // This should be replaced by a proper error and not llvm_unreachable
    // probably
    llvm_unreachable("error: 'num_tasks' and 'grainsize' clause are mutually "
                     "exclusive and may not appear on the same directive!");
  }
  if (grainClause) {
    auto scopeLoc = this->CGF.getLoc(dirCtx.getSourceRange());
    const clang::Expr *grainExpr = grainClause->getGrainsize();
    mlir::Value grainValue = this->CGF.buildScalarExpr(grainExpr);
    mlir::ValueRange grainRange(grainValue);
    mlir::Type uint32Ty = builder.getI32Type();
    result.grainsizeVar =
        builder
            .create<mlir::UnrealizedConversionCastOp>(
                scopeLoc, /*TypeOut*/ uint32Ty, /*Inputs*/ grainRange)
            .getResult(0);
    return true;
  }
  return false;
}

bool CIRClauseProcessor::processNumTasks(
    mlir::omp::NumTasksClauseOps &result) const {
  // check existence of mutually excluse clause grainsize
  const clang::OMPNumTasksClause *tasksClause =
      findUniqueClause<clang::OMPNumTasksClause>();
  const clang::OMPGrainsizeClause *grainClause =
      findUniqueClause<clang::OMPGrainsizeClause>();
  if (tasksClause and grainClause) {
    // This should be replaced by a proper error and not llvm_unreachable
    // probably
    llvm_unreachable("error: 'num_tasks' and 'grainsize' clause are mutually "
                     "exclusive and may not appear on the same directive!");
  }
  if (tasksClause) {
    auto scopeLoc = this->CGF.getLoc(dirCtx.getSourceRange());
    const clang::Expr *tasksExpr = tasksClause->getNumTasks();
    mlir::Value tasksValue = this->CGF.buildScalarExpr(tasksExpr);
    mlir::ValueRange tasksRange(tasksValue);
    mlir::Type uint32Ty = builder.getI32Type();
    result.numTasksVar =
        builder
            .create<mlir::UnrealizedConversionCastOp>(
                scopeLoc, /*TypeOut*/ uint32Ty, /*Inputs*/ tasksRange)
            .getResult(0);
    return true;
  }
  return false;
}
bool CIRClauseProcessor::processNogroup(
    mlir::omp::NogroupClauseOps &result) const {
  return markClauseOccurrence<clang::OMPNogroupClause>(result.nogroupAttr);
}
bool CIRClauseProcessor::processDepend(mlir::omp::DependClauseOps &result,
                                       cir::OMPTaskDataTy &data,
                                       mlir::Location &location) const {
  llvm_unreachable("The following clause is not yet implemented: Depend");
  return findRepeatableClause<clang::OMPDependClause>(
      // Get an mlir value for the pointer of each variable in the var list
      [&](const clang::OMPDependClause *clause) {
        // Get the depend type
        mlir::omp::ClauseTaskDependAttr dependType =
            getDependKindAttr(this->builder, clause);
        auto capturedVarsBegin = clause->varlist_begin();
        auto capturedVarsEnd = clause->varlist_end();
        // Get an mlir value for the pointer of each variable in the var list
        for (auto varIt = capturedVarsBegin; varIt != capturedVarsEnd;
             ++varIt) {
          const clang::DeclRefExpr *varRef =
              dyn_cast<clang::DeclRefExpr>(*varIt);

          if (varRef) {
            cir::LValue capturedLvalue = this->CGF.buildLValue(varRef);
            cir::Address capturedAddress =
                this->CGF.buildLoadOfReference(capturedLvalue, location);
            mlir::Value rawPointer = capturedAddress.emitRawPointer();

            result.dependVars.push_back(rawPointer);
            result.dependTypeAttrs.push_back(dependType);
          }
        }
      });
}
// Helper functions
mlir::omp::ClauseTaskDependAttr
CIRClauseProcessor::getDependKindAttr(cir::CIRGenBuilderTy &builder,
                                      const clang::OMPDependClause *clause) {
  const clang::OpenMPDependClauseKind kind = clause->getDependencyKind();
  mlir::omp::ClauseTaskDepend mlirKind;
  switch (kind) {
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
llvm::StringRef
CIRClauseProcessor::getClauseName(const clang::OMPClause *clause) {
  const clang::OpenMPClauseKind clauseKind = clause->getClauseKind();
  switch (clauseKind) {
  case clang::OpenMPClauseKind::OMPC_ordered:
    return "ordered";
  case clang::OpenMPClauseKind::OMPC_nowait:
    return "nowait";
  case clang::OpenMPClauseKind::OMPC_untied:
    return "untied";
  case clang::OpenMPClauseKind::OMPC_mergeable:
    return "mergeable";
  case clang::OpenMPClauseKind::OMPC_read:
    return "read";
  case clang::OpenMPClauseKind::OMPC_write:
    return "write";
  case clang::OpenMPClauseKind::OMPC_update:
    return "update";
  case clang::OpenMPClauseKind::OMPC_capture:
    return "capture";
  case clang::OpenMPClauseKind::OMPC_compare:
    return "compare";
  case clang::OpenMPClauseKind::OMPC_fail:
    return "fail";
  case clang::OpenMPClauseKind::OMPC_seq_cst:
    return "seq_cst";
  case clang::OpenMPClauseKind::OMPC_acq_rel:
    return "acq_rel";
  case clang::OpenMPClauseKind::OMPC_acquire:
    return "acquire";
  case clang::OpenMPClauseKind::OMPC_release:
    return "release";
  case clang::OpenMPClauseKind::OMPC_relaxed:
    return "relaxed";
  case clang::OpenMPClauseKind::OMPC_weak:
    return "weak";
  case clang::OpenMPClauseKind::OMPC_threads:
    return "threads";
  case clang::OpenMPClauseKind::OMPC_simd:
    return "simd";
  case clang::OpenMPClauseKind::OMPC_nogroup:
    return "nogroup";
  case clang::OpenMPClauseKind::OMPC_unified_address:
    return "unified_address";
  case clang::OpenMPClauseKind::OMPC_unified_shared_memory:
    return "unified_shared_memory";
  case clang::OpenMPClauseKind::OMPC_reverse_offload:
    return "reverse_offload";
  case clang::OpenMPClauseKind::OMPC_dynamic_allocators:
    return "dynamic_allocators";
  case clang::OpenMPClauseKind::OMPC_destroy:
    return "destroy";
  case clang::OpenMPClauseKind::OMPC_full:
    return "full";
  case clang::OpenMPClauseKind::OMPC_partial:
    return "partial";
  case clang::OpenMPClauseKind::OMPC_ompx_bare:
    return "ompx_bare";
  case clang::OpenMPClauseKind::OMPC_if:
    return "if";
  case clang::OpenMPClauseKind::OMPC_final:
    return "final";
  case clang::OpenMPClauseKind::OMPC_num_threads:
    return "num_threads";
  case clang::OpenMPClauseKind::OMPC_safelen:
    return "safelen";
  case clang::OpenMPClauseKind::OMPC_simdlen:
    return "simdlen";
  case clang::OpenMPClauseKind::OMPC_sizes:
    return "sizes";
  case clang::OpenMPClauseKind::OMPC_allocator:
    return "allocator";
  case clang::OpenMPClauseKind::OMPC_collapse:
    return "collapse";
  case clang::OpenMPClauseKind::OMPC_schedule:
    return "schedule";
  case clang::OpenMPClauseKind::OMPC_private:
    return "private";
  case clang::OpenMPClauseKind::OMPC_firstprivate:
    return "firstprivate";
  case clang::OpenMPClauseKind::OMPC_lastprivate:
    return "lastprivate";
  case clang::OpenMPClauseKind::OMPC_shared:
    return "shared";
  case clang::OpenMPClauseKind::OMPC_reduction:
    return "reduction";
  case clang::OpenMPClauseKind::OMPC_task_reduction:
    return "task_reduction";
  case clang::OpenMPClauseKind::OMPC_in_reduction:
    return "in_reduction";
  case clang::OpenMPClauseKind::OMPC_linear:
    return "linear";
  case clang::OpenMPClauseKind::OMPC_aligned:
    return "aligned";
  case clang::OpenMPClauseKind::OMPC_copyin:
    return "copyin";
  case clang::OpenMPClauseKind::OMPC_copyprivate:
    return "copyprivate";
  case clang::OpenMPClauseKind::OMPC_default:
    return "default";
  case clang::OpenMPClauseKind::OMPC_proc_bind:
    return "proc_bind";
  case clang::OpenMPClauseKind::OMPC_threadprivate:
    return "threadprivate";
  case clang::OpenMPClauseKind::OMPC_allocate:
    return "allocate";
  case clang::OpenMPClauseKind::OMPC_flush:
    return "flush";
  case clang::OpenMPClauseKind::OMPC_depobj:
    return "depobj";
  case clang::OpenMPClauseKind::OMPC_depend:
    return "depend";
  case clang::OpenMPClauseKind::OMPC_device:
    return "device";
  case clang::OpenMPClauseKind::OMPC_map:
    return "map";
  case clang::OpenMPClauseKind::OMPC_num_teams:
    return "num_teams";
  case clang::OpenMPClauseKind::OMPC_thread_limit:
    return "thread_limit";
  case clang::OpenMPClauseKind::OMPC_priority:
    return "priority";
  case clang::OpenMPClauseKind::OMPC_grainsize:
    return "grainsize";
  case clang::OpenMPClauseKind::OMPC_num_tasks:
    return "num_tasks";
  case clang::OpenMPClauseKind::OMPC_hint:
    return "hint";
  case clang::OpenMPClauseKind::OMPC_dist_schedule:
    return "dist_schedule";
  case clang::OpenMPClauseKind::OMPC_defaultmap:
    return "defaultmap";
  case clang::OpenMPClauseKind::OMPC_unknown:
    return "unknown";
  case clang::OpenMPClauseKind::OMPC_uniform:
    return "uniform";
  case clang::OpenMPClauseKind::OMPC_to:
    return "to";
  case clang::OpenMPClauseKind::OMPC_from:
    return "from";
  case clang::OpenMPClauseKind::OMPC_use_device_ptr:
    return "use_device_ptr";
  case clang::OpenMPClauseKind::OMPC_use_device_addr:
    return "use_device_addr";
  case clang::OpenMPClauseKind::OMPC_is_device_ptr:
    return "is_device_ptr";
  case clang::OpenMPClauseKind::OMPC_has_device_addr:
    return "has_device_addr";
  case clang::OpenMPClauseKind::OMPC_atomic_default_mem_order:
    return "atomic_default_mem_order";
  case clang::OpenMPClauseKind::OMPC_device_type:
    return "device_type";
  case clang::OpenMPClauseKind::OMPC_match:
    return "match";
  case clang::OpenMPClauseKind::OMPC_nontemporal:
    return "nontemporal";
  case clang::OpenMPClauseKind::OMPC_order:
    return "order";
  case clang::OpenMPClauseKind::OMPC_at:
    return "at";
  case clang::OpenMPClauseKind::OMPC_severity:
    return "severity";
  case clang::OpenMPClauseKind::OMPC_message:
    return "message";
  case clang::OpenMPClauseKind::OMPC_novariants:
    return "novariants";
  case clang::OpenMPClauseKind::OMPC_nocontext:
    return "nocontext";
  case clang::OpenMPClauseKind::OMPC_detach:
    return "detach";
  case clang::OpenMPClauseKind::OMPC_inclusive:
    return "inclusive";
  case clang::OpenMPClauseKind::OMPC_exclusive:
    return "exclusive";
  case clang::OpenMPClauseKind::OMPC_uses_allocators:
    return "uses_allocators";
  case clang::OpenMPClauseKind::OMPC_affinity:
    return "affinity";
  case clang::OpenMPClauseKind::OMPC_when:
    return "when";
  case clang::OpenMPClauseKind::OMPC_ompx_dyn_cgroup_mem:
    return "ompx_dyn_cgroup_mem";
  default:
    return "unknown?";
  }
}