//===--- CIRGenStmtOpenMP.cpp - Emit MLIR Code from OpenMP Statements -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit OpenMP Stmt nodes as MLIR code.
//
//===----------------------------------------------------------------------===//

#include <clang/AST/Stmt.h>
#include <clang/AST/StmtIterator.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LogicalResult.h>

#include "CIRClauseProcessor.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"

#include <clang/AST/ASTFwd.h>
#include <clang/AST/StmtOpenMP.h>
#include <clang/Basic/OpenMPKinds.h>
#include <cstdint>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenMP/OpenMPClauseOperands.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Value.h"
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace cir;
using namespace clang;
using namespace mlir::omp;

static void buildDependences(const OMPExecutableDirective &S,
                             OMPTaskDataTy &Data) {

  // First look for 'omp_all_memory' and add this first.
  bool OmpAllMemory = false;
  if (llvm::any_of(
          S.getClausesOfKind<OMPDependClause>(), [](const OMPDependClause *C) {
            return C->getDependencyKind() == OMPC_DEPEND_outallmemory ||
                   C->getDependencyKind() == OMPC_DEPEND_inoutallmemory;
          })) {
    OmpAllMemory = true;
    // Since both OMPC_DEPEND_outallmemory and OMPC_DEPEND_inoutallmemory are
    // equivalent to the runtime, always use OMPC_DEPEND_outallmemory to
    // simplify.
    OMPTaskDataTy::DependData &DD =
        Data.Dependences.emplace_back(OMPC_DEPEND_outallmemory,
                                      /*IteratorExpr=*/nullptr);
    // Add a nullptr Expr to simplify the codegen in emitDependData.
    DD.DepExprs.push_back(nullptr);
  }
  // Add remaining dependences skipping any 'out' or 'inout' if they are
  // overridden by 'omp_all_memory'.
  for (const auto *C : S.getClausesOfKind<OMPDependClause>()) {
    OpenMPDependClauseKind Kind = C->getDependencyKind();
    if (Kind == OMPC_DEPEND_outallmemory || Kind == OMPC_DEPEND_inoutallmemory)
      continue;
    if (OmpAllMemory && (Kind == OMPC_DEPEND_out || Kind == OMPC_DEPEND_inout))
      continue;
    OMPTaskDataTy::DependData &DD =
        Data.Dependences.emplace_back(C->getDependencyKind(), C->getModifier());
    DD.DepExprs.append(C->varlist_begin(), C->varlist_end());
  }
}

// Operation to create the captured statement of any OpenMP directive
template <typename OmpOp>
mlir::LogicalResult CIRGenFunction::buildOMPCapturedStatement(
    OmpOp &operation, const OMPExecutableDirective &S,
    OpenMPDirectiveKind DKind, bool useCurrentScope) {
  mlir::Location scopeLoc = getLoc(S.getSourceRange());

  const Stmt *capturedStmt;

  if (DKind == clang::OpenMPDirectiveKind::OMPD_taskgroup) {
    auto x = (llvm::dyn_cast<const clang::CapturedStmt>(S.getAssociatedStmt()));
    capturedStmt = x->getCapturedStmt();
  } else
    capturedStmt = S.getCapturedStmt(DKind)->getCapturedStmt();

  mlir::LogicalResult res = mlir::success();
  mlir::Block &block = operation.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);
  // Create a scope for the OpenMP region.
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the body of the region.
        if (buildStmt(capturedStmt, useCurrentScope).failed())
          res = mlir::failure();
      });
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}

// mlir::LogicalResult CIRGenFunction::buildOMPLoopBody(){return
// mlir::success();}

mlir::LogicalResult
CIRGenFunction::buildOMPParallelDirective(const OMPParallelDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  // Create a `omp.parallel` op.
  auto parallelOp = builder.create<ParallelOp>(scopeLoc);
  mlir::Block &block = parallelOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);
  // Create a scope for the OpenMP region.
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the body of the region.
        if (buildStmt(S.getCapturedStmt(OpenMPDirectiveKind::OMPD_parallel)
                          ->getCapturedStmt(),
                      /*useCurrentScope=*/true)
                .failed())
          res = mlir::failure();
      });
  // Add the terminator for `omp.parallel`.
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPTaskwaitDirective(const OMPTaskwaitDirective &S) {
  mlir::LogicalResult res = mlir::success();
  // Getting the source location information of AST node S scope
  auto scopeLoc = getLoc(S.getSourceRange());
  OMPTaskDataTy Data;
  buildDependences(S, Data);
  Data.HasNowaitClause = S.hasClausesOfKind<OMPNowaitClause>();
  CGM.getOpenMPRuntime().emitTaskWaitCall(*this, scopeLoc, Data, builder);
  return res;
}
mlir::LogicalResult
CIRGenFunction::buildOMPTaskyieldDirective(const OMPTaskyieldDirective &S) {
  mlir::LogicalResult res = mlir::success();
  // Getting the source location information of AST node S scope
  auto scopeLoc = getLoc(S.getSourceRange());
  // Creation of an omp.taskyield operation
  auto taskyieldOp = builder.create<mlir::omp::TaskyieldOp>(scopeLoc);

  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPBarrierDirective(const OMPBarrierDirective &S) {
  mlir::LogicalResult res = mlir::success();
  // Getting the source location information of AST node SD scope
  auto scopeLoc = getLoc(S.getSourceRange());
  // Creation of an omp.barrier operation
  auto barrierOp = builder.create<mlir::omp::BarrierOp>(scopeLoc);

  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPTaskgroupDirective(const OMPTaskgroupDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  bool useCurrentScope = true;
  mlir::omp::TaskgroupClauseOps clauseOps;

  CIRClauseProcessor cp = CIRClauseProcessor(*this, S);
  cp.processTODO<clang::OMPTaskReductionClause, clang::OMPAllocateClause>();

  mlir::omp::TaskgroupOp taskgroupOp =
      builder.create<mlir::omp::TaskgroupOp>(scopeLoc, clauseOps);
  const Stmt *capturedStmt = S.getInnermostCapturedStmt()->getCapturedStmt();

  mlir::Block &block = taskgroupOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);

  // Create a scope for the OpenMP region.
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the body of the region.
        if (buildStmt(capturedStmt, useCurrentScope).failed())
          res = mlir::failure();
      });
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPCriticalDirective(const OMPCriticalDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  // WIP: named critical regions still not supported
  mlir::FlatSymbolRefAttr refAttr;

  mlir::omp::CriticalOp criticalOp =
      builder.create<mlir::omp::CriticalOp>(scopeLoc, refAttr);
  const Stmt *capturedStmt = S.getAssociatedStmt();
  // if(!capturedStmt) return mlir::failure();
  mlir::Block &block = criticalOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);

  // Build an scope for the critical region
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the statement within the critical region
        if (buildStmt(capturedStmt, /*useCurrentScope=*/true).failed())
          res = mlir::failure();
      });
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPMasterDirective(const OMPMasterDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());

  mlir::omp::MasterOp masterOp = builder.create<mlir::omp::MasterOp>(scopeLoc);
  const Stmt *capturedStmt = S.getAssociatedStmt();

  mlir::Block &block = masterOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);

  // Build an scope for the critical region
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the statement within the critical region
        if (buildStmt(capturedStmt, /*useCurrentScope=*/true).failed())
          res = mlir::failure();
      });
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}

mlir::LogicalResult 
CIRGenFunction::buildOMPSingleDirective(const OMPSingleDirective &S){
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());

  //WIP: treatment of single clauses
  mlir::omp::SingleClauseOps clauseOps;
  auto singleOp = builder.create<mlir::omp::SingleOp>(scopeLoc, clauseOps);
  const Stmt* capturedStmt = S.getInnermostCapturedStmt()->getCapturedStmt();

  mlir::Block& block = singleOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);

    // Build an scope for the critical region
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the statement within the critical region
        if (buildStmt(capturedStmt, /*useCurrentScope=*/true).failed())
          res = mlir::failure();
      });
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}

// TODO: it should be emitted through runtime and follow Clang Skeleton
mlir::LogicalResult
CIRGenFunction::buildOMPTaskDirective(const OMPTaskDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());

  OMPTaskDataTy data;
  data.Tied = not S.getSingleClause<clang::OMPUntiedClause>();
  clang::OpenMPDirectiveKind DKind = clang::OpenMPDirectiveKind::OMPD_task;
  bool useCurrentScope = true;
  // Create the values and attributes that will be consumed by omp.task
  mlir::omp::TaskClauseOps clauseOps;
  
  CIRClauseProcessor cp = CIRClauseProcessor(*this, S);
  cp.processUntied(clauseOps);
  cp.processMergeable(clauseOps);
  cp.processFinal(clauseOps);
  cp.processIf(clauseOps);
  cp.processPriority(clauseOps);
  cp.processDepend(clauseOps, data, scopeLoc);
  // TODO(cir) Give support to this OpenMP v.5 clauses
  cp.processTODO<clang::OMPAllocateClause, clang::OMPInReductionClause,clang::OMPAffinityClause, clang::OMPDetachClause,clang::OMPDefaultClause>();

    
  // Create a `omp.task` operation
  mlir::omp::TaskOp taskOp = builder.create<mlir::omp::TaskOp>(scopeLoc, clauseOps);
  // Build the captured statement CIR region
  res = buildOMPCapturedStatement(taskOp, S, DKind,
                                  /*useCurrentScope*/ useCurrentScope);

  /*
  auto &&BodyGen = [&S, &DKind, useCurrentScope](CIRGenFunction &CGF,
  mlir::omp::TaskOp& taskOp) { return CGF.buildOMPCapturedStatement(taskOp, S,
  DKind, useCurrentScope);
  };
  //EMIT TASKGEN FUNCTION &&
  auto &&TaskGen = [&S, SharedsTy, CapturedStruct,
                    IfCond](CIRGenFunction &CGF, llvm::Function *OutlinedFn,
                            const OMPTaskDataTy &Data) {
    CGF.CGM.getOpenMPRuntime().emitTaskCall(CGF, S.getBeginLoc(), S, OutlinedFn,
                                            SharedsTy, CapturedStruct, IfCond,
                                            Data);
  };

  */
  // CGM.getOpenMPRuntime().emitOMPTaskBasedDirective(/*CGF*/this,
  // /*Location*/scopeLoc, /*Data*/data);
  return res;
}