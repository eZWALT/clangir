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
#include "CIRClauseProcessor.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenMP/OpenMPClauseOperands.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Value.h"
#include <clang/AST/ASTFwd.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/StmtIterator.h>
#include <clang/AST/StmtOpenMP.h>
#include <clang/Basic/OpenMPKinds.h>
#include <cstdint>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LogicalResult.h>
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
  OMPTaskDataTy Data;
  buildDependences(S, Data);
  Data.HasNowaitClause = S.hasClausesOfKind<OMPNowaitClause>();
  CGM.getOpenMPRuntime().emitTaskWaitCall(builder, *this,
                                          getLoc(S.getSourceRange()), Data);
  return res;
}
mlir::LogicalResult
CIRGenFunction::buildOMPTaskyieldDirective(const OMPTaskyieldDirective &S) {
  mlir::LogicalResult res = mlir::success();
  // Creation of an omp.taskyield operation
  CGM.getOpenMPRuntime().emitTaskyieldCall(builder, *this,
                                           getLoc(S.getSourceRange()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPBarrierDirective(const OMPBarrierDirective &S) {
  mlir::LogicalResult res = mlir::success();
  // Creation of an omp.barrier operation
  CGM.getOpenMPRuntime().emitBarrierCall(builder, *this,
                                         getLoc(S.getSourceRange()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPTaskgroupDirective(const OMPTaskgroupDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  bool useCurrentScope = true;
  // Clause handling
  mlir::omp::TaskgroupClauseOps clauseOps;
  CIRClauseProcessor cp = CIRClauseProcessor(*this, S);
  cp.processTODO<clang::OMPTaskReductionClause, clang::OMPAllocateClause>();
  // Generation of taskgroup op
  mlir::omp::TaskgroupOp taskgroupOp =
      builder.create<mlir::omp::TaskgroupOp>(scopeLoc, clauseOps);
  // Getting the captured statement
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
  // Add a terminator op to delimit the scope
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}
mlir::LogicalResult
CIRGenFunction::buildOMPCriticalDirective(const OMPCriticalDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  // WIP: named critical regions still not supported
  mlir::FlatSymbolRefAttr refAttr;
  // Generation of critical op
  mlir::omp::CriticalOp criticalOp =
      builder.create<mlir::omp::CriticalOp>(scopeLoc, refAttr);
  // Getting the captured statement
  const Stmt *capturedStmt = S.getAssociatedStmt();
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
  // Add a terminator op to delimit the scope
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPMasterDirective(const OMPMasterDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  // Generation of master op
  mlir::omp::MasterOp masterOp = builder.create<mlir::omp::MasterOp>(scopeLoc);
  // Getting the captured statement
  const Stmt *capturedStmt = S.getAssociatedStmt();

  mlir::Block &block = masterOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);
  // Build an scope for the master region
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the statement within the master region
        if (buildStmt(capturedStmt, /*useCurrentScope=*/true).failed())
          res = mlir::failure();
      });
  // Add a terminator op to delimit the scope
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPSingleDirective(const OMPSingleDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());

  // WIP: treatment of single clauses
  // Clause handling
  mlir::omp::SingleClauseOps clauseOps;
  CIRClauseProcessor cp = CIRClauseProcessor(*this, S);
  cp.processNowait(clauseOps);
  // Generation of taskgroup op
  auto singleOp = builder.create<mlir::omp::SingleOp>(scopeLoc, clauseOps);
  // Getting the captured statement
  const Stmt *capturedStmt = S.getInnermostCapturedStmt()->getCapturedStmt();

  mlir::Block &block = singleOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);

  // Build an scope for the single region
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the statement within the single region
        if (buildStmt(capturedStmt, /*useCurrentScope=*/true).failed())
          res = mlir::failure();
      });
  // Add a terminator op to delimit the scope
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPTaskDirective(const OMPTaskDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  OMPTaskDataTy data;
  data.Tied = not S.getSingleClause<clang::OMPUntiedClause>();
  clang::OpenMPDirectiveKind DKind = clang::OpenMPDirectiveKind::OMPD_task;
  bool useCurrentScope = true;
  // Clause handling
  mlir::omp::TaskClauseOps clauseOps;
  CIRClauseProcessor cp = CIRClauseProcessor(*this, S);
  cp.processUntied(clauseOps);
  cp.processMergeable(clauseOps);
  cp.processFinal(clauseOps);
  cp.processIf(clauseOps);
  cp.processPriority(clauseOps);
  // TODO(cir) Give support to this OpenMP v.5 clauses
  cp.processTODO<clang::OMPAllocateClause, clang::OMPInReductionClause,
                 clang::OMPAffinityClause, clang::OMPDetachClause,
                 clang::OMPDefaultClause, clang::OMPDependClause>();

  // Create a `omp.task` operation
  mlir::omp::TaskOp taskOp =
      builder.create<mlir::omp::TaskOp>(scopeLoc, clauseOps);
  // Getting the captured statement
  const Stmt *capturedStmt = S.getCapturedStmt(DKind)->getCapturedStmt();

  mlir::Block &block = taskOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);

  // Create a scope for the OpenMP region.
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the body of the region.
        if (buildStmt(capturedStmt, /*useCurrentScope=*/true).failed())
          res = mlir::failure();
      });
  // Add a terminator op to delimit the scope
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}