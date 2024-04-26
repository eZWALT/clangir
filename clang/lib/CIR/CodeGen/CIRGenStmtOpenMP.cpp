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

#include <clang/AST/StmtIterator.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LogicalResult.h>

#include "CIRClauseProcessor.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPClauseOperands.h"
#include "mlir/IR/Value.h"

#include <clang/AST/ASTFwd.h>
#include <clang/AST/StmtOpenMP.h>
#include <clang/Basic/OpenMPKinds.h>
#include <cstdint>

#include "mlir/Dialect/Arith/IR/Arith.h"
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
mlir::LogicalResult CIRGenFunction::buildCapturedStatement(
    OmpOp &operation, const OMPExecutableDirective &S,
    OpenMPDirectiveKind DKind, bool useCurrentScope) {
  mlir::Location scopeLoc = getLoc(S.getSourceRange());
  const Stmt *capturedStmt = S.getCapturedStmt(DKind)->getCapturedStmt();
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

// TODO: it should be emitted through runtime and follow Clang Skeleton
mlir::LogicalResult
CIRGenFunction::buildOMPTaskDirective(const OMPTaskDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());

  // Create the values and attributes that will be consumed by omp.task
  mlir::UnitAttr untiedAttr, mergeableAttr;
  mlir::Value finalOperand, ifOperand, priorityOperand;
  mlir::omp::DependClauseOps dependOperands;

  OMPTaskDataTy data;
  buildDependences(S, data);
  // Evalutes clauses
  CIRClauseProcessor cp = CIRClauseProcessor(*this, S);
  cp.processUntied(untiedAttr);
  cp.processMergeable(mergeableAttr);
  cp.processFinal(finalOperand);
  cp.processIf(ifOperand);
  cp.processPriority(priorityOperand);
  cp.processDepend(dependOperands, data);

  // Create a `omp.task` operation
  // TODO: add support to these OpenMP v5 features
  mlir::omp::TaskOp taskOp = builder.create<mlir::omp::TaskOp>(
      /*Location*/ scopeLoc,
      /*optional If value*/ ifOperand,
      /*optional Final value*/ finalOperand,
      /*optional Untied attribute*/ untiedAttr,
      /*optional Mergeable attribute*/ mergeableAttr,
      /*In Reduction variables*/ mlir::ValueRange(),
      /*optional In Reductions*/ nullptr,
      /*optional priority value*/ priorityOperand,
      /*optional Dependency Types values*/ dependOperands.dependTypeAttrs,
      /*Dependencies values*/ dependOperands.dependVars,
      /*Allocate values*/ mlir::ValueRange(),
      /*Allocator values*/ mlir::ValueRange());
  // Build the captured statement CIR region
  res = buildCapturedStatement(taskOp, S, OpenMPDirectiveKind::OMPD_task,
                               /*useCurrentScope*/ true);
  return res;
}