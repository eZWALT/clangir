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
#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRCLAUSEPROCESSOR_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRCLAUSEPROCESSOR_H

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include <llvm/ADT/SmallVector.h>

#include "clang/AST/StmtOpenMP.h"
#include "clang/Basic/OpenMPKinds.h"
#include <clang/AST/ASTFwd.h>

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Value.h"
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>

class CIRClauseProcessor {

private:
  cir::CIRGenFunction &CGF;

public:
  CIRClauseProcessor(cir::CIRGenFunction &CGF) : CGF(CGF) {}

  // At most 1 ocurrence clauses

  bool processFinal(const clang::OMPExecutableDirective &dirCtx,
                    mlir::Value &result) const;

  bool processIf(const clang::OMPExecutableDirective &dirCtx,
                 mlir::Value &result) const;

  bool processPriority(const clang::OMPExecutableDirective &dirCtx,
                       mlir::Value &result) const;

  bool processUntied(const clang::OMPExecutableDirective &dirCtx,
                     mlir::UnitAttr &result) const;

  bool processMergeable(const clang::OMPExecutableDirective &dirCtx,
                        mlir::UnitAttr &result) const;

  bool processDepend(const clang::OMPExecutableDirective &dirCtx,
                     mlir::ArrayAttr &dependTypeOperands,
                     llvm::SmallVector<mlir::Value> &dependOperands) const;

  bool processPrivate(const clang::OMPExecutableDirective &dirCtx) const;

  bool processFirstPrivate(const clang::OMPExecutableDirective &dirCtx) const;
};

#endif