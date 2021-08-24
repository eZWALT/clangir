<<<<<<< HEAD
include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

add_subdirectory(Dialect)
add_subdirectory(CodeGen)
add_subdirectory(FrontendAction)
add_subdirectory(Lowering)
add_subdirectory(Interfaces)
=======
set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

include_directories( ${LLVM_MAIN_SRC_DIR}/../mlir/include )
include_directories( ${CMAKE_BINARY_DIR}/tools/mlir/include )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIR
  CIRBuilder.cpp

  DEPENDS
  MLIRCIROpsIncGen

  LINK_LIBS
  clangAST
  clangBasic
  clangEdit
  clangLex
  ${dialect_libs}
  MLIRCIR
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRSupport
  )
>>>>>>> 90a0c4bb8b11 ([CIR] Add more dialect bits for a return op, add more builder content)
