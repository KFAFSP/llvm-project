add_mlir_library(MLIRTyping
  Bound.cpp
  Context.cpp
  Contradiction.cpp
  FixPointTypeChecker.cpp
  LexicalOpQueue.cpp
  Transmutation.cpp
  TypeChecker.cpp
  TypeCheckOpInterface.cpp

  DEPENDS
  MLIRTypeCheckOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
)
