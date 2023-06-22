#


[官网](https://mlir.llvm.org/docs/Tutorials/Toy/)


- Chapter1 --介绍Toy语言以及抽象语法树AST
- Chapter2 --定义Toy Dialect， Toy Operation， 生成MLIR表达式
- Chapter3 --Toy Operation层级的表达式变型
- Chapter4 --使用接口，完成泛化的表达式变型
- Chapter5 --将MLIR表达式进行部分Lowering，并进行优化
- Chapter6 --混合Dialect表达式Lowering到LLVM IR
- Chapter7 --扩展源语言，向Toy语言添加struct数据类型
-----

- 1->2 添加Dialect描述、分析AST、生成MLIR表达式
- 2->3 当MLIR表达式存在冗余，进行针对Operation的表达式变型
- 3->4 使用已有接口，完成泛化的表达式变型
- 4->5 将Toy Dialect的部分Operation映射到Affine Dialect Operation，对Affine MLIR表达式进行优化
- 5->6 混合Dialect MLIR表达式 -> LLVM IR Dialect MLIR表达式 -> LLVM IR 表达式
- 6->7 在现有的编译流程中，添加自定义的数据类型



