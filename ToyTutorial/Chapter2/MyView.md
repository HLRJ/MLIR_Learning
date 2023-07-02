# 学习mlir的感悟

为什么要有mlir， 由于 C、C++ 源码直接转成 AST 时，并不会进行语言特定的优化，程序的优化主要集中于 LLVM IR 阶段。
但 LLVM IR 表示层级较低，会丢失源码中的部分信息(如报错信息)，会导致优化不充分。

类似于Tensorflow、Keras等框架，会先转化为计算图Computation Graph形式，然后会基于图做一定的优化。但图阶段缺少硬件部署的相关信息，
所以后续会转化为某个后端的内部表示，根据不同的硬件(TPU、Phone)，进行算子融合等等优化。


.toy 源文件->
AST ->
MLIRGen(遍历AST生成MLIR表达式) ->
Transformation(变形消除冗余) ->
Lowering ->
LLVM IR / JIT 编译引擎