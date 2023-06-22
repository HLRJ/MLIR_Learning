# Chapter 2: Emitting Basic MLIR

现在我们已经熟悉了我们的语言和AST，让我们看看MLIR如何帮助编译Toy。

## Introduction: Multi-Level Intermediate Representation 

其他编译器，如LLVM（见[Kaleidoscope](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)教程），提供了一套固定的预定义类型和（通常是低级/类似RISC）指令。
在发出LLVM IR之前，由特定语言的前端来执行任何特定语言的类型检查、分析或转换。例如，Clang将使用其AST不仅进行静态分析，还进行转换，
如通过AST克隆和重写进行C++模板实例化。
最后，在比C/C++更高层次上构建的语言可能需要从它们的AST中进行非实质性的降低，以生成LLVM IR。

因此，多个前端最终要重新实现大量的基础设施，以支持这些分析和转换的需要。MLIR通过设计可扩展性来解决这个问题。
因此，很少有预定义的指令（MLIR术语中的操作）或类型。

## Interfacing with MLIR

[Language Reference](https://mlir.llvm.org/docs/LangRef/)

MLIR被设计成一个完全可扩展的基础设施；没有封闭的属性集（想想看：常量元数据）、操作或类型。MLIR通过方言的概念支持这种可扩展性。
方言提供了一种分组机制，用于在一个独特的命名空间下进行抽象化。

在MLIR中，[Operations](https://mlir.llvm.org/docs/LangRef/#operations)是抽象和计算的核心单位，在许多方面与LLVM指令类似。操作可以有特定的应用语义，可以用来表示LLVM中所有的核心IR结构：
指令、球（如函数）、模块等。

下面是一个 transpose的 operation的MLIR汇编。

``` 
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
```

让我们来分析一下这个MLIR operation 的结构：

- %t_tensor
  - 这个operation所定义的结果的名称（包括一个前缀，以避免碰撞）。一个operation可以定义零个或多个结果（在Toy的上下文中，我们只限于单结果操作），它们是SSA值。
名称在解析过程中使用，但不是持久的（例如，它不在SSA值的内存表示中被跟踪）。
- "toy.transpose"
  - operation的名称。它应该是一个唯一的字符串，在". "前加上方言的名称空间。这可以被理解为toy dialect中的转置操作。
- (%tensor)
  - 一个由零个或多个输入operand（或参数）组成的列表，这些操作数是由其他操作定义的SSA值或指代块参数。
- { inplace = true }
  - 一个由零个或多个属性组成的字典，这些属性是特殊的operand，始终是常量。这里我们定义了一个名为 "inplace "的布尔属性，其恒定值为true。
- (tensor<2x3xf64>) -> tensor<3x2xf64>
  - 这指的是函数形式的operation类型，在括号里拼出参数的类型，后面是返回值的类型。
- loc("example/file/path":12:1)
  - 这是该操作在源代码中的位置。

这里显示的是一个operation的一般形式。如上所述，MLIR中的操作集是可扩展的。
operation是用一小套概念来建模的，使操作可以被推理和通用operation。这些概念是：
- operation的名字
- ssa operand 值的列表
- attributes 的列表
- 结果值的类型的列表
- 一个用于调试的源代码位置
- 后继块的列表（主要用于分支）
- 一个区域的列表（用于结构性的operation，比如函数）

在MLIR中，每个操作都与一个强制的源位置（source location）相关联。
与LLVM相反，在LLVM中调试信息位置是元数据，可以被丢弃，而在MLIR中，位置是一个核心要求，并且API依赖于它并对其进行操作。
丢弃一个位置是一个明确的选择，不会出现错误。

举个例子来说明：如果一个转换将一个操作替换为另一个操作，那么新的操作仍必须附带一个位置。这样可以跟踪该操作来自何处。

值得注意的是，mlir-opt工具是用于测试编译器传递的工具，默认情况下不会在输出中包含位置信息。使用"-mlir-print-debuginfo"标志可以指定包含位置信息。
（运行"mlir-opt --help"获取更多选项。）

## Opaque API

MLIR的设计允许所有的IR元素，如属性、操作和类型，都可以被定制。同时，IR元素总是可以被简化为上述基本概念。这使得MLIR可以为任何操作解析、表示和往返IR。
例如，我们可以把上面的Toy操作放到一个.mlir文件中，通过mlir-opt进行round-trip，而不需要注册任何与Toy有关的方言：

