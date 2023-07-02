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
例如，我们可以把上面的Toy操作放到一个.mlir文件中，通过mlir-opt进行round-trip，而不需要注册任何与Toy有关的Dialect：
``` 
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
```

在未注册的属性、操作和类型的情况下，MLIR会强制执行一些结构约束（如支配关系等），但除此之外它们是完全不透明的。
例如，MLIR对于未注册的操作很少了解其是否可以在特定数据类型上操作，它可以接受多少个操作数，以及它会产生多少个结果。
这种灵活性在引导过程中可能很有用，但通常不建议在成熟的系统中使用。未注册的操作在转换和分析中必须保守处理，并且它们更难构建和操作。

通过构造应该是无效的Toy IR，并观察其往返（round-trip）时没有触发验证器，可以观察到这种处理方式。

``` 
func.func @main() {
  %0 = "toy.print"() : () -> tensor<2x3xf64>
}
```

这里存在多个问题：toy.print 操作不是终结符；它应该接受一个操作数；并且它不应该返回任何值。
在下一节中，我们将使用 MLIR 注册我们的 dialect 和操作，连接到验证器，并添加更好的 API 来操作我们的操作。

## Defining a Toy Dialect

为了有效地与 MLIR 进行交互，我们将定义一个新的 Toy 方言。这个方言将模型化 Toy 语言的结构，并提供一个方便的高级分析和转换的途径。
``` 
/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types. It can
/// also override virtual methods to change some general behavior, which will be
/// demonstrated in later chapters of the tutorial.
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace.
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// An initializer called from the constructor of ToyDialect that is used to
  /// register attributes, operations, types, and more within the Toy dialect.
  void initialize();
};
```

这是Dialect的 C++ 定义，但是 MLIR 还支持通过 tablegen 声明式地定义Dialect。使用声明式规范更加简洁，因为它消除了定义新Dialect时的大部分样板代码。
它还可以轻松生成Dialect文档，可以直接在Dialect旁边进行描述。在这种声明式格式中，Toy Dialect将被指定为：

``` 
// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  // The namespace of our dialect, this corresponds 1-1 with the string we
  // provided in `ToyDialect::getDialectNamespace`.
  let name = "toy";

  // A short one-line summary of our dialect.
  let summary = "A high-level dialect for analyzing and optimizing the "
                "Toy language";

  // A much longer description of our dialect.
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results. This dialect
    provides a representation of the language that is amenable to analysis and
    optimization.
  }];

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "toy";
}
```

为了查看这个生成的内容，我们可以使用 mlir-tblgen 命令运行 gen-dialect-decls 动作，像这样：

``` 
${build_root}/bin/mlir-tblgen -gen-dialect-decls 
${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```

在Dialect被定义之后，它现在可以加载到一个 MLIRContext 中：

``` 
context.loadDialect<ToyDialect>();
```

默认情况下，MLIRContext 只加载了内置方言（Builtin Dialect），
它提供了一些核心的 IR 组件，这意味着其他Dialect，比如我们的 Toy Dialect，必须显式地加载进来。


## Defining Toy Operations

现在我们有了 Toy 方言，我们可以开始定义操作了。这将允许提供语义信息，以供系统的其他部分连接进来。
以 toy.constant 操作为例，让我们详细介绍它的创建过程。该操作将表示 Toy 语言中的一个常量值。

``` 
 %4 = "toy.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
```

这个操作不接受任何操作数，它接受一个名为 value 的 dense elements 属性来表示常量值，并返回一个 RankedTensorType 的单个结果。操
作类继承自 CRTP mlir::Op 类，该类还可以使用一些可选的 traits 来自定义其行为。
Traits 是一种机制，可以将额外的行为注入到操作中，比如额外的访问器、验证等。下面是一个可能的常量操作的定义，如上所述：

``` 
class ConstantOp : public mlir::Op<
                     /// `mlir::Op` is a CRTP class, meaning that we provide the
                     /// derived class as a template parameter.
                     ConstantOp,
                     /// The ConstantOp takes zero input operands.
                     mlir::OpTrait::ZeroOperands,
                     /// The ConstantOp returns a single result.
                     mlir::OpTrait::OneResult,
                     /// We also provide a utility `getType` accessor that
                     /// returns the TensorType of the single result.
                     mlir::OpTraits::OneTypedResult<TensorType>::Impl> {

 public:
  /// Inherit the constructors from the base Op class.
  using Op::Op;

  /// Provide the unique name for this operation. MLIR will use this to register
  /// the operation and uniquely identify it throughout the system. The name
  /// provided here must be prefixed by the parent dialect namespace followed
  /// by a `.`.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Return the value of the constant by fetching it from the attribute.
  mlir::DenseElementsAttr getValue();

  /// Operations may provide additional verification beyond what the attached
  /// traits provide.  Here we will ensure that the specific invariants of the
  /// constant operation are upheld, for example the result type must be
  /// of TensorType and matches the type of the constant `value`.
  LogicalResult verifyInvariants();

  /// Provide an interface to build this operation from a set of input values.
  /// This interface is used by the `builder` classes to allow for easily
  /// generating instances of this operation:
  ///   mlir::OpBuilder::create<ConstantOp>(...)
  /// This method populates the given `state` that MLIR uses to create
  /// operations. This state is a collection of all of the discrete elements
  /// that an operation may contain.
  /// Build a constant with the given return type and `value` attribute.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  /// Build a constant and reuse the type from the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  /// Build a constant by broadcasting the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};
```

我们可以在 ToyDialect 的初始化器中注册这个操作：
``` 
void ToyDialect::initialize() {
  addOperations<ConstantOp>();
} 
```

**Op vs Operation: Using MLIR Operations**

现在我们已经定义了一个操作，我们将想要访问和转换它。在 MLIR 中，与操作相关的主要类有两个：Operation 和 Op。
Operation 类用于通用地建模所有操作。它是“不透明”的，意味着它不描述特定操作或操作类型的属性。
相反，Operation 类提供了对操作实例的通用 API。另一方面，每种特定类型的操作都由 Op 派生类表示。
例如，ConstantOp 表示具有零个输入和一个输出的操作，输出始终设置为相同的值。Op 派生类充当围绕 Operation* 的智能指针包装器，提供特定于操作的访问器方法和类型安全的操作属性。
这意味着当我们定义 Toy 操作时，我们只是定义了一个干净、语义有用的接口，用于构建和与 Operation 类交互。
这就是为什么我们的 ConstantOp 定义没有类字段的原因；这个操作的所有数据都存储在引用的 Operation 中。
这种设计的一个副作用是，我们总是通过值传递 Op 派生类，而不是通过引用或指针传递（通过值传递是 MLIR 中的常见习惯用法，类似地适用于属性、类型等）。
给定一个通用的 Operation* 实例，我们可以使用 LLVM 的转换基础设施始终获取一个特定的 Op 实例：

``` 
void processConstantOp(mlir::Operation *operation) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);

  // This operation is not an instance of `ConstantOp`.
  if (!op)
    return;

  // Get the internal operation instance wrapped by the smart pointer.
  mlir::Operation *internalOperation = op.getOperation();
  assert(internalOperation == operation &&
         "these operation instances are the same");
}
```

**Using the Operation Definition Specification (ODS) Framework**

除了对 mlir::Op C++ 模板进行特化外，MLIR 还支持以声明性的方式定义操作。这是通过操作定义规范框架实现的。
关于操作的事实被简洁地指定为一个 TableGen 记录，它将在编译时扩展为等效的 mlir::Op C++ 模板特化。
在面对 C++ API 更改时，使用 ODS 框架是定义 MLIR 操作的理想方式，因为它简单、简洁，并且在 C++ API 更改时通常更加稳定。

让我们看看如何定义我们的 ConstantOp 的 ODS 等效版本：

在 ODS 中，操作通过继承 Op 类来定义。为了简化我们的操作定义，我们将为 Toy 方言中的操作定义一个基类。

``` 
// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

在定义常量操作之前，我们已经定义了所有的前提部分。

我们通过继承上面定义的基类 'Toy_Op' 来定义一个 Toy 操作。在这里，我们提供了操作的助记符和一系列 traits。
助记符在这里与 ConstantOp::getOperationName 中给出的助记符相匹配，但没有方言前缀 "toy."。
在这里我们还没有包含在 C++ 定义中的 ZeroOperands 和 OneResult traits，它们将根据我们稍后定义的参数和结果字段自动推断。

``` 
def ConstantOp : Toy_Op<"constant"> {
}
```

此时，您可能想知道由TableGen生成的C++代码是什么样的。只需运行mlir-tblgen命令，使用gen-op-decls或gen-op-defs操作，如下所示：

``` 
${build_root}/bin/mlir-tblgen -gen-op-defs 
${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```

根据选择的操作，这将打印出ConstantOp类的声明或实现。将此输出与手工编写的实现进行比较，在使用TableGen时非常有帮助。

**Defining Arguments and Results**

在我们定义操作的基本结构之后，现在我们可以为操作提供输入和输出。
操作的输入，也就是参数，可以是属性或用于SSA操作数值的类型。结果对应于操作生成的一组值的类型。

``` 
def ConstantOp : Toy_Op<"constant"> {
  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```

通过为参数或结果提供名称，例如$value，ODS将自动生成相应的访问器：DenseElementsAttr ConstantOp::value()。

**Adding Documentation**

在定义操作之后的下一步是对其进行文档化。操作可以提供摘要和描述字段来描述操作的语义。
这些信息对于方言的用户非常有用，甚至可以用于自动生成 Markdown 文档。

``` 
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```


**Verifying Operation Semantics**


到目前为止，我们已经涵盖了大部分原始的 C++ 操作定义。下一个要定义的部分是验证器。
幸运的是，就像命名访问器一样，ODS 框架将根据我们给定的约束自动生成许多必要的验证逻辑。
这意味着我们不需要验证返回类型的结构，甚至不需要验证输入属性的值。
在许多情况下，对于 ODS 操作，额外的验证甚至都不是必需的。要添加额外的验证逻辑，操作可以重写验证器字段。
验证器字段允许定义一个 C++ 代码块，该代码块将作为 ConstantOp::verify 的一部分运行。
此代码块可以假设操作的所有其他不变性已经通过验证：

``` 
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);

  // Add additional verification logic to the constant operation. Setting this bit
  // to `1` will generate a `::mlir::LogicalResult verify()` declaration on the
  // operation class that is called after ODS constructs have been verified, for
  // example the types of arguments and results. We implement additional verification
  // in the definition of this `verify` method in the C++ source file. 
  let hasVerifier = 1;
}
```

**Attaching build Methods**

原始的 C++ 示例中还缺少的最后一个组成部分是构建方法。ODS 可以自动生成一些简单的构建方法，在这种情况下，它将为我们生成第一个构建方法。
对于其余的构建方法，我们定义了 builders 字段。
该字段接受一个 OpBuilder 对象的列表，每个对象都接受一个与 C++ 参数列表相对应的字符串，以及一个可选的代码块，用于内联指定实现逻辑。

``` 
def ConstantOp : Toy_Op<"constant"> {
  ...

  // Add custom build methods for the constant operation. These methods populate
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      // Call into an autogenerated `build` method.
      build(builder, result, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value. This builder
    // creates a declaration for `ConstantOp::build` with the given parameters.
    OpBuilder<(ins "double":$value)>
  ];
}
```

**Specifying a Custom Assembly Format**

在这一点上，我们可以生成我们的 "toy IR"。例如，下面的内容：

``` 
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

以下为IR的结果：

``` 
module {
  "toy.func"() ({
  ^bb0(%arg0: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1)):
    %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    "toy.return"(%2) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  }) {sym_name = "multiply_transpose", type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>} : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  "toy.func"() ({
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = "toy.generic_call"(%1, %3) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = "toy.generic_call"(%3, %1) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    "toy.print"(%5) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    "toy.return"() : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  }) {sym_name = "main", type = () -> ()} : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

这里需要注意的一点是，我们的所有 Toy 操作都使用通用汇编格式进行打印。
这种格式是在本章开头对 toy.transpose 进行拆解时展示的格式。
MLIR 允许操作以声明方式或通过 C++ 命令式方式定义自定义汇编格式。
定义自定义汇编格式可以将生成的 IR 调整为更可读的形式，通过去除通用格式所需的许多冗余内容。
让我们通过一个希望简化的操作格式示例来详细了解。

**toy.print**

toy.print 当前的格式有点冗长。有许多额外的字符我们想要去掉。让我们首先思考一下 toy.print 的好的格式应该是什么样的，然后看看我们如何实现它。
通过查看 toy.print 的基本形式，我们可以得到以下信息:
```
toy.print %5 : tensor<*xf64> loc(...)
```

在这里，我们将格式简化为最基本的要素，它变得更加可读。
要提供自定义汇编格式，操作可以通过覆盖 hasCustomAssemblyFormat 字段来使用 C++ 格式，或者通过覆盖 assemblyFormat 字段来使用声明性格式。
让我们先看看 C++ 变体，因为这是声明性格式在内部映射的方式。

``` 
/// Consider a stripped definition of `toy.print` here.
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);

  // Divert the printer and parser to `parse` and `print` methods on our operation,
  // to be implemented in the .cpp file. More details on these methods is shown below.
  let hasCustomAssemblyFormat = 1;
}
```

下面是打印器（printer）和解析器（parser）的 C++ 实现示例：

``` 
/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void PrintOp::print(mlir::OpAsmPrinter &printer) {
  printer << "toy.print " << op.input();
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.input().getType();
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand inputOperand;
  mlir::Type inputType;
  if (parser.parseOperand(inputOperand) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(inputType))
    return mlir::failure();

  // Resolve the input operand to the type we parsed in.
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return mlir::failure();

  return mlir::success();
}

```

在将 C++ 实现映射到声明性格式时，主要涉及三个不同的组件：

- 指令（Directives）
  - 这是一种内置函数，带有一组可选参数。
- 文字（Literals）
  - 由 `` 包围的关键字或标点符号。
- 变量（Variables） 
  - 已在操作本身上注册的实体，例如参数（属性或操作数）、结果、后继等。在上面的 PrintOp 示例中，一个变量是 $input。

我们对 C++ 格式进行直接映射的声明性格式如下所示：

```
/// Consider a stripped definition of `toy.print` here.
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);

  // In the following format we have two directives, `attr-dict` and `type`.
  // These correspond to the attribute dictionary and the type of a given
  // variable represectively.
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

在实现自定义格式之前，请确保先了解声明性格式的其他有趣特性。在对一些操作的格式进行美化之后，我们得到了更加可读的格式：

``` 
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    toy.return %2 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    toy.print %5 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

在上面，我们介绍了在ODS框架中定义操作的一些概念，但还有许多其他的概念我们还没有介绍，比如区域（regions）、可变操作数（variadic operands）等等。请查看完整的规范以获取更多详细信息。

**Complete Toy Example**

现在我们可以生成我们的"Toy IR"。
您可以构建toyc-ch2并尝试在上面的示例中运行：toyc-ch2 test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo。
我们还可以检查我们的RoundTrip：toyc-ch2 test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo 2> codegen.mlir，然后执行toyc-ch2 codegen.mlir -emit=mlir。
您还可以对最终的定义文件使用mlir-tblgen命令，并研究生成的C++代码。

到目前为止，MLIR已经了解我们的Toy方言和操作。在下一章中，我们将利用我们的新方言为Toy语言实现一些高级语言特定的分析和转换。