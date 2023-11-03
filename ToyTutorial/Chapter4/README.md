# Enabling Generic Transformation with Interfaces

## Background: Grappling with an Extensible IR

通过方言，MLIR允许表示许多不同级别的抽象；我们之前定义的Toy方言就是一个例子。
尽管这些不同的方言可能表示不同的抽象，但通常有一组共同的转换和分析操作需要执行。
问题是，如果为每个方言单独实现每个转换，就会导致大量的代码重复，因为内部算法通常非常相似，甚至完全相同。
我们希望能够提供转换的能力，使其能够透明地连接到像Toy这样的方言，以获取所需的信息。

MLIR提供了一组始终可用的钩子来支持某些核心转换，正如前一章中所示，我们通过操作的钩子（getCanonicalizationPatterns）注册了一些规范化操作。
然而，这些类型的钩子并不具有良好的可扩展性。
因此，设计了一种更通用的解决方案，即接口，使得MLIR基础设施可以像表示一样具有可扩展性。接口为方言和操作提供了一种通用机制，用于向转换或分析提供信息。


## Shape Inference: Preparing for Code Generation

我们的Toy IR当前在通用张量上运行，这意味着我们除了在常量初始化期间外，不知道张量的形状。
这给优化和代码生成带来了复杂性。幸运的是，我们可以通过计算来传播形状，直到所有形状都是已知的。
问题是如何处理对用户定义的通用函数的调用：每个调用点可能推导出不同的形状。
一种可能的方法是基于参数类型进行符号推断，但如果我们在语言中引入更多的控制流程，这将很难泛化。
另一种方法是函数特化，其中每个具有新参数形状的调用点都会复制被调用的函数并对其进行特化。
我们在Toy中采用的方法是内联所有的函数调用，然后进行函数内部的形状传播。

### Inlining

在这里，我们可以编写一个专门针对Toy方言的内联算法，但根据所需的复杂性级别，这可能变得非常复杂。
忽略成本建模，纯结构转换已经很难从头开始实现。
幸运的是，MLIR提供了一个通用的内联算法，方言可以插入其中。在Toy中，我们只需要提供用于内联器的接口。

首先，我们需要定义在Toy方言中内联操作的约束条件。
这些信息通过方言接口提供。这本质上是一个包含一组虚拟钩子的类，方言可以进行重写。
在这种情况下，接口是DialectInlinerInterface。

``` 
/// This class defines the interface for handling inlining with Toy operations.
/// We simplify inherit from the base interface class and override
/// the necessary methods.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// This hook checks to see if the given callable operation is legal to inline
  /// into the given call. For Toy this hook can simply return true, as the Toy
  /// Call operation is always inlinable.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// This hook checks to see if the given operation is legal to inline into the
  /// given region. For Toy this hook can simply return true, as all Toy
  /// operations are inlinable.
  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }

  /// This hook cheks if the given 'src' region can be inlined into the 'dest'
  /// region. The regions here are the bodies of the callable functions. For
  /// Toy, any function can be inlined, so we simply return true.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  /// This hook is called when a terminator operation has been inlined. The only
  /// terminator that we have in the Toy dialect is the return
  /// operation(toy.return). We handle the return by replacing the values
  /// previously returned by the call operation with the operands of the
  /// return.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
```

此外，inliner将只丢弃私下可见的未使用的函数定义。我们还必须在MLIR生成器中设置函数的可见性（除了主函数）。

``` 
/// Emit a new function and add it to the MLIR module.
mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
  ...
  // If this function isn't main, then set the visibility to private.
  if (funcAST.getProto()->getName() != "main")
    function.setPrivate();

  return function;
}
```

然后我们直接在Toy方言上注册我们的方言接口，就像我们对操作所做的一样。

``` 
void ToyDialect::initialize() {
  addInterfaces<ToyInlinerInterface>();
}
```

接下来，我们需要提供一种方式让内联器知道toy.generic_call表示一个调用，而toy.func表示一个函数。MLIR提供了操作接口，可以用于将操作标记为“调用类”或“可调用类”。
与方言接口不同，操作接口提供了更精细的信息粒度，这些信息是特定于单个操作的核心信息。我们将在这里添加的接口是CallOpInterface和CallableOpInterface。

要添加此接口，我们只需将其定义包含在操作规范文件（Ops.td）中即可：

``` 
def FuncOp : Toy_Op<"func",
    [DeclareOpInterfaceMethods<CallableOpInterface>]> {
  ...
}

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...
}
```

在上面的示例中，我们还使用DeclareOpInterfaceMethods指令在GenericCallOp的类声明中自动声明所有接口方法。这意味着我们只需要提供一个定义即可：

``` 
/// Returns the region on the function operation that is callable.
Region *FuncOp::getCallableRegion() { return &getBody(); }

/// Returns the results types that the callable region produces when
/// executed.
ArrayRef<Type> FuncOp::getCallableResults() { return getType().getResults(); }

/// Returns the argument attributes for all callable region arguments or
/// null if there are none.
ArrayAttr FuncOp::getCallableArgAttrs() {
  return getArgAttrs().value_or(nullptr);
}

/// Returns the result attributes for all callable region results or
/// null if there are none.
ArrayAttr FuncOp::getCallableResAttrs() {
  return getResAttrs().value_or(nullptr);
}

// ....

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return inputs(); }
```

既然内联器已经了解了Toy方言，我们可以将内联器传递给Toy的Pass Manager中：

``` 
pm.addPass(mlir::createInlinerPass());
```

现在让我们来看一个工作示例：

``` 
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
  %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
  %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %5 : tensor<*xf64>
  toy.return
}
```

我们有两个对multiply_transpose的调用，我们希望将它们内联到main函数中，但是如果我们查看输出，什么都没有改变。
我们还缺少最后一个微妙的部分：在调用的边缘存在隐藏的类型转换。如果我们查看上面的代码，generic_call的操作数的类型是tensor<2x3xf64>，而函数的输入参数期望的是tensor<*xf64>。
为了解决这种差异，内联器期望插入一个显式的类型转换操作。
为此，我们需要在Toy方言中添加一个新的操作ToyCastOp（toy.cast），用于表示两个不同形状之间的转换。

``` 
def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    Pure,
    SameOperandsAndResultShape]
  > {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked,
    then shape is required to match. The operation is invalid if converting
    to a mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}
```

请注意，这个类型转换操作的定义将CastOpInterface添加到了traits列表中。
该接口提供了一些用于类型转换操作的实用工具，比如折叠身份转换和验证。我们通过为areCastCompatible方法提供定义来连接到该接口：

``` 
/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // The inputs must be Tensors with the same element type.
  TensorType input = inputs.front().dyn_cast<TensorType>();
  TensorType output = outputs.front().dyn_cast<TensorType>();
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // The shape is required to match if both types are ranked.
  return !input.hasRank() || !output.hasRank() || input == output;
}
```

有了适当的转换操作，现在我们可以在ToyInlinerInterface上重写必要的钩子，以在必要时为我们插入转换操作：

``` 
struct ToyInlinerInterface : public DialectInlinerInterface {
  ...

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};
```

如果我们再次将工作示例运行通过管道，我们将得到预期的结果：

``` 
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>
  %3 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
  %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
  %5 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
  %6 = toy.mul %4, %5 : tensor<*xf64>
  toy.print %6 : tensor<*xf64>
  toy.return
}
```

注意：通用内联器也会进行简化，所以输出可能比预期的要干净一些。

### Intraprocedural Shape Inference

现在，我们已经内联了所有的函数，只剩下一个包含静态和动态形状操作的主函数。现在，我们可以编写一个简单的形状推导 pass，以在函数内部传播形状（intraprocedurally）。
我们可以直接将其作为一个 pass 来编写，该 pass 直接编码了 Toy 方言中操作的约束，但这似乎是一个可以以通用方式编写的转换。
根据经验，最好将转换尽可能地表达得通用，以便将来可以将其扩展到其他方言。无法预测有多少其他方言可能具有类似的需求或遇到相同的问题。

