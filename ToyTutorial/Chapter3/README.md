# Chapter 3: High-level Language-Specific Analysis and Transformation

创建一个紧密代表输入语言语义的方言，可以在MLIR中进行分析、转换和优化，这些分析、转换和优化需要高层次的语言信息，通常在语言AST上进行。
例如，clang有一个相当重的机制来执行C++中的模板实例化。

我们把编译器的转换分为两类：局部和全局。在本章中，我们将重点讨论如何利用Toy方言及其高级语义来执行LLVM中难以实现的局部模式匹配转换。
为此，我们使用MLIR的通用DAG重写器。

有两种方法可以用来实现模式匹配的转换：1.强制性的，C++模式匹配和重写 2.声明性的、基于规则的模式匹配和重写，使用表驱动的声明性重写规则（DRR）。
注意，使用DRR需要使用ODS来定义操作，如第二章所述。


## Optimize Transpose using C++ style pattern-match and rewrite 

让我们从一个简单的模式开始，尝试消除两个转置抵消的序列：transpose(transpose(X)) -> X.这里是相应的Toy例子：

``` 
def transpose_transpose(x) {
  return transpose(transpose(x));
}
```

这对应于以下的IR：

``` 
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %1 : tensor<*xf64>
}
```


这是一个很好的示例，展示了在Toy IR上进行匹配非常简单，但对于LLVM来说却很难实现的转换。
例如，现在的Clang无法优化掉临时数组，并且使用简单的转置方式进行的计算表示如下循环：

``` 
#define N 100
#define M 100

void sink(void *);
void double_transpose(int A[N][M]) {
  int B[M][N];
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       B[j][i] = A[i][j];
    }
  }
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       A[i][j] = B[j][i];
    }
  }
  sink(A);
}
```

对于在IR中匹配树状模式并用不同的操作替换它的简单C++方法，我们可以通过实现RewritePattern来插入MLIR规范化器（Canonicalizer）传递过程：

``` 
/// Fold transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  /// 我们注册此模式以匹配 IR 中的每个 toy.transpose。框架使用“收益”对模式进行排序并按盈利能力顺序处理它们。
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  /// 此方法尝试匹配模式并重写它。重写器参数是重写序列的业务流程协调器。预计它将与它交互，以便从此处对 IR 执行任何更改。
  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.查看当前转置的输入。
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>(); 

    // Input defined by another transpose? If not, no match. 由另一个转置定义的输入？如果没有，则不匹配。
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter. 否则，我们将有一个多余的转置。使用重写器。
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});// 用输入替代掉op
    return success();
  }
};
```

这个重写器的实现可以在ToyCombine.cpp中找到。规范化传递过程以贪婪、迭代的方式应用由操作定义的转换。
为了确保规范化传递过程应用我们的新转换，我们设置hasCanonicalizer = 1，并将模式注册到规范化框架中。

``` 
// Register our patterns for rewrite by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
```

要添加优化流水线，我们需要更新toyc.cpp主文件。在MLIR中，优化通过一个类似于LLVM的PassManager来运行：

``` 
  mlir::PassManager pm(module->getName());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
```

最后，我们可以运行以下命令来观察我们的模式的效果：
``` 
toyc-ch3 test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt
```

``` 
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %arg0 : tensor<*xf64>
}
```

正如预期的那样，我们现在直接返回函数参数，跳过了任何转置操作。然而，仍然有一个转置操作没有被消除。
这不是理想的情况！发生这种情况是因为我们的模式将最后一个转换替换为函数输入，并留下了现在无用的转置输入。
规范化器知道如何清除无用的操作；然而，MLIR保守地假设操作可能具有副作用。我们可以通过为TransposeOp添加一个新的trait，**Pure**，来解决这个问题：

``` 
def TransposeOp : Toy_Op<"transpose", [Pure]> {...}
```

让我们再试一次 toyc-ch3 test/transpose_transpose.toy -emit=mlir -opt：

``` 
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  toy.return %arg0 : tensor<*xf64>
}
```

非常好！没有剩下任何转置操作——代码是最优的。
在下一节中，我们将使用DRR（Dynamic Reshape Recognition）来进行与Reshape操作相关的模式匹配优化。


## Optimize Reshapes using DRR

声明性的基于规则的模式匹配和重写（DRR）是一种基于操作DAG的声明性重写器，它提供了基于表格的语法来定义模式匹配和重写规则。

``` 
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)>;

```

类似于SimplifyRedundantTranspose，可以使用DRR更简洁地表达冗余重塑（reshape）优化，如下所示：

``` 
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

与每个DRR模式对应的自动生成的C++代码可以在
path/to/BUILD/tools/mlir/examples/toy/Ch3/ToyCombine.inc下找到。

DRR还提供了一种方法，当转换是以参数和结果的某些属性为条件时，可以添加参数约束。一个例子是，当reshape是多余的，即当输入和输出的形状是相同的时候，就会消除reshape的转换。

``` 
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

一些优化可能需要对指令参数进行额外的转换。这是通过NativeCodeCall实现的，它允许通过调用C++辅助函数或使用内联C++来实现更复杂的转换。
这种优化的一个例子是FoldConstantReshape，我们通过在原地重塑常量并消除重塑操作来优化常量值的重塑。

``` 
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```

我们用下面的trivial_reshape.toy程序演示这些重塑优化：

``` 
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
```

``` 
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
    %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
    toy.print %3 : tensor<2x1xf64>
    toy.return
  }
}
```

我们可以尝试运行以下命令，观察我们的模式是否生效：

```
toyc-ch3 test/Examples/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt
```

``` 
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

``` 
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

如预期的那样，在进行规范化之后，不再存在任何重塑操作。

有关声明式重写方法的更多详细信息，请参阅Table-driven Declarative Rewrite Rule (DRR)。

在本章中，我们了解了如何通过始终可用的钩子使用某些核心转换。在下一章中，我们将看到如何通过接口使用更具扩展性的通用解决方案。