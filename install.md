# 


[官网](https://mlir.llvm.org/getting_started/)
安装命令行

```shell
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
cmake -G "Ninja" ../llvm \
     -DLLVM_ENABLE_PROJECTS="mlir;lld" \     
     -DLLVM_BUILD_EXAMPLES=ON \
     -DLLVM_TARGETS_TO_BUILD="host" \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=ON
     
cmake --build . --target check-mlir   


bin/toyc-ch2 ../mlir/test/Examples/Toy/Ch2/codegen.toy -emit=ast

```

输出如下

```
  Module:
    Function 
      Proto 'multiply_transpose' @../mlir/test/Examples/Toy/Ch2/codegen.toy:4:1
      Params: [a, b]
      Block {
        Return
          BinOp: * @../mlir/test/Examples/Toy/Ch2/codegen.toy:5:25
            Call 'transpose' [ @../mlir/test/Examples/Toy/Ch2/codegen.toy:5:10
              var: a @../mlir/test/Examples/Toy/Ch2/codegen.toy:5:20
            ]
            Call 'transpose' [ @../mlir/test/Examples/Toy/Ch2/codegen.toy:5:25
              var: b @../mlir/test/Examples/Toy/Ch2/codegen.toy:5:35
            ]
      } // Block
    Function 
      Proto 'main' @../mlir/test/Examples/Toy/Ch2/codegen.toy:8:1
      Params: []
      Block {
        VarDecl a<2, 3> @../mlir/test/Examples/Toy/Ch2/codegen.toy:9:3
          Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @../mlir/test/Examples/Toy/Ch2/codegen.toy:9:17
        VarDecl b<2, 3> @../mlir/test/Examples/Toy/Ch2/codegen.toy:10:3
          Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @../mlir/test/Examples/Toy/Ch2/codegen.toy:10:17
        VarDecl c<> @../mlir/test/Examples/Toy/Ch2/codegen.toy:11:3
          Call 'multiply_transpose' [ @../mlir/test/Examples/Toy/Ch2/codegen.toy:11:11
            var: a @../mlir/test/Examples/Toy/Ch2/codegen.toy:11:30
            var: b @../mlir/test/Examples/Toy/Ch2/codegen.toy:11:33
          ]
        VarDecl d<> @../mlir/test/Examples/Toy/Ch2/codegen.toy:12:3
          Call 'multiply_transpose' [ @../mlir/test/Examples/Toy/Ch2/codegen.toy:12:11
            var: b @../mlir/test/Examples/Toy/Ch2/codegen.toy:12:30
            var: a @../mlir/test/Examples/Toy/Ch2/codegen.toy:12:33
          ]
        Print [ @../mlir/test/Examples/Toy/Ch2/codegen.toy:13:3
          var: d @../mlir/test/Examples/Toy/Ch2/codegen.toy:13:9
        ]
      } // Block

```

```shell
bin/toyc-ch2 ../mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo
```
输出如下

``` 
module  {
  func @multiply_transpose(%arg0: tensor<*xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":4:1)) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":5:25)
    toy.return %2 : tensor<*xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":4:1)
  func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":12:11)
    toy.print %5 : tensor<*xf64> loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("../mlir/test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)

```

