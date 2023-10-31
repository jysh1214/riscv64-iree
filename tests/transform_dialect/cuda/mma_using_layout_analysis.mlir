func.func @matmul(%lhs : tensor<16x16xf16>, %rhs : tensor<16x8xf16>) -> tensor<16x8xf16> {
  %c0 = arith.constant 0.0 : f16
  %0 = tensor.empty() : tensor<16x8xf16>
  %1 = linalg.fill ins(%c0 : f16) outs(%0 : tensor<16x8xf16>) -> tensor<16x8xf16>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x8xf16>)
      outs(%1 : tensor<16x8xf16>) -> tensor<16x8xf16>
  return %2 : tensor<16x8xf16>
}

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-hal-cuda-llvm-target-arch=sm_80 \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-use-transform-dialect-strategy=%p/mma_using_layout_analysis_codegen_spec.mlir | \
// RUN: iree-run-module --module=- --function=matmul --device=cuda \
// RUN: --input="16x16xf16=[[1.0,1.125,1.25,1.375,1.5,1.625,1.75,1.875,2.0,2.125,2.25,2.375,2.5,2.625,2.75,2.875],[3.0,3.125,3.25,3.375,3.5,3.625,3.75,3.875,4.0,4.125,4.25,4.375,4.5,4.625,4.75,4.875],[5.0,5.125,5.25,5.375,5.5,5.625,5.75,5.875,6.0,6.125,6.25,6.375,6.5,6.625,6.75,6.875],[7.0,7.125,7.25,7.375,7.5,7.625,7.75,7.875,8.0,8.125,8.25,8.375,8.5,8.625,8.75,8.875],[9.0,9.125,9.25,9.375,9.5,9.625,9.75,9.875,10.0,10.125,10.25,10.375,10.5,10.625,10.75,10.875],[11.0,11.125,11.25,11.375,11.5,11.625,11.75,11.875,12.0,12.125,12.25,12.375,12.5,12.625,12.75,12.875],[13.0,13.125,13.25,13.375,13.5,13.625,13.75,13.875,14.0,14.125,14.25,14.375,14.5,14.625,14.75,14.875],[15.0,15.125,15.25,15.375,15.5,15.625,15.75,15.875,16.0,16.125,16.25,16.375,16.5,16.625,16.75,16.875],[17.0,17.125,17.25,17.375,17.5,17.625,17.75,17.875,18.0,18.125,18.25,18.375,18.5,18.625,18.75,18.875],[19.0,19.125,19.25,19.375,19.5,19.625,19.75,19.875,20.0,20.125,20.25,20.375,20.5,20.625,20.75,20.875],[21.0,21.125,21.25,21.375,21.5,21.625,21.75,21.875,22.0,22.125,22.25,22.375,22.5,22.625,22.75,22.875],[23.0,23.125,23.25,23.375,23.5,23.625,23.75,23.875,24.0,24.125,24.25,24.375,24.5,24.625,24.75,24.875],[25.0,25.125,25.25,25.375,25.5,25.625,25.75,25.875,26.0,26.125,26.25,26.375,26.5,26.625,26.75,26.875],[27.0,27.125,27.25,27.375,27.5,27.625,27.75,27.875,28.0,28.125,28.25,28.375,28.5,28.625,28.75,28.875],[29.0,29.125,29.25,29.375,29.5,29.625,29.75,29.875,30.0,30.125,30.25,30.375,30.5,30.625,30.75,30.875],[31.0,31.125,31.25,31.375,31.5,31.625,31.75,31.875,32.0,32.125,32.25,32.375,32.5,32.625,32.75,32.875]]" \
// RUN: --input="16x8xf16=[[1.0,1.125,1.25,1.375,1.5,1.625,1.75,1.875],[2.0,2.125,2.25,2.375,2.5,2.625,2.75,2.875],[3.0,3.125,3.25,3.375,3.5,3.625,3.75,3.875],[4.0,4.125,4.25,4.375,4.5,4.625,4.75,4.875],[5.0,5.125,5.25,5.375,5.5,5.625,5.75,5.875],[6.0,6.125,6.25,6.375,6.5,6.625,6.75,6.875],[7.0,7.125,7.25,7.375,7.5,7.625,7.75,7.875],[8.0,8.125,8.25,8.375,8.5,8.625,8.75,8.875],[9.0,9.125,9.25,9.375,9.5,9.625,9.75,9.875],[10.0,10.125,10.25,10.375,10.5,10.625,10.75,10.875],[11.0,11.125,11.25,11.375,11.5,11.625,11.75,11.875],[12.0,12.125,12.25,12.375,12.5,12.625,12.75,12.875],[13.0,13.125,13.25,13.375,13.5,13.625,13.75,13.875],[14.0,14.125,14.25,14.375,14.5,14.625,14.75,14.875],[15.0,15.125,15.25,15.375,15.5,15.625,15.75,15.875],[16.0,16.125,16.25,16.375,16.5,16.625,16.75,16.875]]" |\
// RUN: FileCheck %s --check-prefix=EXEC

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 16x8xf16=[306 310 313.75 317.5 321.5 325.5 329.25 333][578 586 594 601.5 609.5 617.5 625 633][850 862 874 885.5 897.5 909.5 921 933][1122 1138 1154 1170 1186 1201 1217 1233][1394 1414 1434 1454 1474 1493 1513 1533][1666 1690 1714 1738 1762 1785 1809 1833][1938 1966 1994 2022 2050 2078 2106 2134][2210 2242 2274 2306 2338 2370 2402 2434][2482 2518 2554 2590 2626 2662 2698 2734][2754 2794 2834 2874 2914 2954 2994 3034][3026 3070 3114 3158 3202 3246 3290 3334][3298 3346 3394 3442 3490 3538 3586 3634][3570 3622 3674 3726 3778 3830 3882 3934][3842 3898 3954 4010 4066 4120 4176 4232][4112 4172 4232 4292 4352 4412 4472 4532][4384 4448 4512 4576 4640 4704 4768 4832]