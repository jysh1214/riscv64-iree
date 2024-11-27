// This is literally the same as the @causal_attention1x3x4 test in `attention.mlir`,
// but i1 masks are treated as i1 tensors, as the "--iree-experimental-packed-i1-storage"
// option is turned on.
func.func @attention1x3x4_i1() {
  %init = tensor.empty() : tensor<1x3x4xf32>
  %query = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2]]]> : tensor<1x3x4xf32>

  %key = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                          [0.5, 0.6, 0.7, 0.8],
                                          [0.9, 1.0, 1.1, 1.2]]]> : tensor<1x3x4xf32>
  %value = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2]]]> : tensor<1x3x4xf32>
  %mask = util.unfoldable_constant dense<[[[true, false,  false],
                                           [true, true,   false],
                                           [true, true,   true]]]> : tensor<1x3x3xi1>
  %scale = arith.constant 0.5 : f32
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale, %mask : tensor<1x3x4xf32>,
        tensor<1x3x4xf32>, tensor<1x3x4xf32>, f32, tensor<1x3x3xi1>) outs(%init : tensor<1x3x4xf32>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<1x3x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<[[[0.1000, 0.2000, 0.3000, 0.4000],
              [0.3509, 0.4509, 0.5509, 0.6509],
              [0.7011, 0.8011, 0.9011, 1.0011]]]> : tensor<1x3x4xf32>
  ) : tensor<1x3x4xf32>
  return
}
