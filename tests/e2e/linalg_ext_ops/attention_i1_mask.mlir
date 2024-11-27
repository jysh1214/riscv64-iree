func.func @i1_representation() {
  %mask = util.unfoldable_constant dense<[140]> : tensor<1xi8>
  %casted = flow.tensor.bitcast %mask : tensor<1xi8> -> tensor<2x4xi1>
  %bar = util.optimization_barrier %casted : tensor<2x4xi1>
  %tensor_res = flow.tensor.bitcast %bar : tensor<2x4xi1> -> tensor<1xi8>
  check.expect_eq_const(%tensor_res, dense<[140]> : tensor<1xi8>) : tensor<1xi8>
  return
}

func.func @i1_representation_2() {
  %mask = util.unfoldable_constant dense<[140, 77]> : tensor<2xi8>
  %casted = flow.tensor.bitcast %mask : tensor<2xi8> -> tensor<2x8xi1>
  %bar = util.optimization_barrier %casted : tensor<2x8xi1>
  %tensor_res = flow.tensor.bitcast %bar : tensor<2x8xi1> -> tensor<2xi8>
  check.expect_eq_const(%tensor_res, dense<[140, 77]> : tensor<2xi8>) : tensor<2xi8>
  return
}

func.func @i1_representation_3() {
  %mask = util.unfoldable_constant dense<[140, 77]> : tensor<2xi8>
  %casted = flow.tensor.bitcast %mask : tensor<2xi8> -> tensor<4x4xi1>
  %bar = util.optimization_barrier %casted : tensor<4x4xi1>
  %tensor_res = flow.tensor.bitcast %bar : tensor<4x4xi1> -> tensor<2xi8>
  check.expect_eq_const(%tensor_res, dense<[140, 77]> : tensor<2xi8>) : tensor<2xi8>
  return
}

func.func @truncate_i1() {
  %mask = util.unfoldable_constant dense<[1, 1, 0, 0,
                                          0, 0, 1, 1]> : tensor<8xi8>
  %nm = tensor.empty() : tensor<8xi1>
  %truncm = linalg.generic
  {indexing_maps = [
    affine_map<(d0) -> (d0)>,
    affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]}
  ins(%mask: tensor<8xi8>)
  outs(%nm: tensor<8xi1>) {
    ^bb0(%in: i8, %out: i1):
      %zero = arith.constant 0 : i8
      %truncated = arith.cmpi "sgt", %in, %zero : i8
      linalg.yield %truncated : i1
  } -> tensor<8xi1>
  %tensor_res = flow.tensor.bitcast %truncm : tensor<8xi1> -> tensor<1xi8>
  check.expect_eq_const(%tensor_res, dense<[195]> : tensor<1xi8>) : tensor<1xi8>
  return
}

func.func @truncate_i1_2() {
  %mask = util.unfoldable_constant dense<[[0, 0, 1, 1],
                                          [1, 1, 0, 0],
                                          [1, 1, 0, 0],
                                          [0, 0, 1, 1]]> : tensor<4x4xi8>
  %nm = tensor.empty() : tensor<4x4xi1>
  %truncm = linalg.generic
  {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]}
  ins(%mask: tensor<4x4xi8>)
  outs(%nm: tensor<4x4xi1>) {
    ^bb0(%in: i8, %out: i1):
      %zero = arith.constant 0 : i8
      %truncated = arith.cmpi "sgt", %in, %zero : i8
      linalg.yield %truncated : i1
  } -> tensor<4x4xi1>
  %tensor_res = flow.tensor.bitcast %truncm : tensor<4x4xi1> -> tensor<2xi8>
  check.expect_eq_const(%tensor_res, dense<[60, 195]> : tensor<2xi8>) : tensor<2xi8>
  return
}

func.func @attention1x4x4_i1_all_ones() {
  %init = tensor.empty() : tensor<1x4x4xf32>
  %query = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2],
                                            [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>

  %key = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                          [0.5, 0.6, 0.7, 0.8],
                                          [0.9, 1.0, 1.1, 1.2],
                                          [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>
  %value = util.unfoldable_constant dense<[[[0.1, 0.2, 0.3, 0.4],
                                            [0.5, 0.6, 0.7, 0.8],
                                            [0.9, 1.0, 1.1, 1.2],
                                            [1.3, 1.4, 1.5, 1.6]]]> : tensor<1x4x4xf32>

  %i8mask = util.unfoldable_constant dense<[165, 165]> : tensor<2xi8>
  %mask = flow.tensor.bitcast %i8mask : tensor<2xi8> -> tensor<1x4x4xi1>

  %scale = arith.constant 0.5 : f32
  %1 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale, %mask : tensor<1x4x4xf32>,
        tensor<1x4x4xf32>, tensor<1x4x4xf32>, f32, tensor<1x4x4xi1>) outs(%init : tensor<1x4x4xf32>) {
          ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<1x4x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<[[[0.57895, 0.67895, 0.77895, 0.87895],
              [1.09108, 1.19108, 1.29108, 1.39108],
              [0.774324, 0.874324, 0.974324, 1.07432],
              [1.22842, 1.32842, 1.42842, 1.52842]]]> : tensor<1x4x4xf32>
  ) : tensor<1x4x4xf32>
  return
}
