// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/bitmap.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::hsa {
namespace {

struct BitmapTest : public ::testing::Test {
  iree_allocator_t host_allocator = iree_allocator_system();
};

TEST_F(BitmapTest, XXXX) {
  // DO NOT SUBMIT bitmap tests
}

}  // namespace
}  // namespace iree::hal::hsa
