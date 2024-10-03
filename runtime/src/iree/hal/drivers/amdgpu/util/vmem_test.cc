// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/vmem.h"

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::hsa {
namespace {

struct VMemTest : public ::testing::Test {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_amdgpu_libhsa_t libhsa;
  iree_hal_amdgpu_topology_t topology;

  void SetUp() override {
    IREE_TRACE_SCOPE();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_ignore(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
    if (topology.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  void TearDown() override {
    IREE_TRACE_SCOPE();
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }
};

TEST_F(VMemTest, RingbufferLifetime) {
  IREE_TRACE_SCOPE();

  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_fine_global_memory_pool(
      &libhsa, gpu_agent, &memory_pool));

  const iree_device_size_t min_capacity = 1 * 1024 * 1024;
  iree_hal_amdgpu_vmem_ringbuffer_t ringbuffer = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_vmem_ringbuffer_initialize_with_topology(
      &libhsa, gpu_agent, memory_pool, min_capacity, &topology,
      IREE_HAL_AMDGPU_ACCESS_MODE_SHARED, &ringbuffer));

  EXPECT_GE(ringbuffer.capacity, min_capacity);
  EXPECT_EQ(ringbuffer.ring_base_ptr,
            (uint8_t*)ringbuffer.va_base_ptr + ringbuffer.capacity);
  EXPECT_TRUE(iree_device_size_has_alignment(
      (iree_device_size_t)ringbuffer.ring_base_ptr, 64));

  iree_hal_amdgpu_vmem_ringbuffer_deinitialize(&libhsa, &ringbuffer);
}

TEST_F(VMemTest, RingbufferWrap) {
  IREE_TRACE_SCOPE();

  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  hsa_amd_memory_pool_t memory_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_find_fine_global_memory_pool(
      &libhsa, gpu_agent, &memory_pool));

  const iree_device_size_t min_capacity = 1 * 1024 * 1024;
  iree_hal_amdgpu_vmem_ringbuffer_t ringbuffer = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_vmem_ringbuffer_initialize_with_topology(
      &libhsa, gpu_agent, memory_pool, min_capacity, &topology,
      IREE_HAL_AMDGPU_ACCESS_MODE_SHARED, &ringbuffer));

  // Fill entire range [0,capacity).
  iree_device_size_t capacity_u32 = ringbuffer.capacity / sizeof(uint32_t);
  uint32_t* ptr = (uint32_t*)ringbuffer.ring_base_ptr;
  for (iree_device_size_t i = 0; i < capacity_u32; ++i) {
    ptr[i] = (uint32_t)i;
  }

  // Compare some locations off the base to ensure wrapping is valid.
  EXPECT_EQ(ptr[0], ptr[capacity_u32]);
  EXPECT_EQ(ptr[-1], ptr[capacity_u32 - 1]);
  EXPECT_EQ(ptr[1], ptr[capacity_u32 + 1]);

  iree_hal_amdgpu_vmem_ringbuffer_deinitialize(&libhsa, &ringbuffer);
}

}  // namespace
}  // namespace iree::hal::hsa
