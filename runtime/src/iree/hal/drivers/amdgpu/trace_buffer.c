// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/trace_buffer.h"

#include "iree/hal/drivers/amdgpu/device/kernels.h"
#include "iree/hal/drivers/amdgpu/device/tracing.h"
#include "iree/hal/drivers/amdgpu/util/kfd.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_trace_buffer_t
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

// TODO(benvanik): maybe move somewhere common? only two uses so far in the same
// code path (queue init) so just pulling from there for now.
void iree_hal_amdgpu_kernel_dispatch(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_queue_t* queue,
    IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t*
        kernel_args,
    const uint32_t grid_size[3], IREE_AMDGPU_DEVICE_PTR void* kernarg_address,
    hsa_signal_t completion_signal);

// Asynchronously dispatches the device-side trace initializer kernel:
// `iree_hal_amdgpu_device_trace_buffer_initialize`.
static void iree_hal_amdgpu_trace_buffer_enqueue_device_initialize(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer, uint32_t signal_count,
    hsa_queue_t* control_queue,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_trace_buffer_kernargs_t*
        kernargs,
    hsa_signal_t completion_signal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, signal_count);

  // Distributed by groups of signals across all signals.
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args =
      &trace_buffer->kernels->iree_hal_amdgpu_device_trace_buffer_initialize;
  const uint32_t grid_size[3] = {signal_count, 1, 1};

  // Populate kernargs needed by the dispatch.
  // TODO(benvanik): use an HSA API for copying this? Seems to work as-is.
  kernargs->trace_buffer = trace_buffer->device_buffer;

  // Dispatch the initialization kernel asynchronously.
  iree_hal_amdgpu_kernel_dispatch(trace_buffer->libhsa, control_queue,
                                  kernel_args, grid_size, kernargs,
                                  completion_signal);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_trace_buffer_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, int kfd_fd,
    iree_string_view_t executor_name, hsa_agent_t host_agent,
    hsa_agent_t device_agent, hsa_queue_t* control_queue,
    hsa_amd_memory_pool_t memory_pool, iree_device_size_t ringbuffer_capacity,
    const iree_hal_amdgpu_device_library_t* device_library,
    const iree_hal_amdgpu_device_kernels_t* kernels,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_trace_buffer_kernargs_t*
        kernargs,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_trace_buffer_t* out_trace_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_trace_buffer, 0, sizeof(*out_trace_buffer));
  out_trace_buffer->libhsa = libhsa;
  out_trace_buffer->kfd_fd = kfd_fd;
  out_trace_buffer->device_agent = device_agent;
  out_trace_buffer->kernels = kernels;

  // Ringbuffer must be a power-of-two for the indexing tricks we use.
  if (!iree_device_size_is_power_of_two(ringbuffer_capacity)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "trace ringbuffer capacity must be a power-of-two "
                             "(provided %" PRIdsz ")"));
  }

  // Query the UID of the device so that we can communicate with the kfd.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_agent_get_info(libhsa, device_agent,
                              (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_UID,
                              &out_trace_buffer->device_driver_uid));

  // Query device library code ranges used to translate embedded data from
  // pointers on the device to pointers on the host.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_device_library_populate_agent_code_range(
              device_library, device_agent,
              &out_trace_buffer->device_library_code_range));

  // Allocate ringbuffer. Only the device will produce and only the host will
  // consume so we explicitly restrict access.
  hsa_amd_memory_access_desc_t access_descs[2] = {
      (hsa_amd_memory_access_desc_t){
          .agent_handle = device_agent,
          .permissions = HSA_ACCESS_PERMISSION_RW,
      },
      (hsa_amd_memory_access_desc_t){
          .agent_handle = host_agent,
          .permissions = HSA_ACCESS_PERMISSION_RO,
      },
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_vmem_ringbuffer_initialize(
              libhsa, device_agent, memory_pool, ringbuffer_capacity,
              IREE_ARRAYSIZE(access_descs), access_descs,
              &out_trace_buffer->ringbuffer));

  // Allocate device-side trace buffer instance.
  iree_hal_amdgpu_device_trace_buffer_t* device_buffer = NULL;
  iree_status_t status = iree_hsa_amd_memory_pool_allocate(
      libhsa, memory_pool, sizeof(*device_buffer),
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&device_buffer);
  out_trace_buffer->device_buffer = device_buffer;
  if (iree_status_is_ok(status)) {
    hsa_agent_t access_agents[2] = {
        host_agent,
        device_agent,
    };
    status =
        iree_hsa_amd_agents_allow_access(libhsa, IREE_ARRAYSIZE(access_agents),
                                         access_agents, NULL, device_buffer);
  }

  // Link back the device to the host trace buffer.
  // Used when the device posts messages to the host in order to route back to
  // this particular instance.
  device_buffer->host_trace_buffer = (uint64_t)out_trace_buffer;

  device_buffer->ringbuffer_base = out_trace_buffer->ringbuffer.ring_base_ptr;
  device_buffer->ringbuffer_capacity = out_trace_buffer->ringbuffer.capacity;

  // Perform initial calibration so we can setup the context.
  // This may change over time due to clock drift.
  uint64_t cpu_timestamp = 0;
  uint64_t gpu_timestamp = 0;
  float timestamp_period = 1.0f;  // maybe?
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_clock_counters_t clock_counters = {0};
    status = iree_hal_amdgpu_kfd_get_clock_counters(
        out_trace_buffer->kfd_fd, out_trace_buffer->device_driver_uid,
        &clock_counters);
    // DO NOT SUBMIT tracy timestamp calibration
    // derive tracy-compatible timestamps
    // store initial counters in the trace buffer as a timebase
  }

  // Allocate a tracing context.
  // This is a limited resource today in Tracy (255 max) and if we ever need
  // more we'll have to fix that limit (painfully).
  if (iree_status_is_ok(status)) {
    device_buffer->executor_id =
        (iree_hal_amdgpu_trace_executor_id_t)iree_tracing_gpu_context_allocate(
            IREE_TRACING_GPU_CONTEXT_TYPE_OPENCL, executor_name.data,
            executor_name.size,
            /*is_calibrated=*/true, cpu_timestamp, gpu_timestamp,
            timestamp_period);
  }

  // Asynchronously issue device-side initialization for the ringbuffer.
  //
  // NOTE: this must happen after all other initialization is complete so that
  // the device has all information available for use.
  //
  // If any part of initialization fails the completion signal must be waited
  // on to ensure the device is not still using the trace buffer resources.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_trace_buffer_enqueue_device_initialize(
        out_trace_buffer, IREE_HAL_AMDGPU_DEVICE_QUERY_RINGBUFFER_CAPACITY,
        control_queue, kernargs, initialization_signal);
  } else {
    iree_hal_amdgpu_trace_buffer_deinitialize(out_trace_buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_trace_buffer_deinitialize(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer) {
  if (!trace_buffer->libhsa) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Flush any remaining information in the buffer to complete the trace.
  // This should only be called when all device-side emitters have terminated
  // and we have exclusive access to the buffer.
  if (trace_buffer->ringbuffer.ring_base_ptr) {
    IREE_IGNORE_ERROR(iree_hal_amdgpu_trace_buffer_flush(trace_buffer));
  }

  if (trace_buffer->device_buffer) {
    IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(
        trace_buffer->libhsa, trace_buffer->device_buffer));
  }

  iree_hal_amdgpu_vmem_ringbuffer_deinitialize(trace_buffer->libhsa,
                                               &trace_buffer->ringbuffer);

  IREE_TRACE_ZONE_END(z0);
}

static void* iree_hal_amdgpu_trace_buffer_translate_ptr(
    const iree_hal_amdgpu_trace_buffer_t* trace_buffer, uint64_t device_ptr) {
  const uint64_t device_code_ptr =
      trace_buffer->device_library_code_range.device_ptr;
  if (device_ptr >= device_code_ptr &&
      device_ptr <
          device_code_ptr + trace_buffer->device_library_code_range.size) {
    // Address is in the device code range so translate it to the host copy of
    // the code.
    return (void*)((uint64_t)device_ptr - device_code_ptr +
                   trace_buffer->device_library_code_range.host_ptr);
  } else {
    // Address is outside of the device code range and likely a valid host
    // pointer or some other virtual address the host can access.
    return (void*)device_ptr;
  }
}

static const iree_hal_amdgpu_trace_src_loc_t*
iree_hal_amdgpu_trace_buffer_translate_src_loc(
    const iree_hal_amdgpu_trace_buffer_t* trace_buffer,
    iree_hal_amdgpu_trace_src_loc_ptr_t device_ptr) {
  return (const iree_hal_amdgpu_trace_src_loc_t*)
      iree_hal_amdgpu_trace_buffer_translate_ptr(trace_buffer, device_ptr);
}

static const char* iree_hal_amdgpu_trace_buffer_translate_literal(
    const iree_hal_amdgpu_trace_buffer_t* trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t device_ptr) {
  return (const char*)iree_hal_amdgpu_trace_buffer_translate_ptr(trace_buffer,
                                                                 device_ptr);
}

iree_status_t iree_hal_amdgpu_trace_buffer_flush(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT tracy flush timestamp rebasing
  // can use to adjust like GpuAgent::TranslateTime
  // https://sourcegraph.com/github.com/ROCm/ROCR-Runtime@909b82d4632b86dff0faadcb19488a43d2108686/-/blob/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp?L2048
  iree_hal_amdgpu_clock_counters_t clock_counters = {0};
  iree_status_t status = iree_hal_amdgpu_kfd_get_clock_counters(
      trace_buffer->kfd_fd, trace_buffer->device_driver_uid, &clock_counters);

  // Consume all packets from the last flush to the new write offset.
  const uint64_t write_commit_offset =
      iree_atomic_load(&trace_buffer->device_buffer->write_commit_offset,
                       iree_memory_order_acquire);
  const uint64_t read_reserve_offset =
      iree_atomic_exchange(&trace_buffer->last_read_offset, write_commit_offset,
                           iree_memory_order_relaxed);
  const uint8_t* ring_base_ptr = trace_buffer->ringbuffer.ring_base_ptr;
  const uint64_t ring_mask = trace_buffer->ringbuffer.capacity - 1;
  for (uint64_t offset = read_reserve_offset; offset < write_commit_offset;) {
    const uint8_t* packet_ptr = ring_base_ptr + (offset & ring_mask);
    const iree_hal_amdgpu_trace_event_type_t event_type = packet_ptr[0];
    switch (event_type) {
      case IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_BEGIN: {
        const iree_hal_amdgpu_trace_zone_begin_t* event =
            (const iree_hal_amdgpu_trace_zone_begin_t*)packet_ptr;
        const iree_hal_amdgpu_trace_src_loc_t* src_loc =
            iree_hal_amdgpu_trace_buffer_translate_src_loc(trace_buffer,
                                                           event->src_loc);
        fprintf(stderr, "IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_BEGIN %s %s %s %d\n",
                src_loc->name, src_loc->function, src_loc->file, src_loc->line);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_END: {
        const iree_hal_amdgpu_trace_zone_end_t* event =
            (const iree_hal_amdgpu_trace_zone_end_t*)packet_ptr;
        fprintf(stderr, "IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_END\n");
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_I64: {
        const iree_hal_amdgpu_trace_zone_value_i64_t* event =
            (const iree_hal_amdgpu_trace_zone_value_i64_t*)packet_ptr;
        fprintf(stderr,
                "IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_I64 %" PRIi64
                " %016" PRIx64 "\n",
                event->value, event->value);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_LITERAL: {
        const iree_hal_amdgpu_trace_zone_value_text_literal_t* event =
            (const iree_hal_amdgpu_trace_zone_value_text_literal_t*)packet_ptr;
        const char* value = iree_hal_amdgpu_trace_buffer_translate_literal(
            trace_buffer, event->value);
        fprintf(stderr,
                "IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_LITERAL `%s`\n",
                value);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_DYNAMIC: {
        const iree_hal_amdgpu_trace_zone_value_text_dynamic_t* event =
            (const iree_hal_amdgpu_trace_zone_value_text_dynamic_t*)packet_ptr;
        fprintf(stderr,
                "IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_DYNAMIC `%.*s`\n",
                (int)event->length, event->value);
        offset += sizeof(*event) + event->length;
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_BEGIN: {
        const iree_hal_amdgpu_trace_execution_zone_begin_t* event =
            (const iree_hal_amdgpu_trace_execution_zone_begin_t*)packet_ptr;
        const iree_hal_amdgpu_trace_src_loc_t* src_loc =
            iree_hal_amdgpu_trace_buffer_translate_src_loc(trace_buffer,
                                                           event->src_loc);
        fprintf(stderr,
                "IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_BEGIN [%02" PRId32
                "] q=%u %s %s %s %d\n",
                event->execution_query_id, event->execution_query_id,
                src_loc->name, src_loc->function, src_loc->file, src_loc->line);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_END: {
        const iree_hal_amdgpu_trace_execution_zone_end_t* event =
            (const iree_hal_amdgpu_trace_execution_zone_end_t*)packet_ptr;
        fprintf(stderr,
                "IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_END [%02" PRId32
                "] q=%u\n",
                event->execution_query_id, event->execution_query_id);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_DISPATCH: {
        const iree_hal_amdgpu_trace_execution_zone_dispatch_t* event =
            (const iree_hal_amdgpu_trace_execution_zone_dispatch_t*)packet_ptr;
        fprintf(stderr,
                "IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_DISPATCH\n");
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_NOTIFY: {
        const iree_hal_amdgpu_trace_execution_zone_notify_t* event =
            (const iree_hal_amdgpu_trace_execution_zone_notify_t*)packet_ptr;
        fprintf(stderr, "IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_NOTIFY\n");
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_NOTIFY_BATCH: {
        const iree_hal_amdgpu_trace_execution_zone_notify_batch_t* event =
            (const iree_hal_amdgpu_trace_execution_zone_notify_batch_t*)
                packet_ptr;
        fprintf(stderr,
                "IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_NOTIFY_BATCH\n");
        offset += sizeof(*event) + event->execution_query_count *
                                       sizeof(event->execution_timestamps[0]);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_ALLOC: {
        const iree_hal_amdgpu_trace_memory_alloc_t* event =
            (const iree_hal_amdgpu_trace_memory_alloc_t*)packet_ptr;
        fprintf(stderr, "IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_ALLOC\n");
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_FREE: {
        const iree_hal_amdgpu_trace_memory_free_t* event =
            (const iree_hal_amdgpu_trace_memory_free_t*)packet_ptr;
        fprintf(stderr, "IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_FREE\n");
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_LITERAL: {
        const iree_hal_amdgpu_trace_message_literal_t* event =
            (const iree_hal_amdgpu_trace_message_literal_t*)packet_ptr;
        const char* value = iree_hal_amdgpu_trace_buffer_translate_literal(
            trace_buffer, event->value);
        fprintf(stderr, "IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_LITERAL `%s`\n",
                value);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_DYNAMIC: {
        const iree_hal_amdgpu_trace_message_dynamic_t* event =
            (const iree_hal_amdgpu_trace_message_dynamic_t*)packet_ptr;
        fprintf(stderr, "IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_DYNAMIC `%.*s`\n",
                (int)event->length, event->value);
        offset += sizeof(*event) + event->length;
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_CONFIG: {
        const iree_hal_amdgpu_trace_plot_config_t* event =
            (const iree_hal_amdgpu_trace_plot_config_t*)packet_ptr;
        const char* name = iree_hal_amdgpu_trace_buffer_translate_literal(
            trace_buffer, event->name);
        fprintf(stderr, "IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_CONFIG %s\n", name);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_VALUE_I64: {
        const iree_hal_amdgpu_trace_plot_value_i64_t* event =
            (const iree_hal_amdgpu_trace_plot_value_i64_t*)packet_ptr;
        const char* plot_name = iree_hal_amdgpu_trace_buffer_translate_literal(
            trace_buffer, event->plot_name);
        fprintf(stderr,
                "IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_VALUE_I64 %s = %" PRIi64 "\n",
                plot_name, event->value);
        offset += sizeof(*event);
      } break;
      default: {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "invalid trace event type %d; possibly corrupt ringbuffer",
            event_type);
      } break;
    }
  }

  // Notify the device that we've read up to the write offset.
  iree_atomic_store(&trace_buffer->device_buffer->read_commit_offset,
                    write_commit_offset, iree_memory_order_release);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#else

iree_status_t iree_hal_amdgpu_trace_buffer_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_string_view_t executor_name,
    hsa_agent_t device_agent, hsa_queue_t* control_queue,
    iree_device_size_t ringbuffer_capacity,
    const iree_hal_amdgpu_device_kernels_t* kernels,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_trace_buffer_t* out_trace_buffer) {
  // No-op.
  return iree_ok_status();
}

void iree_hal_amdgpu_trace_buffer_deinitialize(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer) {
  // No-op.
}

iree_status_t iree_hal_amdgpu_trace_buffer_flush(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer) {
  // No-op.
  return iree_ok_status();
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
