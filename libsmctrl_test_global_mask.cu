#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "libsmctrl.h"

#define SAFE(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) \
    error(1, EIO, "CUDA call failed: %s", cudaGetErrorString(err)); \
} while (0)

__global__ void read_and_store_smid(uint32_t* smid_arr, unsigned long long cycles) {
  if (threadIdx.x != 1)
    return;
  int smid;
  unsigned long long start;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  while (true) {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (now - start >= cycles)
      break;
  }
  smid_arr[blockIdx.x] = smid;
}

#define BLOCK_THREADS 1024
#define SPIN_CYCLES 100000000ull

int sort_asc(const void* a, const void* b) {
  uint32_t lhs = *(const uint32_t*)a;
  uint32_t rhs = *(const uint32_t*)b;
  return (lhs > rhs) - (lhs < rhs);
}

// Warning: Mutates input array via qsort
int count_unique(uint32_t* arr, int len) {
  qsort(arr, len, sizeof(uint32_t), sort_asc);
  int num_uniq = 1;
  for (int i = 0; i < len - 1; i++)
    num_uniq += (arr[i] != arr[i + 1]);
  return num_uniq;
}

int main() {
  int res;
  uint32_t *smids_native_d, *smids_native_h;
  uint32_t *smids_partitioned_d, *smids_partitioned_h;
  int uniq_native, uniq_partitioned;
  uint32_t num_tpcs;
  int num_sms, sms_per_tpc, num_blocks;

  // Determine number of SMs per TPC
  SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  if (res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0))
    error(1, res, "libsmctrl_test_global: Unable to get TPC configuration for test");
  sms_per_tpc = num_sms/num_tpcs;
  num_blocks = num_sms * 2;

  // Test baseline (native) behavior without partitioning
  SAFE(cudaMalloc(&smids_native_d, num_blocks * sizeof(uint32_t)));
  if (!(smids_native_h = (uint32_t*)malloc(num_blocks * sizeof(uint32_t))))
    error(1, errno, "libsmctrl_test_global: Unable to allocate memory for test");
  read_and_store_smid<<<num_blocks, BLOCK_THREADS>>>(smids_native_d, SPIN_CYCLES);
  SAFE(cudaGetLastError());
  SAFE(cudaMemcpy(smids_native_h, smids_native_d, num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  uniq_native = count_unique(smids_native_h, num_blocks);
  if (uniq_native < sms_per_tpc) {
    printf("libsmctrl_test_global: ***Test failure.***\n"
           "libsmctrl_test_global: Reason: In baseline test, %d blocks of %d "
           "threads were launched on the GPU, but only %d SMs were utilized, "
           "when it was expected that at least %d would be used.\n", num_blocks,
           BLOCK_THREADS, uniq_native, sms_per_tpc);
    return 1;
  }

  // Verify that partitioning changes the SMID distribution
  libsmctrl_set_global_mask(~0x1); // Enable only one TPC
  SAFE(cudaMalloc(&smids_partitioned_d, num_blocks * sizeof(uint32_t)));
  if (!(smids_partitioned_h = (uint32_t*)malloc(num_blocks * sizeof(uint32_t))))
    error(1, errno, "libsmctrl_test_global: Unable to allocate memory for test");
  read_and_store_smid<<<num_blocks, BLOCK_THREADS>>>(smids_partitioned_d, SPIN_CYCLES);
  SAFE(cudaGetLastError());
  SAFE(cudaMemcpy(smids_partitioned_h, smids_partitioned_d, num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  // Make sure it only ran on the number of TPCs provided
  // May run on up to two SMs, as up to two per TPC
  uniq_partitioned = count_unique(smids_partitioned_h, num_blocks);
  if (uniq_partitioned > sms_per_tpc) {
    printf("libsmctrl_test_global: ***Test failure.***\n"
           "libsmctrl_test_global: Reason: With global TPC mask set to "
           "constrain all kernels to a single TPC, a kernel of %d blocks of "
           "1024 threads was launched and found to run on %d SMs (at most %d---"
           "one TPC---expected).\n", num_blocks, uniq_partitioned, sms_per_tpc);
    return 1;
  }

  // Make sure it ran on the right TPC
  if (smids_partitioned_h[num_blocks - 1] > (uint32_t)sms_per_tpc - 1) {
    printf("libsmctrl_test_global: ***Test failure.***\n"
           "libsmctrl_test_global: Reason: With global TPC mask set to"
           "constrain all kernels to the first TPC, a kernel was run and found "
           "to run on an SM ID as high as %u (max of %d expected).\n",
           smids_partitioned_h[num_blocks - 1], sms_per_tpc - 1);
    return 1;
  }

  printf("libsmctrl_test_global: Test passed!\n"
         "libsmctrl_test_global: Reason: With a global partition enabled which "
         "contained only TPC ID 0, the test kernel was found to use only %d "
         "SMs (%d without), and all SMs in-use had IDs below %d (were contained"
         " in the first TPC).\n", uniq_partitioned, uniq_native, sms_per_tpc);
  return 0;
}
