#include <errno.h>
#include <error.h>
#include <stdio.h>
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

int count_unique(uint32_t* arr, int len) {
  qsort(arr, len, sizeof(uint32_t), sort_asc);
  int num_uniq = 1;
  for (int i = 0; i < len - 1; i++)
    num_uniq += (arr[i] != arr[i + 1]);
  return num_uniq;
}

void run_smid_kernel(uint32_t* smids_h, uint32_t* smids_d, int num_blocks) {
  read_and_store_smid<<<num_blocks, BLOCK_THREADS>>>(smids_d, SPIN_CYCLES);
  SAFE(cudaGetLastError());
  SAFE(cudaMemcpy(smids_h, smids_d, num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

int main() {
  int res;
  uint32_t *smids_d, *smids_next_h, *smids_native_h;
  uint32_t num_tpcs;
  int num_sms, sms_per_tpc, num_blocks;
  int uniq_next, uniq_native;

  SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  if (res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0))
    error(1, res, "libsmctrl_test_next: Unable to get TPC configuration for test");
  sms_per_tpc = num_sms/num_tpcs;
  num_blocks = num_sms * 2;

  SAFE(cudaMalloc(&smids_d, num_blocks * sizeof(uint32_t)));
  if (!(smids_next_h = (uint32_t*)malloc(num_blocks * sizeof(uint32_t))))
    error(1, errno, "libsmctrl_test_next: Unable to allocate memory for test");
  if (!(smids_native_h = (uint32_t*)malloc(num_blocks * sizeof(uint32_t))))
    error(1, errno, "libsmctrl_test_next: Unable to allocate memory for test");

  libsmctrl_set_next_mask(~0x1ull);
  run_smid_kernel(smids_next_h, smids_d, num_blocks);
  run_smid_kernel(smids_native_h, smids_d, num_blocks);

  uniq_next = count_unique(smids_next_h, num_blocks);
  uniq_native = count_unique(smids_native_h, num_blocks);
  if (uniq_next > sms_per_tpc || smids_next_h[num_blocks - 1] > (uint32_t)sms_per_tpc - 1) {
    printf("libsmctrl_test_next: ***Test failure.***\n"
           "libsmctrl_test_next: Reason: With next-launch TPC mask set to "
           "constrain kernels to TPC ID 0, the next kernel ran on %d unique "
           "SMs and reached SM ID %u (expected at most %d SMs and max ID %d).\n",
           uniq_next, smids_next_h[num_blocks - 1], sms_per_tpc, sms_per_tpc - 1);
    return 1;
  }
  if (uniq_native <= sms_per_tpc) {
    printf("libsmctrl_test_next: ***Test failure.***\n"
           "libsmctrl_test_next: Reason: The launch after the next-launch mask "
           "used only %d unique SMs; expected the one-shot mask to be cleared.\n",
           uniq_native);
    return 1;
  }

  printf("libsmctrl_test_next: Test passed!\n"
         "libsmctrl_test_next: Reason: The next-launch partition used only %d "
         "SMs, and the following unmasked launch used %d SMs.\n",
         uniq_next, uniq_native);
  return 0;
}
