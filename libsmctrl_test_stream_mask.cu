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

int main() {
  int res;
  uint32_t *smids_d, *smids_h;
  uint32_t num_tpcs;
  int num_sms, sms_per_tpc, num_blocks, uniq_sms;
  cudaStream_t stream;

  SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  if (res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0))
    error(1, res, "libsmctrl_test_stream: Unable to get TPC configuration for test");
  sms_per_tpc = num_sms/num_tpcs;
  num_blocks = num_sms * 2;

  SAFE(cudaStreamCreate(&stream));
  libsmctrl_set_stream_mask(stream, ~0x1ull);

  SAFE(cudaMalloc(&smids_d, num_blocks * sizeof(uint32_t)));
  if (!(smids_h = (uint32_t*)malloc(num_blocks * sizeof(uint32_t))))
    error(1, errno, "libsmctrl_test_stream: Unable to allocate memory for test");

  read_and_store_smid<<<num_blocks, BLOCK_THREADS, 0, stream>>>(smids_d, SPIN_CYCLES);
  SAFE(cudaGetLastError());
  SAFE(cudaMemcpyAsync(smids_h, smids_d, num_blocks * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost, stream));
  SAFE(cudaStreamSynchronize(stream));

  uniq_sms = count_unique(smids_h, num_blocks);
  if (uniq_sms > sms_per_tpc || smids_h[num_blocks - 1] > (uint32_t)sms_per_tpc - 1) {
    printf("libsmctrl_test_stream: ***Test failure.***\n"
           "libsmctrl_test_stream: Reason: With stream TPC mask set to "
           "constrain kernels to TPC ID 0, a kernel ran on %d unique SMs "
           "and reached SM ID %u (expected at most %d SMs and max ID %d).\n",
           uniq_sms, smids_h[num_blocks - 1], sms_per_tpc, sms_per_tpc - 1);
    return 1;
  }

  printf("libsmctrl_test_stream: Test passed!\n"
         "libsmctrl_test_stream: Reason: With a stream partition enabled which "
         "contained only TPC ID 0, the test kernel was found to use only %d "
         "SMs, and all SMs in-use had IDs below %d.\n", uniq_sms, sms_per_tpc);
  return 0;
}
