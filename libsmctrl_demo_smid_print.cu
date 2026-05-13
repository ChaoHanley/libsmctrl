#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "libsmctrl.h"

#define SAFE(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) \
    error(1, EIO, "CUDA call failed: %s", cudaGetErrorString(err)); \
} while (0)

#define BLOCK_THREADS 1024
#define SPIN_CYCLES 100000000ull
#define MAX_ROUNDS 8

__global__ void print_smid_once_per_sm(int* seen, int* seen_count,
                                       unsigned long long cycles, int masked) {
  if (threadIdx.x != 1)
    return;

  int smid;
  unsigned long long start;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));

  if (atomicCAS(&seen[smid], 0, 1) == 0) {
    atomicAdd(seen_count, 1);
    if (masked)
      printf("masked smid %d\n", smid);
    else
      printf("native smid %d\n", smid);
  }

  while (true) {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (now - start >= cycles)
      break;
  }
}

static int run_smid_print(int num_sms, int target_sms, int masked) {
  int *seen_d, *seen_count_d;
  int seen_count_h = 0;
  int num_blocks = target_sms * 4;

  SAFE(cudaMalloc(&seen_d, num_sms * sizeof(int)));
  SAFE(cudaMalloc(&seen_count_d, sizeof(int)));
  SAFE(cudaMemset(seen_d, 0, num_sms * sizeof(int)));
  SAFE(cudaMemset(seen_count_d, 0, sizeof(int)));

  for (int round = 0; round < MAX_ROUNDS && seen_count_h < target_sms; round++) {
    print_smid_once_per_sm<<<num_blocks, BLOCK_THREADS>>>(
      seen_d, seen_count_d, SPIN_CYCLES, masked);
    SAFE(cudaGetLastError());
    SAFE(cudaDeviceSynchronize());
    SAFE(cudaMemcpy(&seen_count_h, seen_count_d, sizeof(int), cudaMemcpyDeviceToHost));
  }

  SAFE(cudaFree(seen_d));
  SAFE(cudaFree(seen_count_d));
  return seen_count_h;
}

static uint64_t parse_u64(const char* str, const char* name) {
  char* end = NULL;
  errno = 0;
  uint64_t val = strtoull(str, &end, 0);
  if (errno || end == str || *end != '\0')
    error(1, EINVAL, "Invalid %s: %s", name, str);
  return val;
}

static int popcount_u64(uint64_t val) {
  return __builtin_popcountll(val);
}

static void print_usage(const char* argv0) {
  fprintf(stderr,
          "Usage:\n"
          "  %s [allowed_tpc_mask]\n"
          "  %s --smid <smid>\n\n"
          "Examples:\n"
          "  %s 0x1        # allow TPC 0, expected SMIDs 0 and 1 on A100\n"
          "  %s 0x4        # allow TPC 2, expected SMIDs 4 and 5 on A100\n"
          "  %s 0x5        # allow TPCs 0 and 2\n"
          "  %s --smid 6   # allow the TPC containing SMID 6\n",
          argv0, argv0, argv0, argv0, argv0, argv0);
}

int main(int argc, char** argv) {
  int res;
  int num_sms, sms_per_tpc;
  uint32_t num_tpcs;
  uint64_t allowed_tpc_mask = 0x1ull;

  SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  if (res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0))
    error(1, res, "libsmctrl_demo_smid_print: Unable to get TPC configuration");
  sms_per_tpc = num_sms / num_tpcs;

  if (argc == 2) {
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
      print_usage(argv[0]);
      return 0;
    }
    allowed_tpc_mask = parse_u64(argv[1], "allowed TPC mask");
  } else if (argc == 3 && !strcmp(argv[1], "--smid")) {
    uint64_t smid = parse_u64(argv[2], "SMID");
    if (smid >= (uint64_t)num_sms)
      error(1, ERANGE, "SMID %lu is outside valid range 0-%d", smid, num_sms - 1);
    allowed_tpc_mask = 1ull << (smid / sms_per_tpc);
  } else if (argc != 1) {
    print_usage(argv[0]);
    return 1;
  }

  if (!allowed_tpc_mask)
    error(1, EINVAL, "Allowed TPC mask must enable at least one TPC");
  if (allowed_tpc_mask >> num_tpcs)
    error(1, ERANGE, "Allowed TPC mask %#lx references TPCs outside valid range 0-%u",
          allowed_tpc_mask, num_tpcs - 1);

  uint64_t libsmctrl_disable_mask = ~allowed_tpc_mask;
  int target_sms = popcount_u64(allowed_tpc_mask) * sms_per_tpc;

  printf("native baseline: expect up to %d SMIDs\n", num_sms);
  int native_seen = run_smid_print(num_sms, num_sms, 0);

  printf("masked run: allowed_tpc_mask=%#lx, libsmctrl disable mask=%#lx, "
         "expect up to %d SMIDs\n",
         allowed_tpc_mask, libsmctrl_disable_mask, target_sms);
  libsmctrl_set_global_mask(libsmctrl_disable_mask);
  int masked_seen = run_smid_print(num_sms, target_sms, 1);

  printf("summary: native printed %d SMIDs, masked printed %d SMIDs\n",
         native_seen, masked_seen);
  return 0;
}
