# fork of libsmctrl (you probably want to use the original repo which has new features)
Artifact from [Hardware Compute Partitioning on NVIDIA GPUs*
](https://www.cs.unc.edu/~jbakita/rtas23.pdf) paper. Original repo is [here](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/).

## Build
```
make libsmctrl.a
```

## Test
```
make tests
./libsmctrl_test_global_mask
./libsmctrl_test_stream_mask
./libsmctrl_test_next_mask
```

The CUDA tests use `NVCCFLAGS ?= -arch=all-major` by default so CUDA 13.x
toolchains emit SASS for installed GPUs instead of relying on driver JIT for a
newer PTX version. Override it if your `nvcc` does not support this flag:
```
make tests NVCCFLAGS="-arch=sm_80"
```

`./libsmctrl_test_gpc_info` requires the `nvdebug` kernel module and its
`/proc/gpu*` files.

## SMID print demo
```
make libsmctrl_demo_smid_print
./libsmctrl_demo_smid_print
./libsmctrl_demo_smid_print 0x4
./libsmctrl_demo_smid_print --smid 6
```

The demo runs one native baseline kernel and one masked kernel. The kernel
prints each SMID the first time it observes it:
```
native smid 0
native smid 1
...
masked smid 0
masked smid 1
summary: native printed 108 SMIDs, masked printed 2 SMIDs
```

The optional positional argument is an allowed TPC mask, where set bits mean
"allow this TPC". For example, `0x4` allows TPC 2. `--smid N` allows the TPC
containing SMID `N`; masking is still TPC-granular, so it may print neighboring
SMIDs from the same TPC.
