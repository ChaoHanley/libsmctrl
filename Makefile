CC = gcc
NVCC ?= nvcc
NVCCFLAGS ?= -arch=all-major
# -fPIC is needed in all cases, as we may be linked into another shared library
CFLAGS = -fPIC
LDFLAGS = -lcuda -I/usr/local/cuda/include

.PHONY: clean tests

all: libsmctrl.so libsmctrl.a

libsmctrl.so: libsmctrl.c libsmctrl.h
	$(CC) $< -shared -o $@ $(CFLAGS) $(LDFLAGS)

libsmctrl.a: libsmctrl.c libsmctrl.h
	$(CC) $< -c -o libsmctrl.o $(CFLAGS) $(LDFLAGS)
	ar rcs $@ libsmctrl.o

# Use static linking with tests to avoid LD_LIBRARY_PATH issues
libsmctrl_test_gpc_info: libsmctrl_test_gpc_info.c libsmctrl.a
	$(CC) $< -o $@ -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_global_mask: libsmctrl_test_global_mask.cu libsmctrl.a
	$(NVCC) $< -o $@ -g $(NVCCFLAGS) -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_stream_mask: libsmctrl_test_stream_mask.cu libsmctrl.a
	$(NVCC) $< -o $@ -g $(NVCCFLAGS) -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_next_mask: libsmctrl_test_next_mask.cu libsmctrl.a
	$(NVCC) $< -o $@ -g $(NVCCFLAGS) -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_demo_smid_print: libsmctrl_demo_smid_print.cu libsmctrl.a
	$(NVCC) $< -o $@ -g $(NVCCFLAGS) -L. -l:libsmctrl.a $(LDFLAGS)

tests: libsmctrl_test_gpc_info libsmctrl_test_global_mask libsmctrl_test_stream_mask libsmctrl_test_next_mask

clean:
	rm -f libsmctrl.so libsmctrl.a libsmctrl.o libsmctrl_test_gpc_info \
	      libsmctrl_test_global_mask libsmctrl_test_stream_mask \
	      libsmctrl_test_next_mask libsmctrl_demo_smid_print
