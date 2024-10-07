#include <assert.h>
#include <time.h>
#include <errno.h>
#include <sys/time.h>
#include "xtop_wrapper.h"

#ifdef UBUF_FLAG
#include "libubuf/libubuf.h"
#else
#include "xrt.h"
#endif /* UBUF_FLAG */

#define PARAMS_DIM 8649648
#define ACT_DIM 416*416*3
#define OUT_DIM2 512*13*13
#define OUT_DIM1 256*26*26

#ifndef UBUF_FLAG
uint64_t xrt_get_phys(XclDeviceHandle device, XclBufferHandle buffer)
{
	struct xclBOProperties properties;
	xclGetBOProperties(device, buffer, &properties);
	return properties.paddr;
}

XclBufferHandle xrt_create(XclDeviceHandle device, size_t size)
{
	return xclAllocBO(device, size, 0, 1UL << 31);
}
#endif /* UBUF_FLAG */

int main(int argc, char *argv[])
{
	struct timeval t0, t1;
	double dt, dt2;
	int ret;
	srand(time(NULL));

	if (argc != 2) {
		printf("Usage: %s <input images>\n", argv[0]);
		return 1;
	}

	const unsigned long sz = atoi(argv[1]);
	const unsigned long n_inp = ACT_DIM * sz;
	const unsigned long n_out1 = OUT_DIM1 * sz;
	const unsigned long n_out2 = OUT_DIM2 * sz;

#ifdef UBUF_FLAG
  
	// Allocate space for weights and biases
  int32_t *c_params = ubuf_create(PARAMS_DIM);
  assert(c_params != NULL);
  uint64_t c_params_daddr = ubuf_get_phys(c_params);
  assert(c_params_daddr != 0);
  
	// Allocate space for input activations
	int32_t *inp_1 = ubuf_create(n_inp);
	assert(inp_1 != NULL);
	uint64_t inp_1_daddr = ubuf_get_phys(inp_1);
	assert(inp_1_daddr != 0);

  // Allocate space for output activations
	int32_t *o_outp1 = ubuf_create(n_out1);
	assert(o_outp1 != NULL);
	uint64_t o_outp1_daddr = ubuf_get_phys(o_outp1);
	assert(o_outp1_daddr != 0);
	  // Allocate space for output activations
	int32_t *o_outp2 = ubuf_create(n_out2);
	assert(o_outp2 != NULL);
	uint64_t o_outp2_daddr = ubuf_get_phys(o_outp2);
	assert(o_outp2_daddr != 0);

#else
	xclDeviceHandle device = xclOpen(0, 0, 0);
	
	// Allocate space for weights and biases
	xclBufferHandle c_params = xrt_create(device, PARAMS_DIM);
	assert(c_params < 0x80000000);
  uint64_t c_params_daddr = xrt_get_phys(device, c_params);
	assert(c_params_daddr != 0);

	// Allocate space for input activations
	xclBufferHandle inp_1 = xrt_create(device, n_inp);
	assert(inp_1 < 0x80000000);
	uint64_t inp_1_daddr = xrt_get_phys(device, inp_1);
	assert(inp_1_daddr != 0);

	// Allocate space for output activations
	xclBufferHandle o_outp1 = xrt_create(device, n_out1);
	assert(o_outp1 < 0x80000000);
	uint64_t o_outp1_daddr = xrt_get_phys(device, o_outp1);
	assert(o_outp1_daddr != 0);

	xclBufferHandle o_outp2 = xrt_create(device, n_out2);
	assert(o_outp2 < 0x80000000);
	uint64_t o_outp2_daddr = xrt_get_phys(device, o_outp2);
	assert(o_outp2_daddr != 0);

#endif /* UBUF_FLAG */

	// Fill input activations with random data
	unsigned char *temp_inp_1;
	temp_inp_1 = (unsigned char *)malloc(n_inp);
	for (unsigned long i = 0; i < n_inp; ++i) {
		inp_1[i] = rand() % 256;
	}
	ret = xclWriteBO(device, inp_1, temp_inp_1, n_inp, 0);
	assert(ret == 0);
	xclSyncBO(device, inp_1, XCL_BO_SYNC_BO_TO_DEVICE, n_inp, 0);

	// Fill weights and biases with random data
	FILE *file_weights = fopen("resent8_weights.bin", "rb");
	if (!file_weights)
	{
		printf("Error: unable to open the parameters file.\n");
		exit(-1);
	}

	unsigned char temp_params[PARAMS_DIM]; // Assuming t_params_st is defined appropriately
	size_t result = fread(temp_params, sizeof(t_params_st), PARAMS_DIM, file_weights);
	if (result != PARAMS_DIM)
	{
		printf("Error: failed to read the correct number of parameters.\n");
		fclose(file_weights);
		exit(-1);
	}
	fclose(file_weights);
	ret = xclWriteBO(device, c_params, temp_params, PARAMS_DIM, 0);
	assert(ret == 0);
	xclSyncBO(device, c_params, XCL_BO_SYNC_BO_TO_DEVICE, PARAMS_DIM, 0);

	XTop_wrapper top_wrap;
	ret = XTop_wrapper_Initialize(&top_wrap, "top_wrap");
	assert(ret == XST_SUCCESS);

	// memset((void *)in1, c1, BUF_N*4);
	// memset((void *)in2, c2, BUF_N*4);
  
  // ret = ubuf_sync();
  // assert(ret == 0);
  // XTop_wrapper_Set_in1(&top_wrap, in1_daddr);
  // XTop_wrapper_Set_in2(&top_wrap, in2_daddr);
  // XTop_wrapper_Set_out_r(&top_wrap, out_r_daddr);
  // XTop_wrapper_Set_size(&top_wrap, sz);
	XTop_wrapper_Set_n_inp(&top_wrap, n_inp);
	XTop_wrapper_Set_n_out(&top_wrap, n_out1);
	XTop_wrapper_Set_n_out(&top_wrap, n_out2);
	XTop_wrapper_Set_inp_1(&top_wrap, inp_1_daddr);
	XTop_wrapper_Set_c_params(&top_wrap, c_params_daddr);
	XTop_wrapper_Set_o_outp1(&top_wrap, o_outp1_daddr);
	XTop_wrapper_Set_o_outp2(&top_wrap, o_outp2_daddr);

	gettimeofday(&t0, NULL);
	XTop_wrapper_Start(&top_wrap);
  while(XTop_wrapper_IsDone(&top_wrap)==0);
  gettimeofday(&t1, NULL);
	ret = XTop_wrapper_Release(&top_wrap);
	assert(ret == XST_SUCCESS);
  
	dt = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_usec - t0.tv_usec)/1000.0;
	printf("Done, time: %.1fms\n", dt);

	// Read output activations
	unsigned char *temp_outp1;
	temp_outp1 = (unsigned char *)malloc(n_out1);
	xclSyncBO(device, o_outp1, XCL_BO_SYNC_BO_FROM_DEVICE, n_out1, 0);
	ret = xclReadBO(device, o_outp1, temp_outp1, n_out1, 0);
	assert(ret == 0);

	unsigned char *temp_outp2;
	temp_outp2 = (unsigned char *)malloc(n_out2);
	xclSyncBO(device, o_outp2, XCL_BO_SYNC_BO_FROM_DEVICE, n_out2, 0);
	ret = xclReadBO(device, o_outp2, temp_outp2, n_out2, 0);
	assert(ret == 0);

	for (unsigned long i = 0; i < n_out1; ++i) {
		printf("Output %lu: %u\n", i, temp_outp1[i]);
	}
	
	for (unsigned long i = 0; i < n_out2; ++i) {
		printf("Output %lu: %u\n", i, temp_outp2[i]);
	}
	
#ifdef UBUF_FLAG

	ret = ubuf_destroy(c_params);
	ret |= ubuf_destroy(inp_1);
	ret |= ubuf_destroy(o_outp1);
	ret |= ubuf_destroy(o_outp2);

#else

	ret = xclFreeBO(device, c_params);
	ret |= xclFreeBO(device, inp_1);
	ret |= xclFreeBO(device, o_outp1);
	ret |= xclFreeBO(device, o_outp2);
	free(temp_inp_1);
	free(temp_outp1);
	free(temp_outp2);
	xclClose(device);

#endif /* UBUF_FLAG */
	assert(ret == 0);
	return 0;
}
