#ifndef __NETWORK_SIM__
#define __NETWORK_SIM__
#include "params.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include "top_wrapper.h"


std::chrono::duration<double> networkSim(
	int argc,
	char** argv,
	std::string prj_root,
	const unsigned int n_inp,
	const unsigned int n_out1,
	const unsigned int n_out2,
	const t_in_mem* inp_1,
	t_out_mem1* o_outp1,
	t_out_mem2* o_outp2
) {
	
	const int c_params_dim = 8649648;
	t_params_st *c_params;
	
	posix_memalign((void**)&c_params, 4096, 8649648 * sizeof(t_params_st));
	std::ifstream file_weights(prj_root + "npy/yolo_weights.bin", std::ios::binary);
	if (!file_weights.is_open()) {
		std::cerr << "Error: unable to open the parameters file." << std::endl;
		exit(-1);
	}
	file_weights.read(reinterpret_cast<char*>(c_params), 8649648 * sizeof(t_params_st));
	file_weights.close();

	auto start = std::chrono::high_resolution_clock::now();

	top_wrapper(
		inp_1,
		c_params,
		o_outp1,
		o_outp2
	);

	auto end = std::chrono::high_resolution_clock::now();

	free(c_params);
	return (end - start);
}

#endif
