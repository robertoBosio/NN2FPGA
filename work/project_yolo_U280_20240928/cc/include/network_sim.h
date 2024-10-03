#ifndef __NETWORK_SIM__
#define __NETWORK_SIM__
#include "params.h"
#include "nn2fpga/debug.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include "yolo.h"
#include "nn2fpga/mm2s.h"
#include "nn2fpga/s2mm.h"

#ifndef CSIM
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "cmdlineparser.h"
#endif /* CSIM */


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
	
/************************* c_params *************************/
	const int c_params_dim = 8649648;
	t_params_st *c_params = nullptr;
	posix_memalign((void**)&c_params, 4096, 8649648 * sizeof(t_params_st));
	std::ifstream file_weights(prj_root + "npy/yolo_weights.bin", std::ios::binary);
	if (!file_weights.is_open()) {
		std::cerr << "Error: unable to open the parameters file." << std::endl;
		exit(-1);
	}
	file_weights.read(reinterpret_cast<char*>(c_params), 8649648 * sizeof(t_params_st));
	file_weights.close();

#ifdef CSIM
	hls::stream<t_params_stream> i_data_params;
	hls::stream<t_in_mem> c_inp_1_stream;
	hls::stream<t_net_19> c_outp1_stream;
	hls::stream<t_net_25> c_outp2_stream;
	nn2fpga::mm2s <
		t_params_st,
		t_params_stream>
	(
		c_params,
		c_params_dim,
		i_data_params
	);

	nn2fpga::mm2s <
		t_in_mem,
		t_in_mem>
	(
		inp_1,
		n_inp,
		c_inp_1_stream
	);

	auto start = std::chrono::high_resolution_clock::now();

	yolo(
		c_inp_1_stream,
		i_data_params,
		c_outp1_stream,
		c_outp2_stream
	);

	auto end = std::chrono::high_resolution_clock::now();

	nn2fpga::s2mm <
		t_out_mem1,
		t_net_19>
	(
		o_outp1,
		n_out1,
		c_outp1_stream
	);

	nn2fpga::s2mm <
		t_out_mem2,
		t_net_25>
	(
		o_outp2,
		n_out2,
		c_outp2_stream
	);

#else
	sda::utils::CmdLineParser parser;
	parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
	parser.addSwitch("--device_id", "-d", "device index", "0");
	parser.addSwitch("--n_images", "-n", "input number of images", "1");
	parser.addSwitch("--upload_weights", "-w", "input upload weights flag", "1");
	parser.parse(argc, argv);
	std::string binaryFile = parser.value("xclbin_file");
	int device_index = stoi(parser.value("device_id"));
	int upload_weights_flag = stoi(parser.value("upload_weights"));
	std::cout << "Opening the device " << device_index << std::endl;
	auto device = xrt::device(device_index);
	std::cout << "Loading the xclbin " << binaryFile << std::endl;
	auto uuid = device.load_xclbin(binaryFile);

	auto mm2s_activations = xrt::kernel(device, uuid, "mm2s_activations");
	auto buff_activations = xrt::bo(device, (int*)inp_1, n_inp * 8, mm2s_activations.group_id(0));
	std::cout << "Synching h2d inp_1" << std::endl;
	buff_activations.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	auto mm2s_a = xrt::run(mm2s_activations);
	mm2s_a.set_arg(0, buff_activations);
	mm2s_a.set_arg(1, n_inp);

	auto mm2s_weights = xrt::kernel(device, uuid, "mm2s_weights");
	auto buff_weights = xrt::bo(device, (int*)c_params, 8649648, mm2s_weights.group_id(0));
	std::cout << "Synching h2d c_params" << std::endl;
	buff_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	auto mm2s_w = xrt::run(mm2s_weights);
	mm2s_w.set_arg(0, buff_weights);
	mm2s_w.set_arg(1, c_params_dim);

	auto s2mm_output = xrt::kernel(device, uuid, "s2mm_outputs");
	auto buff_output = xrt::bo(device, (int*)o_outp1, n_out, s2mm_output.group_id(0));
	auto s2mm_o = xrt::run(s2mm_output);
	s2mm_o.set_arg(0, buff_output);
	s2mm_o.set_arg(1, n_out);

	auto s2mm_output = xrt::kernel(device, uuid, "s2mm_outputs");
	auto buff_output = xrt::bo(device, (int*)o_outp1, n_out, s2mm_output.group_id(0));
	auto s2mm_o = xrt::run(s2mm_output);
	s2mm_o.set_arg(0, buff_output);
	s2mm_o.set_arg(1, n_out);

	s2mm_o.start();
	s2mm_o.start();

	auto start = std::chrono::high_resolution_clock::now();

	if (upload_weights_flag) {
		mm2s_w.start();
	}
	mm2s_a.start();
	s2mm_o.wait();
	s2mm_o.wait();

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Synching d2h o_outp1" << std::endl;
	buff_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

#endif /* CSIM */

	free(c_params);
	return (end - start);
}

#endif