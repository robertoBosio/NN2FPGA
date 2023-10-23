#include <unistd.h>

#include "cifar/cifar10_reader.hpp"
#include "nn2fpga/debug.h"
#include "cmdlineparser.h"
#include "params.h"
#include <iostream>
#include <cstring>
#include <chrono>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

char *getcwd(char *buf, size_t size);
#define READ_WIDTH 8
#define READ_BYTES 1

int main(int argc, char** argv) {
  char cwd[100];

  std::cout << "CURRENT WORKING DIRECTORY" << std::endl;
  std::cout << getcwd(cwd, sizeof(cwd)) << std::endl;
  
  // Command Line Parser
  sda::utils::CmdLineParser parser;

  // Switches
  //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
  parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
  parser.addSwitch("--device_id", "-d", "device index", "0");
  parser.parse(argc, argv);

  // Read settings
  std::string binaryFile = parser.value("xclbin_file");
  int device_index = stoi(parser.value("device_id"));

  if (argc < 3) {
    parser.printHelp();
    return EXIT_FAILURE;
  }
  std::cout << "Open the device" << device_index << std::endl;
  auto device = xrt::device(device_index);
  std::cout << "Load the xclbin " << binaryFile << std::endl;
  auto uuid = device.load_xclbin(binaryFile);

  // Allocate Memory in Host Memory
  const int c_weights_dim = 544;
  const int c_par = c_inp_1 / 8;
  const int c_index =
      (c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw) / c_par;
  const unsigned int c_labels = 1;
  const unsigned int c_batch = 2;
  const unsigned int n_bytes = c_index * c_par * c_batch;
  const unsigned int c_classes = 10 * c_batch;
  auto dataset = cifar::read_dataset<std::vector, std::vector, char, char>();
  
  auto mm2s_a = xrt::kernel(device, uuid, "mm2s_activations");
  auto mm2s_w = xrt::kernel(device, uuid, "mm2s_weights");
  auto s2mm = xrt::kernel(device, uuid, "s2mm");

  auto buff_activations = xrt::bo(device, n_bytes, mm2s_a.group_id(0));
  auto buff_weights = xrt::bo(device, (int*)c_weights, c_weights_dim, mm2s_w.group_id(0));
  auto buff_results = xrt::bo(device, c_classes, s2mm.group_id(0));

  std::cout << "SENDING " << c_batch << " IMAGES" << std::endl;
  std::cout << "SENDING " << n_bytes << " BYTES" << std::endl;

  auto mem_activations = buff_activations.map<ap_uint<64>*>();
  auto mem_output = buff_results.map<t_net_conv_3*>();
  std::cout << "Allocating " << c_index * c_batch << " ap_uint<64> for activations." << std::endl;
  std::cout << "Allocating " << c_classes << " ap_uint<8> for output results." << std::endl;

  int s_batch = 0;
  int results[c_batch];
  int mem_activations_p = 0;
  for (auto it = dataset.test_images.begin(); it != dataset.test_images.end();
       ++it) {
    
    ap_uint<64> send_data = 0;
    unsigned int s_bytes = 0;
    for (auto itt = it->begin(); itt != it->end(); itt++) {

      int s_par = (s_bytes % c_par);
      unsigned int data = (ap_uint<8>)(*itt);
      send_data.range(8 * (s_par + 1) - 1, 8 * s_par) = (ap_uint<8>)(data);

      if (s_par % 8 == 7)
        mem_activations[mem_activations_p++] = send_data;
      s_bytes++;
      if (s_bytes == n_bytes) break;
    }

    s_batch++;
    if (s_batch == c_batch) break;
  }

  std::cout << "Copying data activations" << std::endl;
  buff_activations.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  ///////////////////////// KERNEL EXECUTION ON IMAGE ///////////////////////
  std::cout << "Copying weights" << std::endl;
  buff_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::cout << "Launching mm2s weights..." << std::endl;
  auto mm2s_w_run = xrt::run(mm2s_w);
  mm2s_w_run.set_arg(0, buff_weights);
  mm2s_w_run.set_arg(1, c_weights_dim);
  mm2s_w_run.start();

  std::cout << "Launching s2mm..." << std::endl;
  auto s2mm_run = xrt::run(s2mm);
  s2mm_run.set_arg(0, buff_results);
  s2mm_run.set_arg(1, c_classes);
  s2mm_run.start();

  std::cout << "Launching mm2s activations..." << std::endl;
  auto mm2s_a_run = xrt::run(mm2s_a);
  mm2s_a_run.set_arg(0, buff_activations);
  mm2s_a_run.set_arg(1, c_index * c_batch);
  mm2s_a_run.start();

  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "Waiting mm2s weights to finish..." << std::endl;
  mm2s_w_run.wait();
  std::cout << "Waiting mm2s activation to finish..." << std::endl;
  mm2s_a_run.wait();
  std::cout << "Waiting s2mm to finish..." << std::endl;
  s2mm_run.wait();

  // Calculate the duration in milliseconds
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Inference took " << duration.count() * 1000 << " milliseconds."
            << std::endl;

  // networkSim(c_index, 10, mem_activations, mem_output);

  std::cout << "Getting Results..." << std::endl;
  buff_results.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  for (int image = 0; image < c_batch; image++) {
    t_net_conv_3 max_value = INT32_MIN;
    int max_index = 0;
    for (int g = 0; g < 10; g++) {
      t_net_conv_3 data = mem_output[g + image * 10];
      std::cout << data << std::endl;
      if (data > max_value) {
        max_value = data;
        max_index = g;
      }
    }
    std::cout << "COMPUTED LABEL " << max_index << " -------- ";
    std::cout << "EXPECTED LABEL " << (ap_int<8>)(dataset.test_labels[image])
              << std::endl;
    results[image] = max_index;
  }

  const int n_bytes_labels = c_batch;

  int s_labels = 0;
  float correct = 0;
  for (auto it = dataset.test_labels.begin(); it != dataset.test_labels.end();
       ++it) {
    if ((int)(*it) == results[s_labels]) correct++;

    s_labels++;

    if (s_labels == (n_bytes_labels)) break;
  }

  std::cout << "ACCURACY " << correct / (float)(c_batch) << std::endl;

  /* while(o_last == 0); */
  /* while(o_last == 1); */

  /* std::cout << "EXP: " << o_data_exp[0] << "\n"; */

  return 0;
}
