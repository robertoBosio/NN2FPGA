#include <unistd.h>

#include "cifar/cifar10_reader.hpp"
#include "nn2fpga/debug.h"
#include "cmdlineparser.h"
#include "params.h"
#include <iostream>
#include <cstring>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

char *getcwd(char *buf, size_t size);
#define READ_WIDTH 8
#define READ_BYTES 1

int main(int argc, char** argv) {
  char cwd[100];
  const t_weights_st c_weights[544] __attribute__ ((aligned(4096))) = {
    38,  237, 225, 3,   246, 242, 214, 4,   50,  227, 50,  249, 3,   5,   25,
    249, 33,  254, 217, 7,   2,   236, 206, 0,   24,  233, 45,  4,   248, 2,
    22,  2,   32,  253, 218, 3,   255, 249, 198, 6,   12,  242, 64,  0,   252,
    250, 27,  255, 7,   242, 0,   249, 254, 255, 194, 248, 30,  236, 9,   247,
    252, 5,   29,  6,   14,  237, 12,  243, 0,   244, 198, 246, 6,   244, 17,
    245, 8,   251, 21,  8,   22,  248, 20,  2,   5,   244, 197, 248, 242, 242,
    16,  248, 253, 253, 22,  250, 15,  251, 86,  251, 8,   18,  222, 255, 44,
    255, 210, 7,   7,   5,   26,  254, 2,   231, 80,  8,   4,   9,   221, 244,
    14,  0,   205, 254, 6,   254, 22,  8,   22,  236, 86,  247, 253, 10,  225,
    255, 251, 241, 225, 251, 254, 255, 31,  1,   14,  0,   164, 6,   10,  240,
    245, 50,  251, 236, 38,  0,   249, 252, 248, 249, 27,  242, 157, 0,   255,
    252, 239, 51,  233, 239, 39,  9,   247, 247, 245, 250, 37,  241, 163, 249,
    248, 245, 3,   47,  213, 225, 34,  4,   252, 248, 253, 247, 5,   253, 212,
    244, 0,   250, 5,   52,  7,   249, 246, 1,   255, 0,   232, 1,   255, 0,
    201, 246, 0,   248, 5,   40,  231, 244, 238, 4,   249, 251, 232, 5,   3,
    248, 193, 245, 2,   253, 250, 47,  203, 250, 239, 248, 252, 4,   242, 6,
    6,   254, 17,  245, 252, 251, 12,  46,  255, 0,   187, 252, 253, 246, 243,
    248, 253, 252, 23,  9,   255, 250, 5,   48,  223, 248, 194, 254, 252, 250,
    250, 7,   15,  244, 25,  252, 3,   8,   4,   36,  209, 12,  197, 3,   250,
    247, 231, 0,   228, 20,  213, 254, 5,   27,  42,  217, 34,  22,  46,  2,
    247, 3,   237, 251, 221, 26,  217, 6,   5,   23,  31,  203, 245, 10,  46,
    245, 247, 255, 245, 8,   241, 26,  210, 255, 246, 19,  39,  203, 248, 18,
    45,  251, 252, 252, 234, 8,   203, 22,  250, 6,   245, 31,  48,  213, 16,
    31,  6,   5,   255, 4,   246, 9,   210, 29,  233, 246, 5,   35,  36,  205,
    249, 32,  248, 11,  252, 1,   243, 246, 212, 28,  1,   9,   6,   25,  50,
    218, 239, 33,  254, 252, 255, 248, 240, 255, 215, 12,  70,  246, 246, 41,
    48,  202, 27,  48,  212, 0,   6,   252, 229, 8,   198, 27,  50,  246, 251,
    45,  45,  207, 3,   26,  222, 252, 254, 248, 243, 252, 221, 26,  70,  8,
    246, 46,  64,  217, 233, 39,  207, 2,   247, 246, 230, 252, 28,  100, 235,
    176, 30,  142, 254, 169, 253, 86,  222, 80,  15,  19,  9,   198, 8,   10,
    252, 14,  254, 55,  253, 71,  255, 135, 255, 198, 252, 172, 250, 252, 220,
    51,  68,  36,  220, 227, 3,   225, 0,   254, 108, 26,  234, 255, 98,  255,
    15,  1,   240, 0,   1,   16,  1,   240, 240, 212, 17,  45,  12,  2,   34,
    209, 210, 219, 5,   193, 213, 244, 35,  188, 30,  61,  78,  32,  45,  20,
    241, 255, 237, 2,   96,  218, 220, 31,  95,  31,  0,   1,   0,   31,  0,
    0,   1,   16,  240, 240, 1,   255, 240, 0,   66,  30,  14,  239, 32,  0,
    0,   0,   0,   15
  };

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
  const unsigned int n_bytes = c_index * c_par;
  const unsigned int c_classes = 10;
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
  std::cout << "Allocating " << c_index << " ap_uint<64> for activations." << std::endl;
  std::cout << "Allocating " << c_classes << " ap_uint<8> for output results." << std::endl;

  int s_batch = 0;
  int results[c_batch];
  std::cout << "Copying weights" << std::endl;
  buff_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::cout << "Launching mm2s weights..." << std::endl;
  auto mm2s_w_run = xrt::run(mm2s_w);
  mm2s_w_run.set_arg(0, buff_weights);
  mm2s_w_run.set_arg(1, c_weights_dim);
  mm2s_w_run.start();

  for (auto it = dataset.test_images.begin(); it != dataset.test_images.end();
       ++it) {
    int s_bytes = 0;
    auto mem_activations_p = 0;

    for (auto itt = it->begin(); itt != it->end(); ++itt) {
      mem_activations[mem_activations_p++] = (ap_uint<64>)(*itt);
      s_bytes += c_par;
      if (s_bytes == n_bytes) break;
    }

    std::cout << "Copying data activations" << std::endl;
    buff_activations.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    ///////////////////////// KERNEL EXECUTION ON IMAGE ///////////////////////

    std::cout << "Launching s2mm..." << std::endl;
    auto s2mm_run = xrt::run(s2mm);
    s2mm_run.set_arg(0, buff_results);
    s2mm_run.set_arg(1, c_classes);
    s2mm_run.start();
    
    std::cout << "Launching mm2s activations..." << std::endl;
    auto mm2s_a_run = xrt::run(mm2s_a);
    mm2s_a_run.set_arg(0, buff_activations);
    mm2s_a_run.set_arg(1, c_index);
    mm2s_a_run.start();

    std::cout << "Waiting mm2s weights to finish..." << std::endl;
    mm2s_w_run.wait();
    std::cout << "Waiting mm2s activation to finish..." << std::endl;
    mm2s_w_run.wait();
    std::cout << "Waiting s2mm to finish..." << std::endl;
    s2mm_run.wait();

    // networkSim(c_index, 10, mem_activations, mem_output);

    std::cout << "Getting Results..." << std::endl;
    buff_results.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::cout << "" << std::endl;
    auto max_value = UINT32_MAX;
    int max_index = 0;

    for (int g = 0; g < 10; g++){
       t_net_conv_3 data = mem_output[g];
       std::cout << data << std::endl;
       if (data > max_value){
        max_value = data;
        max_index = g;
       }
    }

    // if (max_index != (ap_int<8>)(dataset.test_labels[s_batch]))
    //   std::cout << "ERROR" << std::endl;
    std::cout << "COMPUTED LABEL " << max_index << " -------- ";
    std::cout << "EXPECTED LABEL " << (ap_int<8>)(dataset.test_labels[s_batch])
              << std::endl;
    results[s_batch] = max_index;

    s_batch++;
    if (s_batch == c_batch) break;
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
