// #include <ap_utils.h>
#include <unistd.h>

#include "cifar/cifar10_reader.hpp"
#include "nn2fpga/debug.h"
#include "network_sim.h"
#include "hls_stream.h"
/* #include "MemoryWeights.hpp" */
/* #include "../src/MemoryManagement.hpp" */

char *getcwd(char *buf, size_t size);
#define READ_WIDTH 8
#define READ_BYTES 1

int main() {
  // hls::stream<t_inp_1> i_data;
  /* t_o_data o_data_exp[12]; */
  // hls::stream<t_o_data> o_data_sim;
// #pragma HLS interface axis port = i_data
// #pragma HLS interface axis port = o_data_sim

  const int c_par = c_inp_1 / 8;
  const int c_index =
      (c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw) / c_par;
  const int c_labels = 1;

  char cwd[100];
  std::cout << "CURRENT WORKING DIRECTORY" << std::endl;
  std::cout << getcwd(cwd, sizeof(cwd)) << std::endl;
  auto dataset = cifar::read_dataset<std::vector, std::vector, char, char>();

  /* for (int i = 0; i < dataset.test_images.size(); i++) { */
  /* 	std::cout << dataset.test_images.at(i) << ' '; */
  /* } */
  /* const int c_batch = dataset.test_images.size(); */
  const int c_batch = 2;
  const int n_bytes = c_index * c_par;
  std::cout << "SENDING " << c_batch << " IMAGES" << std::endl;
  std::cout << "SENDING " << n_bytes << " BYTES" << std::endl;

  ap_uint<64> mem_activations[c_index];
  t_net_conv_3 mem_output[10];
  std::cout << "Allocating " << c_index << " ap_uint<64> for activations." << std::endl;
  std::cout << "Allocating " << 10 << " ap_uint<8> for output results." << std::endl;

  int s_batch = 0;
  int results[c_batch];

  for (auto it = dataset.test_images.begin(); it != dataset.test_images.end();
       ++it) {
    int s_bytes = 0;
    auto mem_activations_p = 0;

    for (auto itt = it->begin(); itt != it->end(); ++itt) {
      mem_activations[mem_activations_p++] = (ap_uint<64>)(*itt);
#ifdef DEBUG
      std::cout << (ap_uint<8>)(*itt) << " ";
#endif
      s_bytes += c_par;
      if (s_bytes == n_bytes) break;
    }

#ifdef DEBUG
    std::cout << std::endl;
#endif
    // INIT DATA

    ///////////////////////// KERNEL EXECUTION ON IMAGE ///////////////////////

    /* std::cout << "--------------------- KERNEL -----------------------" <<
     * "\n"; */
    networkSim(c_index, 10, mem_activations, mem_output);

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
