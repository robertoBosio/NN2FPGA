#include <ap_utils.h>
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
  hls::stream<t_inp_1> i_data;
  /* t_o_data o_data_exp[12]; */
  hls::stream<t_o_data> o_data_sim;
#pragma HLS interface axis port = i_data
#pragma HLS interface axis port = o_data_sim

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

  int s_batch = 0;
  int results[c_batch];

  for (auto it = dataset.test_images.begin(); it != dataset.test_images.end();
       ++it) {
    int s_bytes = 0;

    for (auto itt = it->begin(); itt != it->end(); ++itt) {
      t_inp_1 s_data;
      int s_par = (s_bytes % c_par);
      unsigned int data = (ap_uint<8>)(*itt);
      // data = data*255/256;
      s_data.data(8 * (s_par + 1) - 1, 8 * s_par) = (ap_uint<8>)(data);
      s_data.keep = -1;

#ifdef DEBUG
      std::cout << (ap_uint<8>)(*itt) << " ";
#endif

      if (s_bytes == (n_bytes - 1))
        s_data.last = true;
      else
        s_data.last = false;
      if (s_par == (c_par - 1)) {
        i_data.write(s_data);
      }

      s_bytes++;
      if (s_bytes == n_bytes) break;
    }

#ifdef DEBUG
    std::cout << std::endl;
#endif
    // INIT DATA

    ///////////////////////// KERNEL EXECUTION ON IMAGE ///////////////////////

    /* std::cout << "--------------------- KERNEL -----------------------" <<
     * "\n"; */
    networkSim(i_data,
            o_data_sim);

    std::cout << "" << std::endl;
    t_o_data s_o_data;
    s_o_data = o_data_sim.read();
    auto max_value = s_o_data.data;
    std::cout << s_o_data.data << std::endl;
    int max_index = 0;
    int s_index = 1;

    do {
      s_o_data = o_data_sim.read();
      std::cout << s_o_data.data << std::endl;
      if (s_o_data.data > max_value) {
        max_value = s_o_data.data;
        /* std::cout << "INDEX " << s_index << std::endl; */
        /* std::cout << "MAX VALUE " << (int32_t)(max_value) << std::endl; */
        max_index = s_index;
      }
      s_index++;
    } while (!s_o_data.last);
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
