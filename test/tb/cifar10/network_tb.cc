#include <unistd.h>
#include <iostream>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include "network_sim.h"
#include "params.h"
#include "cmdlineparser.h"
#include "cifar/cifar10_reader.hpp"
#include "nn2fpga/debug.h"

#define READ_WIDTH 8
#define READ_BYTES 1
#define ACTIVATION_PARALLELISM 8
#define CLASSES 10

bool directoryExists(const std::string &path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

int main(int argc, char** argv) {
	sda::utils::CmdLineParser parser;
	parser.addSwitch("--n_images", "-n", "input number of images", "1");
	parser.addSwitch("--upload_weights", "-w", "input upload weights flag", "1");
  parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
  parser.addSwitch("--device_id", "-d", "device index", "0");
	parser.parse(argc, argv);
 
  /* Images per batch */
  const unsigned int c_batch = stoi(parser.value("n_images"));
  // const unsigned int c_batch = 2;
  /* Bytes per activation data stream */
  const unsigned int c_par = c_inp_1 / ACTIVATION_PARALLELISM;
  /* Bytes per image */
  const unsigned int c_index =
    (c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw) / c_par;
  /* Bytes per batch */
  const int n_bytes = c_index * c_par;
  
  std::chrono::duration<double> inference_time;
  unsigned int results[c_batch];

  // Get the current working directory
  char currentDirBuffer[FILENAME_MAX];
  bool foundCifar = false;
  std::string cifarPath = "";
  if (getcwd(currentDirBuffer, FILENAME_MAX) != nullptr) {
    std::string currentDirStr(currentDirBuffer);
    std::cout << "Current working directory: " << currentDirStr << std::endl;

    // Find the position of "NN2FPGA" in the path
    size_t topPos = currentDirStr.find("NN2FPGA");

    if (topPos != std::string::npos) {
      // Remove everything after NN2FPGA
      currentDirStr = currentDirStr.substr(0, topPos);
      currentDirStr += "NN2FPGA/work/";
      std::cout << currentDirStr << std::endl;
      // Check if the new path exists
      if (directoryExists(currentDirStr)) {
        cifarPath = currentDirStr;
      }
    }
  }
  
  if (cifarPath == "")
    return -1;

  std::cout << "Sending " << c_batch << " images." << std::endl;
  // t_in_mem mem_activations [c_index * c_batch] __attribute__((aligned(4096)));
  // t_out_mem mem_output [CLASSES * c_batch] __attribute__((aligned(4096)));

  t_in_mem *mem_activations;
  posix_memalign((void**)&mem_activations, 4096, c_index * c_batch * sizeof(t_in_mem));
  t_out_mem *mem_outputs;
  posix_memalign((void**)&mem_outputs, 4096, CLASSES * c_batch * sizeof(t_out_mem));
  
  std::cout << "Allocated " << c_index * c_batch << " ap_uint<64> for activations." << std::endl;
  std::cout << "Allocated " << CLASSES * c_batch << " ap_uint<8> for output results." << std::endl;

#ifndef CSIM
	if (argc < 3) {
		return EXIT_FAILURE;
	}
#endif /* CSIM */

  auto dataset = cifar::read_dataset<std::vector, std::vector, char, char>(cifarPath + "data/cifar-10-batches-bin/");
  int s_batch = 0;
  int mem_activations_p = 0;
  for (auto it = dataset.test_images.begin(); it != dataset.test_images.end();
       ++it) {

    ap_uint<64> send_data = 0;
    unsigned int s_bytes = 0;
    for (auto itt = it->begin(); itt != it->end(); itt++) {

      int s_par = (s_bytes % c_par);
      unsigned int data = (ap_uint<8>)(*itt);
      send_data.range(8 * (s_par + 1) - 1, 8 * s_par) = (ap_uint<8>)(data);
      std::cout << (ap_uint<8>)data << " ";
      if (s_par == c_par - 1)
        std::cout << std::endl;

      if ((s_par % ACTIVATION_PARALLELISM) == ACTIVATION_PARALLELISM - 1)
        mem_activations[mem_activations_p++] = send_data;
      s_bytes++;
      if (s_bytes == n_bytes)
        break;
    }

    // TODO: Change arguments to -w 0 after first run in CSIM, to remove the
    // warning of not empty stream at the end of the simulation
#ifdef CSIM
  std::cout << "Starting inference" << std::endl;
  inference_time = networkSim(argc,
                              argv,
                              cifarPath,
                              c_index,
                              CLASSES,
                              &mem_activations[c_index * s_batch],
                              &mem_outputs[s_batch * CLASSES]);
#endif /* CSIM */

    s_batch++;
    if (s_batch == c_batch)
      break;
  }

#ifndef CSIM
  inference_time = networkSim(argc,
                              argv,
                              cifarPath,
                              c_index * c_batch,
                              CLASSES * c_batch,
                              mem_activations,
                              mem_outputs);

#endif
  
    for (int image = 0; image < c_batch; image++) {
    t_out_mem max_value = INT32_MIN;
    int max_index = 0;
    std::cout << image << " image" << std::endl;
    for (int g = 0; g < CLASSES; g++) {
      auto data = mem_outputs[g + image * CLASSES];
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

  int s_labels = 0;
  float correct = 0;
  for (auto it = dataset.test_labels.begin(); it != dataset.test_labels.end();
       ++it) {
    if ((int)(*it) == results[s_labels])
      correct++;

    s_labels++;

    if (s_labels == (c_batch))
      break;
  }

  std::cout << "ACCURACY " << correct / (float)(c_batch) << std::endl;
  std::cout << "FPS: " << (c_batch) / (inference_time.count())<< std::endl;
  std::cout << "AVG LATENCY: " << (inference_time.count() * 1000000) / c_batch
            << " us" << std::endl;
  
  free(mem_activations);
  free(mem_outputs);
  return 0;
}
