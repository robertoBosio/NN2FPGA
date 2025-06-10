#include <unistd.h>
#include <iostream>
#include <string>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <unistd.h>
#include <sys/stat.h>
#include "network_sim.h"
#include "params.h"
#include "cmdlineparser.h"
#include "nn2fpga/debug.h"

#define READ_WIDTH 8
#define READ_BYTES 1
#define CLASSES 10

bool directoryExists(const std::string &path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

template<typename T>
std::vector<T>
readBinaryFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return std::vector<T>();
  }
  std::vector<T> array_fromfile;
  T value;
  while (file.read(reinterpret_cast<char*>(&value), sizeof(T))) {
    array_fromfile.push_back(value);
  }
  file.close();
  return array_fromfile;
}

int main(int argc, char** argv) {
	sda::utils::CmdLineParser parser;
	parser.addSwitch("--n_images", "-n", "input number of images", "1");
	parser.addSwitch("--upload_weights", "-w", "input upload weights flag", "1");
  parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
  parser.addSwitch("--device_id", "-d", "device index", "0");
  parser.addSwitch("--onnx_path", "-o", "onnx absolute path", "");
  parser.addSwitch("--dataset", "-dt", "datset name", "");
	parser.parse(argc, argv);
  
  /* Onnx full path */
  std::string onnx_path = parser.value("onnx_path");

  /* Dataset name */
  std::string dataset = parser.value("dataset");
 
  /* Images per batch */
  const unsigned int c_batch = stoi(parser.value("n_images"));
  
  /* Bytes per image */
  const unsigned int c_index =
    (c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw) / c_data_per_packet;
  
  /* Bytes per batch */
  const int n_bytes =
    ((c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw) *
     c_act_width / 8);

  std::chrono::duration<double> inference_time;
  unsigned int results[c_batch];

  // Get the current working directory
  char currentDirBuffer[FILENAME_MAX];
  bool foundCifar = false;
  std::string cifarPath = "";
  std::string projPath = "";
  if (getcwd(currentDirBuffer, FILENAME_MAX) != nullptr) {
    std::string currentDirStr(currentDirBuffer);
    std::cout << "Current working directory: " << currentDirStr << std::endl;

    // Find the position of "NN2FPGA" in the path
    size_t topPos = currentDirStr.find("NN2FPGA");
    
    // find the position of "project_" in the path
    size_t projPos = currentDirStr.find("project_");
    // Search the "/" after projPos
    size_t projEndPos = currentDirStr.find("/", projPos);
    std::cout << "projPos: " << projPos << std::endl;
    std::cout << "projEndPos: " << projEndPos << std::endl;
    
    if (projEndPos != std::string::npos){
      currentDirStr = currentDirStr.substr(0, projEndPos + 1);
    } else {
      currentDirStr += "/";
    }

    if (directoryExists(currentDirStr)) {
      projPath = currentDirStr;
    } else {
      std::cerr << "Error: Could not find the path to the project directory." << std::endl;
      std::cerr << projPath << std::endl;
      return -1;
    }

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
  
  if (cifarPath == "" || projPath == ""){
    std::cerr << "Error: Could not find the path to the CIFAR-10 dataset or the project directory." << std::endl;
    std::cerr << "cifar: " << cifarPath << std::endl;
    std::cerr << "proj:" << projPath << std::endl;
    return -1;
  }  

  std::cout << "Sending " << c_batch << " images." << std::endl;
  t_in_mem *mem_activations;
  posix_memalign((void**)&mem_activations, 4096, c_index * c_batch * sizeof(t_in_mem));
  t_out_mem1 *mem_outputs;
  posix_memalign((void**)&mem_outputs, 4096, CLASSES * c_batch * sizeof(t_out_mem1));
  
  std::cout << "Allocated " << c_index * c_batch << " ap_uint<64> for activations." << std::endl;
  std::cout << "Allocated " << CLASSES * c_batch << " ap_uint<8> for output results." << std::endl;

#ifndef CSIM
	if (argc < 3) {
		return EXIT_FAILURE;
	}
#endif /* CSIM */

 std::string command = "python3 preprocess.py " + std::to_string(c_batch) + " " + onnx_path + " " + dataset;
 int status = system(command.c_str());

 if (status < 0)
   std::cout << "Error: " << strerror(errno) << ".\n";
 else {
   if (WIFEXITED(status)) {
     if (WEXITSTATUS(status) != 0){
       std::cout << "ONNX inference failed.\n";
       return -1;
     }
   } else {
     std::cout << "Program exited abnormaly.\n";
   }
 }

  std::vector<float> images = readBinaryFile<float>("/tmp/images_preprocessed.bin");
  std::vector<int> labels = readBinaryFile<int>("/tmp/labels_preprocessed.bin");
  std::vector<float> expected_results = readBinaryFile<float>("/tmp/results_preprocessed.bin");

  int mem_activations_p = 0;
  for (int i = 0; i < c_batch; i++) {
    ap_uint<c_width_act_stream> send_data = 0;
    unsigned int s_bytes = 0;
    for (auto it = 0; it < c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw; it++) {

      int s_par = (s_bytes % c_data_per_packet);
      float data_f = images[i * c_produce_stream_ich * c_produce_stream_ih *
                              c_produce_stream_iw +
                            it];
      t_inp_1_part data_uf = data_f;
      ap_uint<c_act_width> data_u;
      data_u.range(c_act_width - 1, 0) = data_uf.range(c_act_width - 1, 0);
      send_data.range(c_act_width * (s_par + 1) - 1, c_act_width * s_par) =
        data_u;

      if (s_par == c_data_per_packet - 1) {
        mem_activations[mem_activations_p++] = send_data;
        send_data = 0;
      }
      s_bytes++;
    }
    
    // TODO: Change arguments to -w 0 after first run in CSIM, to remove the
    // warning of not empty stream at the end of the simulation
#ifdef CSIM
  std::cout << "Starting inference" << std::endl;
  inference_time = networkSim(argc,
                              argv,
                              projPath,
                              c_index,
                              CLASSES,
                              &mem_activations[i * c_index],
                              &mem_outputs[i * CLASSES]);
#endif /* CSIM */
  }

#ifndef CSIM
  inference_time = networkSim(argc,
                              argv,
                              projPath,
                              c_index * c_batch,
                              CLASSES * c_batch,
                              mem_activations,
                              mem_outputs);

#endif

  unsigned int correct = 0;
  bool passed = true;
  for (int image = 0; image < c_batch; image++) {
    t_out_mem1 max_value = INT32_MIN;
    int max_index = 0;
    std::cout << image << " image" << std::endl;
    for (int g = 0; g < CLASSES; g++) {
      auto data = mem_outputs[g + image * CLASSES];
      t_out_mem1 expected_data = expected_results[g + image * CLASSES];
      ap_int<c_act_width> data_int[2];
      data_int[0].range(c_act_width - 1, 0) = data.range(c_act_width - 1, 0);
      data_int[1].range(c_act_width - 1, 0) = expected_data.range(c_act_width - 1, 0);
      std::cout << data << "\t(" << data_int[0] << ")\t" << expected_data
                << "\t(" << data_int[1] << ")" << std::endl;
      passed &= (data == expected_data);
      if (data > max_value) {
        max_value = data;
        max_index = g;
      }
    }
    std::cout << "COMPUTED LABEL " << max_index << " -------- ";
    std::cout << "EXPECTED LABEL " << (ap_int<8>)(labels[image])
              << std::endl;
    results[image] = max_index;
    if (max_index == labels[image]) {
      correct++;
    }
  }

  std::cout << "ACCURACY " << correct / (float)(c_batch) << std::endl;
  std::cout << "FPS: " << (c_batch) / (inference_time.count())<< std::endl;
  std::cout << "AVG LATENCY: " << (inference_time.count() * 1000000) / c_batch
            << " us" << std::endl;
  std::cout << "######## TEST " << (passed ? "PASSED" : "FAILED") << " ########" << std::endl;
  
  free(mem_activations);
  free(mem_outputs);
  return (passed ? 0 : -1);
}
