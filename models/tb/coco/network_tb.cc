#include <unistd.h>
#include <iostream>
#include <string>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include "network_sim.h"
#include "params.h"
#include "cmdlineparser.h"
#include "nn2fpga/debug.h"
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc/imgproc.hpp>

#define READ_WIDTH 8
#define READ_BYTES 1 
#define CLASSES  512*13*13

// Get directory names in a given path
std::vector<std::string> getDirectories(const std::string& path) {
    std::vector<std::string> directories;
    
    DIR* dir = opendir(path.c_str());
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_DIR) {
                std::string dirName = entry->d_name;
                if (dirName != "." && dirName != "..") {
                    directories.push_back(dirName);
                }
            }
        }
        closedir(dir);
    }
    return directories;
}

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
  parser.addSwitch("--dataset", "-dt", "dataset name", "");
	parser.parse(argc, argv);
  
  /* Onnx full path */
  std::string onnx_path = parser.value("onnx_path");
  std::cout << "ONNX path: " << onnx_path << std::endl;

  /* Dataset name */
  std::string dataset = parser.value("dataset");
  std::cout << "Dataset: " << dataset << std::endl;
 
  /* Images per batch */
  const unsigned int c_batch = stoi(parser.value("n_images"));
  
  /* Bytes per image */
  const unsigned int c_index =
    (c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw) /
    c_data_per_packet;
  
  /* Bytes per batch */
  const int n_bytes =
    (c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw) *
    c_act_width / 8;

  std::chrono::duration<double> inference_time;
  unsigned int results[c_batch];
  unsigned int correct_labels[c_batch];
  std::unordered_map<std::string, int> pathToLabelHashTable;

  // Get the current working directory
  char currentDirBuffer[FILENAME_MAX];
  bool foundImagenet = false;
  std::string imagenetPath = "";
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
    if (projEndPos != std::string::npos){
      currentDirStr = currentDirStr.substr(0, projEndPos + 1);
      // Check if the new path exists
      if (directoryExists(currentDirStr)) {
        projPath = currentDirStr;
      }
    }
    
    if (topPos != std::string::npos) {
      // Remove everything after NN2FPGA
      currentDirStr = currentDirStr.substr(0, topPos);
      currentDirStr += "NN2FPGA/work/";
      std::cout << currentDirStr << std::endl;
      // Check if the new path exists
      if (directoryExists(currentDirStr)) {
        imagenetPath = currentDirStr;
      }
    }
  }
  
  if (imagenetPath == "" || projPath == "")
    return -1;

  std::cout << "Sending " << c_batch << " images." << std::endl;
  // t_in_mem mem_activations [c_index * c_batch] __attribute__((aligned(4096)));
  // t_out_mem mem_output [CLASSES * c_batch] __attribute__((aligned(4096)));

  t_in_mem *mem_activations;
  posix_memalign((void**)&mem_activations, 4096, c_index * c_batch * sizeof(t_in_mem));
  t_out_mem1 *mem_outputs1;
  posix_memalign((void**)&mem_outputs1, 4096,  256 * 26 * 26 * c_batch * sizeof(t_out_mem1));
  t_out_mem2 *mem_outputs2;
  posix_memalign((void**)&mem_outputs2, 4096, 512 * 13 * 13 * c_batch * sizeof(t_out_mem2));
  std::cout << "Allocated " << c_index * c_batch << " ap_uint<64> for activations." << std::endl;
  std::cout << "Allocated " << 256 * 26 * 26 * c_batch << " ap_uint<8> for output1 results." << std::endl;
  std::cout << "Allocated " << 512 * 13 * 13 * c_batch << " ap_uint<8> for output2 results." << std::endl;

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
    // for (auto itt = it->begin(); itt != it->end(); itt++) {
    for (auto it = 0; it < c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw; it++) {

      int s_par = (s_bytes % c_data_per_packet);
      float data_f = images[i * c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw + it];
      t_inp_1_part data_uf = data_f;
      ap_uint<c_act_width> data_u;
      data_u.range(c_act_width - 1, 0) = data_uf.range(c_act_width - 1, 0);
      send_data.range(c_act_width * (s_par + 1) - 1, c_act_width * s_par) =
        data_u;
      if (s_par == c_data_per_packet - 1){
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
                              &mem_outputs1[i * 256 * 26 * 26],
                              &mem_outputs2[i * 512 * 13 * 13]);
#endif /* CSIM */
  }

#ifndef CSIM
  inference_time = networkSim(argc,
                              argv,
                              projPath,
                              c_index * c_batch,
                              CLASSES * c_batch,
                              mem_activations,
                              mem_outputs1
                              mem_outputs2);

#endif

  free(mem_activations);
  free(mem_outputs1);
  free(mem_outputs2);
  // return (passed ? 0 : -1);
  return 0;
}
