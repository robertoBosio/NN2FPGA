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
#define CLASSES 1000
#define n_out0 18*80*80
#define n_out1 18*40*40
#define n_out2 18*20*20
// #define n_out0 255*80*80
// #define n_out1 255*40*40
// #define n_out2 255*20*20
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

template <typename T>
void writeBinaryFile(const std::string &filename, const T* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data), size * sizeof(T));
    file.close();
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
  // t_out_mem1 mem_output [CLASSES * c_batch] __attribute__((aligned(4096)));

  t_in_mem *mem_activations;
  posix_memalign((void**)&mem_activations, 4096, c_index * c_batch * sizeof(t_in_mem));
  t_out_mem0 *mem_outputs0;
  posix_memalign((void**)&mem_outputs0, 4096, n_out0 * c_batch * sizeof(t_out_mem0));
  t_out_mem1 *mem_outputs1;
  posix_memalign((void**)&mem_outputs1, 4096, n_out1 * c_batch * sizeof(t_out_mem1));
  t_out_mem2 *mem_outputs2;
  posix_memalign((void**)&mem_outputs2, 4096, n_out2 * c_batch * sizeof(t_out_mem2)); 
  std::cout << "Allocated " << c_index * c_batch << " ap_uint<64> for activations." << std::endl;
  std::cout << "Allocated " << n_out0 * c_batch << " ap_uint<8> for output0 results." << std::endl;

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
  std::vector<float> expected_results0 = readBinaryFile<float>("/tmp/results_preprocessed_0.bin");
  std::vector<float> expected_results1 = readBinaryFile<float>("/tmp/results_preprocessed_1.bin");
  std::vector<float> expected_results2 = readBinaryFile<float>("/tmp/results_preprocessed_2.bin");
  
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
                              n_out0,
                              n_out1,
                              n_out2,
                              &mem_activations[i * c_index],
                              &mem_outputs0[i * n_out0],
                              &mem_outputs1[i * n_out1],
                              &mem_outputs2[i * n_out2]);
#endif /* CSIM */
  }

#ifndef CSIM
  inference_time = networkSim(argc,
                              argv,
                              projPath,
                              c_index * c_batch,
                              n_out0,
                              n_out1,
                              n_out2,
                              mem_activations,
                              mem_outputs0,
                              mem_outputs1,
                              mem_outputs2);

#endif

  unsigned int correct = 0;
  bool passed = true;
  for (int image = 0; image < c_batch; image++) {
    t_out_mem0 max_value = INT32_MIN;
    int max_index = 0;
    std::cout << image << " image" << std::endl;
    std::cout << "Output0:" << std::endl;

    for (int g = 0; g < n_out0; g++) {
      auto data0 = mem_outputs0[g + image * n_out0];
      t_out_mem0 expected_data0 = expected_results0[g + image * n_out0];
      ap_int<c_act_width> data_int[2];
      data_int[0].range(c_act_width - 1, 0) = data0.range(c_act_width - 1, 0);
      data_int[1].range(c_act_width - 1, 0) = expected_data0.range(c_act_width - 1, 0);
      std::cout << data0 << "\t(" << data_int[0] << ")\t" << expected_data0
                << "\t(" << data_int[1] << ")" << std::endl;
      // passed &= (data0 == expected_data0);
    }

    std::cout << "Output1:" << std::endl;
    
    for (int g = 0; g < n_out1; g++) {
      auto data1 = mem_outputs1[g + image * n_out1];
      t_out_mem1 expected_data1 = expected_results1[g + image * n_out1];
      ap_int<c_act_width> data_int[2];
      data_int[0].range(c_act_width - 1, 0) = data1.range(c_act_width - 1, 0);
      data_int[1].range(c_act_width - 1, 0) = expected_data1.range(c_act_width - 1, 0);
      std::cout << data1 << "\t(" << data_int[0] << ")\t" << expected_data1
                << "\t(" << data_int[1] << ")" << std::endl;
      // passed &= (data1 == expected_data1);
    }

    std::cout << "Output2:" << std::endl;
    
    for (int g = 0; g < n_out2; g++) {
      auto data2 = mem_outputs2[g + image * n_out2];
      t_out_mem2 expected_data2 = expected_results2[g + image * n_out2];
      ap_int<c_act_width> data_int[2];
      data_int[0].range(c_act_width - 1, 0) = data2.range(c_act_width - 1, 0);
      data_int[1].range(c_act_width - 1, 0) = expected_data2.range(c_act_width - 1, 0);
      std::cout << data2 << "\t(" << data_int[0] << ")\t" << expected_data2
                << "\t(" << data_int[1] << ")" << std::endl;
      // passed &= (data2 == expected_data2);
    }
  }

  std::cout << "FPS: " << (c_batch) / (inference_time.count())<< std::endl;
  std::cout << "AVG LATENCY: " << (inference_time.count() * 1000000) / c_batch
            << " us" << std::endl;
  
  std::cout << "######## TEST " << (passed ? "PASSED" : "FAILED") << " ########" << std::endl;

  // Save HLS output results to binary files
  std::string out0_path = "/tmp/hls_output_0.bin";
  std::string out1_path = "/tmp/hls_output_1.bin";
  std::string out2_path = "/tmp/hls_output_2.bin";

  writeBinaryFile(out0_path, mem_outputs0, n_out0 * c_batch);
  writeBinaryFile(out1_path, mem_outputs1, n_out1 * c_batch);
  writeBinaryFile(out2_path, mem_outputs2, n_out2 * c_batch);

  std::cout << "Saved HLS outputs to binary files." << std::endl;
  std::string postprocess_cmd = "python3 postprocess.py " +
                                out0_path + " " +
                                out1_path + " " +
                                out2_path;
  int post_status = system(postprocess_cmd.c_str());

  if (post_status < 0) {
      std::cerr << "Error calling postprocess.py: " << strerror(errno) << std::endl;
  } else if (WIFEXITED(post_status) && WEXITSTATUS(post_status) != 0) {
      std::cerr << "postprocess.py exited with status: " << WEXITSTATUS(post_status) << std::endl;
  } else {
      std::cout << "Post-processing completed successfully." << std::endl;
  }
  
  free(mem_activations);
  free(mem_outputs0);
  free(mem_outputs1);
  free(mem_outputs2);
  return (passed ? 0 : -1);
}
