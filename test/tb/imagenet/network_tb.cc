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
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define READ_WIDTH 8
#define READ_BYTES 1
#define ACTIVATION_PARALLELISM 8
#define CLASSES 1000

// Cross-platform function to get directory names in a given path
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

// Sort and fill hash table function
void sortAndFillHashTable(const std::string& path, std::unordered_map<std::string, int>& hashTable) {
    std::vector<std::string> directories = getDirectories(path);

    // Sort the directories by name
    std::sort(directories.begin(), directories.end());

    // Fill the hash table with directory names and their positions
    for (int i = 0; i < directories.size(); ++i) {
        hashTable[directories[i]] = i;
    }
}

bool directoryExists(const std::string &path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

void
printFlattenedMatToFile(const cv::Mat& mat, const std::string& outputFilePath)
{
  if (mat.empty()) {
    std::cerr << "Input matrix is empty!" << std::endl;
    return;
  }

  // Open the file for writing
  std::ofstream outputFile(outputFilePath);

  if (!outputFile.is_open()) {
    std::cerr << "Unable to open the output file!" << std::endl;
    return;
  }
  
  int rows = mat.rows;
  int cols = mat.cols;
  int channels = mat.channels();
  std::cout << channels << "x" << rows << "x" << cols << std::endl;

  // Write matrix values to the file using nested loops
  for (int k = 0; k < channels; ++k) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        outputFile << std::setprecision(32) << mat.at<cv::Vec3f>(i, j)[k]
                   << '\n';
      }
    }
  }

  // Close the file
  outputFile.close();
}

cv::Mat opencv_transform(cv::Mat image) {
    // Resize the shorter side to 256 pixels while maintaining the aspect ratio

    // Convert to tensor
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);  // Convert BGR to RGB

    // Convert to NumPy array and normalize
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    // image /= 255.0;  // Assuming the original image values are in the range [0, 255]

    int h = image.rows;
    int w = image.cols;
    std::cout << "#### Resizing image" << std::endl;
    std::cout << "#### Original size: " << h << "x" << w << std::endl;
    int new_h, new_w;

    if (h < w) {
        new_h = 256;
        new_w = static_cast<int>(w * (256.0 / h));
    } else {
        new_w = 256;
        new_h = static_cast<int>(h * (256.0 / w));
    }
    std::cout << "#### New Size: " << new_h << "x" << new_w << std::endl;

    // cv::resize(image, image, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    cv::resize(image, image, cv::Size(new_w, new_h), cv::INTER_LINEAR);

    // Center crop to (224, 224)
    h = image.rows;
    w = image.cols;
    int i = (h - 224) / 2;
    int j = (w - 224) / 2;
    std::cout << "#### Extracting region of interest" << std::endl;
    cv::Rect roi(j, i, 224, 224);
    image = image(roi);
    
    // printFlattenedMatToFile(image, "/home/roberto/Documents/NN2FPGA/nn2fpga/tmp/logs/image_preprocessed_opencv.txt");
    return image;
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
  // const unsigned int c_batch = 10;
  /* Bytes per activation data stream */
  const unsigned int c_par = c_inp_1 / ACTIVATION_PARALLELISM;
  /* Bytes per image */
  const unsigned int c_index =
    (c_produce_stream_ich * c_produce_stream_ih * c_produce_stream_iw) / c_par;
  /* Bytes per batch */
  const int n_bytes = c_index * c_par;
  
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
  t_out_mem *mem_outputs;
  posix_memalign((void**)&mem_outputs, 4096, CLASSES * c_batch * sizeof(t_out_mem));
  
  std::cout << "Allocated " << c_index * c_batch << " ap_uint<64> for activations." << std::endl;
  std::cout << "Allocated " << CLASSES * c_batch << " ap_uint<8> for output results." << std::endl;
  std::string path = "/tools/datasets/Imagenet/train/n01440764/";
  // std::string path = "/tools/datasets/Imagenet/train/n01440764/";
  // std::string path = "/tools/datasets/Imagenet/train/n01530575/";
  // std::string path = "/home/filippo/workspace/NN2FPGA/test/tb/imagenet/images/n15075141/";
  std::cout << "Taking images from " << path << std::endl;
  sortAndFillHashTable("/tools/datasets/Imagenet/train/", pathToLabelHashTable);

#ifndef CSIM
	if (argc < 3) {
		return EXIT_FAILURE;
	}
#endif /* CSIM */


  int mem_activations_p = 0;
  int s_batch = 0;
  DIR *dir;
  struct dirent *ent;
  std::cout << "OPENING DIRECTORY" << std::endl;
  typedef ap_ufixed<8,0> t_transform;
  t_transform mean[3] = {0.485, 0.456, 0.406};
  t_transform std[3] = {0.229, 0.224, 0.225};
  if ((dir = opendir (path.c_str())) != NULL) {
    /* print all the files and directories within directory */
    std::cout << "OPENED DIRECTORY" << std::endl;
    while ((ent = readdir (dir)) != NULL) {
        printf ("%s\n", ent->d_name);
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0){
            printf("Image not found\n");
            continue;
        }

        // Retrieving the correct label for the image.
        size_t lastSlashPos = path.find_last_of("/\\");
        if (lastSlashPos != std::string::npos) {
          auto it = pathToLabelHashTable.find(path.substr(lastSlashPos + 1));
          if (it != pathToLabelHashTable.end()) {
            correct_labels[s_batch] = it->second;
          }
        }

        std::string file_path = path + ent->d_name;
        std::cout << "path: " << file_path << std::endl;
        cv::Mat img;
        img = cv::imread(file_path);
        auto result_ocv = opencv_transform(img);

        // Iterate over elements of result_ocv per channel
        unsigned int s_bytes = 0;
        ap_uint<64> s_data = 0;
        std::cout << "sending image with rows: " << result_ocv.rows << " cols: " << result_ocv.cols << std::endl;
        for (int i = 0; i < result_ocv.rows; i++) {
            for (int j = 0; j < result_ocv.cols; j++) {
                cv::Vec3f pixel = result_ocv.at<cv::Vec3f>(i, j);
                // Iterate over channels (typically BGR)
                // Iterate over channells on RGB
                for (int c = 0; c < 3; c++) {
                // for (int c = 2; c > -1; c--) {
                    //std::cout << (int)pixel[c] << std::endl;
                    int s_par = (s_bytes % ACTIVATION_PARALLELISM);
                    // if (s_par == 0) {
                    //   std::cout << "Packet: ";
                    // }
                    // t_transform tmp = (float)pixel[c];
                    // std::cout << tmp << " ";
                    ap_ufixed<8,0,AP_RND_CONV,AP_SAT> tmp2 = pixel[c];
                    s_data.range(8 * (s_par + 1) - 1, 8 * s_par) = tmp2.range(7,0);

                    // #ifdef DEBUG
                    // std::cout << (ap_uint<8>)(pixel[c]) << " ";
                    // #endif

                    if (s_par == (c_par - 1)) {
                        mem_activations[mem_activations_p++] = s_data;
                        // std::cout << std::endl;
                    }

                    s_bytes++;
                }
            }
        }

    // TODO: Change arguments to -w 0 after first run in CSIM, to remove the
    // warning of not empty stream at the end of the simulation
#ifdef CSIM
  std::cout << "STARTING CSIM" << std::endl;
  inference_time = networkSim(argc,
                              argv,
                              projPath,
                              c_index,
                              CLASSES,
                              &mem_activations[c_index * s_batch],
                              &mem_outputs[s_batch * CLASSES]);
#endif /* CSIM */

        s_batch++;
        if (s_batch == c_batch)
            break;
    }
    closedir (dir);
  } else {
    /* could not open directory */
    perror ("");
    return EXIT_FAILURE;
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
  
  float correct = 0;
  for (int image = 0; image < c_batch; image++) {
    t_out_mem max_value = INT32_MIN;
    int max_index = 0;
    std::cout << image << " image" << std::endl;
    for (int g = 0; g < CLASSES; g++) {
      t_out_mem data = mem_outputs[g + image * CLASSES];
      std::cout << g << ": " << data << std::endl;
      if (data > max_value) {
        max_value = data;
        max_index = g;
      }
    }
    std::cout << "COMPUTED LABEL " << max_index << " -------- ";
    std::cout << "EXPECTED LABEL " << (correct_labels[image])
              << std::endl;
    if (max_index == correct_labels[image])
      correct++;
    results[image] = max_index;
  }

  std::cout << "ACCURACY " << correct / (float)(c_batch) << std::endl;
  std::cout << "FPS: " << (c_batch) / (inference_time.count())<< std::endl;
  std::cout << "AVG LATENCY: " << (inference_time.count() * 1000000) / c_batch
            << " us" << std::endl;
  
  free(mem_activations);
  free(mem_outputs);
  return 0;
}
