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
#define CLASSES 2

bool directoryExists(const std::string &path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

cv::Mat opencv_transform(cv::Mat image) {
    // Resize the shorter side to 256 pixels while maintaining the aspect ratio

    // Convert to tensor
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);  // Convert BGR to RGB

    // Convert to NumPy array and normalize
    image.convertTo(image, CV_32F);

    int h = image.rows;
    int w = image.cols;
    std::cout << "#### Resizing image" << std::endl;
    std::cout << "#### Original size: " << h << "x" << w << std::endl;

    // cv::Scalar mean(0.485, 0.456, 0.406);
    // cv::Scalar std(0.229, 0.224, 0.225);
    // image = (image - mean) / std;

    // cv::split(transposed, image_channels);

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
  bool founddataset = false;
  std::string datasetPath = "";
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
        datasetPath = currentDirStr;
      }
    }
  }
  
  if (datasetPath == "")
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
  std::string path = "/tools/datasets/vw/vw_coco2014_96/person/";
  std::cout << "Taking images from " << path << std::endl;

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
  typedef ap_fixed<32,16> t_transform;
  if ((dir = opendir (path.c_str())) != NULL) {
    /* print all the files and directories within directory */
    std::cout << "OPENED DIRECTORY" << std::endl;
    while ((ent = readdir (dir)) != NULL) {
        printf ("%s\n", ent->d_name);
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0){
            printf("Image not found\n");
            continue;
        }
        std::string file_path = path + ent->d_name;
        std::cout << "path: " << file_path << std::endl;

        cv::Mat img;
    
        img = cv::imread(file_path);

        auto result_ocv = opencv_transform(img);
        
        // cv::resize(img,result_ocv,cv::Size(c_produce_stream_iw,c_produce_stream_ih),0,0,cv::INTER_AREA);

        // Iterate over elements of result_ocv per channel
        unsigned int s_bytes = 0;
        ap_uint<64> s_data = 0;
        std::cout << "STORING IMAGE" << std::endl;
        for (int i = 0; i < result_ocv.rows; i++) {
            for (int j = 0; j < result_ocv.cols; j++) {
                cv::Vec3f pixel = result_ocv.at<cv::Vec3f>(i, j);
                // Iterate over channels (typically BGR)
                // Iterate over channells on RGB
                for (int c = 0; c < 3; c++) {
                // for (int c = 2; c > -1; c--) {
                    //std::cout << (int)pixel[c] << std::endl;
                    int s_par = (s_bytes % ACTIVATION_PARALLELISM);
                    if (s_par == 0) {
                      std::cout << "Packet: ";
                    }
                    t_transform tmp = (float)pixel[c];
                    std::cout << tmp << " ";
                    ap_uint<8> tmp2 = tmp;
                    s_data.range(8 * (s_par + 1) - 1, 8 * s_par) = tmp2.range(7,0);

                    // #ifdef DEBUG
                    // std::cout << (ap_uint<8>)(pixel[c]) << " ";
                    // #endif

                    if (s_par == (c_par - 1)) {
                        mem_activations[mem_activations_p++] = s_data;
                        std::cout << std::endl;
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
                              datasetPath,
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
                              datasetPath,
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
      float data = mem_outputs[g + image * CLASSES];
      std::cout << data << " " << mem_outputs[g + image * CLASSES] << std::endl;
      if (data > max_value) {
        max_value = data;
        max_index = g;
      }
    }
    std::cout << "COMPUTED LABEL " << max_index << " -------- ";
    // std::cout << "EXPECTED LABEL " << (ap_int<8>)(dataset.test_labels[image])
            //   << std::endl;
    results[image] = max_index;
  }

  int s_labels = 0;
  float correct = 0;
//   for (auto it = dataset.test_labels.begin(); it != dataset.test_labels.end();
//        ++it) {
//     if ((int)(*it) == results[s_labels])
//       correct++;

//     s_labels++;

//     if (s_labels == (c_batch))
//       break;
//   }

  std::cout << "ACCURACY " << correct / (float)(c_batch) << std::endl;
  std::cout << "FPS: " << (c_batch) / (inference_time.count())<< std::endl;
  std::cout << "AVG LATENCY: " << (inference_time.count() * 1000000) / c_batch
            << " us" << std::endl;
  
  free(mem_activations);
  free(mem_outputs);
  return 0;
}
