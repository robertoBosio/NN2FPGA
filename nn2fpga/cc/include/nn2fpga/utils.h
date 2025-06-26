#ifndef NN2FPGA_UTILS_H_
#define NN2FPGA_UTILS_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#pragma GCC diagnostic pop

#include "ap_int.h"
#include "hls_stream.h"
#include "cnpy.h"
#include <vector>

namespace nn2fpga
{

  template <typename T>
  void hls_stream_to_npy(const std::string &output_path, hls::stream<T> &stream)
  {
    std::vector<T> output_data;
    while (!stream.empty())
    {
      output_data.push_back(stream.read());
    }

    // Save to .npy file
    cnpy::npy_save(output_path, &output_data[0], {output_data.size()}, "w");
  }

  template <typename T>
  void npy_to_hls_stream(const std::string &input_path, hls::stream<T> &stream)
  {
    cnpy::NpyArray input_array = cnpy::npy_load(input_path);
    T *input_data = input_array.data<T>();
    std::vector<size_t> shape = input_array.shape;

    if (shape.size() != 1)
    {
      std::cerr << "Input array must be 1D.\n";
      return;
    }

    size_t num_elements = shape[0];
    for (int i = 0; i < num_elements; ++i)
    {
      stream.write(input_data[i]);
    }
  }

} // namespace nn2fpga

#endif // NN2FPGA_UTILS_H_
