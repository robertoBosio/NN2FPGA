#pragma once

#include "ap_int.h"
#include "cnpy.h"
#include "hls_stream.h"
#include <cmath>
#include <vector>

template <typename TAxi, typename TData, typename TDataNumpy>
void hls_stream_to_npy(const std::string &output_path,
                       hls::stream<TAxi> &stream, int data_per_word,
                       int bits_per_data, const std::vector<size_t> &shape) {
  std::vector<TDataNumpy> output_data;
  while (!stream.empty()) {
    TAxi word = stream.read();
    for (int j = 0; j < data_per_word; j++) {
      TData hls_data =
          word.data.range((j + 1) * bits_per_data - 1, j * bits_per_data);
      TDataNumpy quant_data = static_cast<TDataNumpy>(hls_data);
      output_data.push_back(quant_data);
    }
  }

  // Save to .npy file
  cnpy::npy_save(output_path, &output_data[0], shape, "w");
}

template <typename TAxi, typename TData, typename TDataNumpy>
void npy_to_hls_stream(const std::string &input_path, hls::stream<TAxi> &stream,
                       int data_per_word, int bits_per_data) {
  cnpy::NpyArray input_array = cnpy::npy_load(input_path);
  TDataNumpy *input_data = input_array.data<TDataNumpy>();
  std::vector<size_t> shape = input_array.shape;

  if (shape.size() != 4) {
    std::cerr << "Input array must be 4D.\n";
    return;
  }

  size_t num_elements = shape[0] * shape[1] * shape[2] * shape[3];
  for (int i = 0; i < num_elements; i += data_per_word) {
    TAxi word;
    word.data = 0; // Initialize the data field to zero
    for (int j = 0; j < data_per_word; j++) {
      TData hls_data = static_cast<TData>(input_data[i + j]);
      word.data.range((j + 1) * bits_per_data - 1, j * bits_per_data) =
          hls_data;
    }
    word.last = (i + data_per_word >= num_elements)
                    ? 1
                    : 0; // Set last bit if this is the last word
    word.keep = (1 << data_per_word) - 1; // Set keep bits
    word.strb = (1 << data_per_word) - 1; // Set strb bits
    stream.write(word);
  }
}