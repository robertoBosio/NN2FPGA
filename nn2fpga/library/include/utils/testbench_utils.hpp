#pragma once

#include "ap_int.h"
#include "cnpy.h"
#include "hls_stream.h"
#include <cmath>
#include <vector>

// Function to quantize a floating-point value to an integer
// with round half to even
int quantize(float value, float scale, int zeropt) {
  return static_cast<int>(std::nearbyint(value / scale)) + zeropt;
}

float dequantize(int value, float scale, int zeropt) {
  return (value - zeropt) * scale;
}

template <typename TAxi, typename TData>
void hls_stream_to_npy(const std::string &output_path,
                       hls::stream<TAxi> &stream, int data_per_word,
                       int bits_per_data, float scale, int zeropt, const std::vector<size_t> &shape) {
  std::vector<float> output_data;
  while (!stream.empty()) {
    TAxi word = stream.read();
    for (int j = 0; j < data_per_word; j++) {
      TData hls_data =
          word.data.range((j + 1) * bits_per_data - 1, j * bits_per_data);
      int quant_data = static_cast<int>(hls_data);
      float dequantized_value = dequantize(quant_data, scale, zeropt);
      output_data.push_back(dequantized_value);
    }
  }

  // Save to .npy file
  cnpy::npy_save(output_path, &output_data[0], shape, "w");
}

template <typename TAxi, typename TData>
void npy_to_hls_stream(const std::string &input_path, hls::stream<TAxi> &stream,
                       int data_per_word, int bits_per_data, float scale,
                       int zeropt) {
  cnpy::NpyArray input_array = cnpy::npy_load(input_path);
  float *input_data = input_array.data<float>();
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
      int quant_data = quantize(input_data[i + j], scale, zeropt);
      TData hls_data = static_cast<TData>(quant_data);
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