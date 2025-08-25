#include "DequantQuant.hpp"
#include "StreamingConv.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include "test_config.hpp"
#include <array>
#include <cassert>
#include <iostream>

bool test_run_simple() {

  using TInputStruct = std::array<test_config::TInput, test_config::IN_CH_PAR>;
  using TWeightStruct =
      std::array<std::array<test_config::TWeight, test_config::IN_CH_PAR>,
                 test_config::OUT_CH_PAR>;
  using TBiasStruct = std::array<test_config::TBias, test_config::OUT_CH_PAR>;
  using TOutputStruct =
      std::array<test_config::TOutput, test_config::OUT_CH_PAR>;
  using TAcc = test_config::TBias;
  size_t FW_EXPAND =
      test_config::FW + (test_config::W_PAR - 1) * test_config::STRIDE_W;
  size_t IN_HEIGHT = ((test_config::OUT_HEIGHT - 1) * test_config::STRIDE_H +
                      test_config::DIL_H * (test_config::FH - 1) + 1 -
                      test_config::PAD_T - test_config::PAD_B);
  size_t IN_WIDTH = ((test_config::OUT_WIDTH - 1) * test_config::STRIDE_W +
                     test_config::DIL_W * (test_config::FW - 1) + 1 -
                     test_config::PAD_L - test_config::PAD_R);

  // Create input and output streams
  hls::stream<TInputStruct> i_data[test_config::FH * FW_EXPAND];
  hls::stream<TWeightStruct> i_weights[test_config::FH * test_config::FW];
  hls::stream<TBiasStruct> i_biases[1];
  hls::stream<TOutputStruct> o_data[test_config::W_PAR];

  // Fill input streams with test data
  for (size_t h = 0; h < test_config::OUT_HEIGHT; h++) {
    for (size_t w = 0; w < test_config::OUT_WIDTH; w += test_config::W_PAR) {
      for (size_t i_ich = 0; i_ich < test_config::IN_CH;
           i_ich += test_config::IN_CH_PAR) {
        for (size_t fh = 0; fh < test_config::FH; fh++) {
          for (size_t fw = 0; fw < FW_EXPAND; fw++) {

            size_t input_index_h =
                (h * test_config::STRIDE_H) - test_config::PAD_T + fh;
            size_t input_index_w =
                (w * test_config::STRIDE_W) - test_config::PAD_L + fw;

            TInputStruct input_data;
            for (size_t i_ch_par = 0; i_ch_par < test_config::IN_CH_PAR;
                 i_ch_par++) {
              if (input_index_h < 0 || input_index_h >= IN_HEIGHT ||
                  input_index_w < 0 || input_index_w >= IN_WIDTH) {
                input_data[i_ch_par] = 0; // Padding with zeros
              } else {
                input_data[i_ch_par] =
                    test_config::input_tensor[0][i_ich + i_ch_par]
                                             [input_index_h][input_index_w];
              }
            }
            i_data[fh * FW_EXPAND + fw].write(input_data);
          }
        }
      }
    }
  }

  // Fill weight streams with test data
  for (size_t i_hw = 0; i_hw < test_config::OUT_HEIGHT *
                                   test_config::OUT_WIDTH / test_config::W_PAR;
       i_hw++) {
    for (size_t i_ich = 0; i_ich < test_config::IN_CH;
         i_ich += test_config::IN_CH_PAR) {
      for (size_t i_och = 0; i_och < test_config::OUT_CH;
           i_och += test_config::OUT_CH_PAR) {
        for (size_t fh = 0; fh < test_config::FH; fh++) {
          for (size_t fw = 0; fw < test_config::FW; fw++) {
            TWeightStruct weight_data;
            for (size_t i_ich_par = 0; i_ich_par < test_config::IN_CH_PAR;
                 i_ich_par++) {
              for (size_t i_och_par = 0; i_och_par < test_config::OUT_CH_PAR;
                   i_och_par++) {
                weight_data[i_och_par][i_ich_par] =
                    test_config::weight_tensor[i_och + i_och_par]
                                              [i_ich + i_ich_par][fh][fw];
              }
            }
            i_weights[fh * test_config::FW + fw].write(weight_data);
          }
        }
      }
    }
  }

  for (size_t i_hw = 0; i_hw < test_config::OUT_HEIGHT *
                                   test_config::OUT_WIDTH / test_config::W_PAR;
       i_hw++) {
    for (size_t i_och = 0; i_och < test_config::OUT_CH;
         i_och += test_config::OUT_CH_PAR) {
      TBiasStruct bias_data;
      for (size_t i_och_par = 0; i_och_par < test_config::OUT_CH_PAR;
           i_och_par++) {
        bias_data[i_och_par] = test_config::bias_tensor[i_och + i_och_par];
      }
      i_biases[0].write(bias_data);
    }
  }

  // Run the convolution
  StreamingConv<TInputStruct, test_config::TInput, TWeightStruct, TBiasStruct,
                TOutputStruct, test_config::TOutput, TAcc,
                test_config::Quantizer, test_config::OUT_CH, test_config::IN_CH,
                test_config::OUT_HEIGHT, test_config::OUT_WIDTH, test_config::GROUP,
                test_config::FH, test_config::FW, test_config::STRIDE_H,
                test_config::STRIDE_W, test_config::IN_CH_PAR,
                test_config::OUT_CH_PAR, test_config::W_PAR>
      conv;
  conv.run(i_data, i_weights, i_biases, o_data);

  // Check output streams for expected results
  for (size_t h = 0; h < test_config::OUT_HEIGHT; h++) {
    for (size_t w = 0; w < test_config::OUT_WIDTH; w += test_config::W_PAR) {
      for (size_t i_och = 0; i_och < test_config::OUT_CH;
           i_och += test_config::OUT_CH_PAR) {
        for (size_t i_w_par = 0; i_w_par < test_config::W_PAR; i_w_par++) {
          TOutputStruct output_data = o_data[i_w_par].read();
          for (size_t i_och_par = 0; i_och_par < test_config::OUT_CH_PAR;
               i_och_par++) {

            // Check if the output data matches the expected result
            if (output_data[i_och_par] !=
                test_config::output_tensor[0][i_och + i_och_par][h][w + i_w_par]
                                          ) {
              std::cerr << "Output mismatch at (" << h << ", " << w + i_w_par
                        << ", " << i_och + i_och_par
                        << "): " << output_data[i_och_par] << " != "
                        << test_config::output_tensor[0][i_och + i_och_par][h]
                                                     [w + i_w_par]
                        << std::endl;
              return false;
            }
          }
        }
      }
    }
  }

  // Ensure all input streams are empty
  for (size_t fh = 0; fh < test_config::FH; fh++) {
    for (size_t fw = 0; fw < FW_EXPAND; fw++) {
      if (!i_data[fh * FW_EXPAND + fw].empty()) {
        return false;
      }
    }
  }

  for (size_t fh = 0; fh < test_config::FH; fh++) {
    for (size_t fw = 0; fw < test_config::FW; fw++) {
      if (!i_weights[fh * test_config::FW + fw].empty()) {
        return false;
      }
    }
  }

  if (!i_biases[0].empty()) {
    return false;
  }

  // Ensure all output streams are empty
  for (size_t w = 0; w < test_config::W_PAR; w++) {
    if (!o_data[w].empty()) {
      return false;
    }
  }

  return true;
}

bool test_step_simple_pipelined() {

  using TInputStruct = std::array<test_config::TInput, test_config::IN_CH_PAR>;
  using TWeightStruct =
      std::array<std::array<test_config::TWeight, test_config::IN_CH_PAR>,
                 test_config::OUT_CH_PAR>;
  using TBiasStruct = std::array<test_config::TBias, test_config::OUT_CH_PAR>;
  using TOutputStruct =
      std::array<test_config::TOutput, test_config::OUT_CH_PAR>;
  using TAcc = test_config::TBias;
  size_t FW_EXPAND =
      test_config::FW + (test_config::W_PAR - 1) * test_config::STRIDE_W;
  size_t IN_HEIGHT = ((test_config::OUT_HEIGHT - 1) * test_config::STRIDE_H +
                      test_config::DIL_H * (test_config::FH - 1) + 1 -
                      test_config::PAD_T - test_config::PAD_B);
  size_t IN_WIDTH = ((test_config::OUT_WIDTH - 1) * test_config::STRIDE_W +
                     test_config::DIL_W * (test_config::FW - 1) + 1 -
                     test_config::PAD_L - test_config::PAD_R);

  // Create input and output streams
  hls::stream<TInputStruct> i_data[test_config::FH * FW_EXPAND];
  hls::stream<TWeightStruct> i_weights[test_config::FH * test_config::FW];
  hls::stream<TBiasStruct> i_biases[1];
  hls::stream<TOutputStruct> o_data[test_config::W_PAR];

  // Run the convolution
  StreamingConv<TInputStruct, test_config::TInput, TWeightStruct, TBiasStruct,
                TOutputStruct, test_config::TOutput, TAcc,
                test_config::Quantizer, test_config::OUT_CH, test_config::IN_CH,
                test_config::OUT_HEIGHT, test_config::OUT_WIDTH, test_config::GROUP,
                test_config::FH, test_config::FW, test_config::STRIDE_H,
                test_config::STRIDE_W, test_config::IN_CH_PAR,
                test_config::OUT_CH_PAR, test_config::W_PAR>
      conv(test_config::PIPELINE_DEPTH);

  // Check step function not progressing before any input
  ActorStatus actor_status =
      conv.step(i_data, i_weights, i_biases, o_data);
  bool flag = actor_status.empty() && actor_status.get_current_index() == 0;

  std::cout << "Initial step check: " << flag << std::endl;

  // Fill input streams with test data
  for (size_t h = 0; h < test_config::OUT_HEIGHT; h++) {
    for (size_t w = 0; w < test_config::OUT_WIDTH; w += test_config::W_PAR) {
      for (size_t i_ich = 0; i_ich < test_config::IN_CH;
           i_ich += test_config::IN_CH_PAR) {
        for (size_t fh = 0; fh < test_config::FH; fh++) {
          for (size_t fw = 0; fw < FW_EXPAND; fw++) {

            size_t input_index_h =
                (h * test_config::STRIDE_H) - test_config::PAD_T + fh;
            size_t input_index_w =
                (w * test_config::STRIDE_W) - test_config::PAD_L + fw;

            TInputStruct input_data;
            for (size_t i_ch_par = 0; i_ch_par < test_config::IN_CH_PAR;
                 i_ch_par++) {
              if (input_index_h < 0 || input_index_h >= IN_HEIGHT ||
                  input_index_w < 0 || input_index_w >= IN_WIDTH) {
                input_data[i_ch_par] = 0; // Padding with zeros
              } else {
                input_data[i_ch_par] =
                    test_config::input_tensor[0][i_ich + i_ch_par]
                                             [input_index_h][input_index_w];
              }
            }
            i_data[fh * FW_EXPAND + fw].write(input_data);
          }
        }
      }
    }
  }

  // Fill weight streams with test data
  for (size_t i_hw = 0; i_hw < test_config::OUT_HEIGHT *
                                   test_config::OUT_WIDTH / test_config::W_PAR;
       i_hw++) {
    for (size_t i_ich = 0; i_ich < test_config::IN_CH;
         i_ich += test_config::IN_CH_PAR) {
      for (size_t i_och = 0; i_och < test_config::OUT_CH;
           i_och += test_config::OUT_CH_PAR) {
        for (size_t fh = 0; fh < test_config::FH; fh++) {
          for (size_t fw = 0; fw < test_config::FW; fw++) {
            TWeightStruct weight_data;
            for (size_t i_ich_par = 0; i_ich_par < test_config::IN_CH_PAR;
                 i_ich_par++) {
              for (size_t i_och_par = 0; i_och_par < test_config::OUT_CH_PAR;
                   i_och_par++) {
                weight_data[i_och_par][i_ich_par] =
                    test_config::weight_tensor[i_och + i_och_par]
                                              [i_ich + i_ich_par][fh][fw];
              }
            }
            i_weights[fh * test_config::FW + fw].write(weight_data);
          }
        }
      }
    }
  }

  for (size_t i_hw = 0; i_hw < test_config::OUT_HEIGHT *
                                   test_config::OUT_WIDTH / test_config::W_PAR;
       i_hw++) {
    for (size_t i_och = 0; i_och < test_config::OUT_CH;
         i_och += test_config::OUT_CH_PAR) {
      TBiasStruct bias_data;
      for (size_t i_och_par = 0; i_och_par < test_config::OUT_CH_PAR;
           i_och_par++) {
        bias_data[i_och_par] = test_config::bias_tensor[i_och + i_och_par];
      }
      i_biases[0].write(bias_data);
    }
  }
  
  // Check behaviour at the start of the pipeline
  for (size_t i = 1; i < test_config::PIPELINE_DEPTH; i++) {
    actor_status = conv.step(i_data, i_weights, i_biases, o_data);
    flag &= actor_status.size() == i;
    flag &= actor_status.get_current_index() == i;
    for (size_t i_w_par = 0; i_w_par < test_config::W_PAR; i_w_par++) {
      flag &= o_data[i_w_par].empty(); // No output yet
    }
  }

  std::cout << "Check behaviour at the start of the pipeline: " << flag
            << std::endl;

  // Check delayed output
  size_t cycles_before_output =
      (test_config::OUT_CH / test_config::OUT_CH_PAR) *
      ((test_config::IN_CH / test_config::IN_CH_PAR) - 1);

  for (size_t i = 0; i < cycles_before_output; i++) {
    actor_status = conv.step(i_data, i_weights, i_biases, o_data);
    flag &= actor_status.size() == test_config::PIPELINE_DEPTH - 1;
    for (size_t i_w_par = 0; i_w_par < test_config::W_PAR; i_w_par++) {
      flag &= o_data[i_w_par].empty(); // No output yet
    }
  }

  std::cout << "Check delayed output: " << flag << std::endl;

  // Check arrival of first output
  actor_status = conv.step(i_data, i_weights, i_biases, o_data);
  flag &= actor_status.size() == test_config::PIPELINE_DEPTH - 1;
  for (size_t i_w_par = 0; i_w_par < test_config::W_PAR; i_w_par++) {
    flag &= !o_data[i_w_par].empty(); // One output ready
  }

  std::cout << "Check arrival of first output: " << flag << std::endl;

  // Step through convolution
  for (size_t i = test_config::PIPELINE_DEPTH + cycles_before_output + 1;
       i < test_config::OUT_HEIGHT * test_config::OUT_WIDTH *
               test_config::OUT_CH * test_config::IN_CH /
               (test_config::OUT_CH_PAR * test_config::IN_CH_PAR *
                test_config::W_PAR);
       i++) {
    actor_status = conv.step(i_data, i_weights, i_biases, o_data);
    flag &= actor_status.size() == test_config::PIPELINE_DEPTH - 1;
  }

  std::cout << "Step through convolution: " << flag << std::endl;

  // Step until the end of the pipeline
  for (size_t i = 0; i < test_config::PIPELINE_DEPTH - 1; i++) {
    actor_status = conv.step(i_data, i_weights, i_biases, o_data);
    flag &= actor_status.size() == test_config::PIPELINE_DEPTH - 1 - i;
  }

  std::cout << "Step until the end of the pipeline: " << flag << std::endl;

  // Flush outputs
  while (o_data[0].size() > 0) {
    for (size_t i_w_par = 0; i_w_par < test_config::W_PAR; i_w_par++) {
      o_data[i_w_par].read();
    }
  }

  return flag;
}

int main() {

  bool all_passed = true;

  all_passed &= test_run_simple();
  all_passed &= test_step_simple_pipelined();
  if (!all_passed) {
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  return all_passed ? 0 : 1;
}