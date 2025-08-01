#include "BandwidthAdjust.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include <array>
#include <cassert>
#include <iostream>

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t IN_W_PAR, size_t OUT_W_PAR, size_t CH_PAR>
bool test_run_increaseWPAR() {

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(ap_uint<8> data) const {
      return static_cast<TOutput>(data);
    }
  };

  BandwidthAdjustIncreaseStreams<TInputStruct, TInput, TOutputStruct, TOutput,
                                 TruncQuantizer, IN_HEIGHT, IN_WIDTH, IN_CH,
                                 IN_W_PAR, OUT_W_PAR, CH_PAR, CH_PAR>
      bandwidth_adjust;

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[IN_W_PAR];
  hls::stream<TOutputStruct> out_stream[OUT_W_PAR];

  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += IN_W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += CH_PAR) {
      for (size_t w_par = 0; w_par < IN_W_PAR; w_par++) {
        TInputStruct input_struct;
        for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++) {
          input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
        }
        in_stream[w_par].write(input_struct);
      }
    }
  }

  // Run the operator
  bandwidth_adjust.run(in_stream, out_stream);

  // Check output
  bool flag = true;
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += OUT_W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += CH_PAR) {
      for (size_t w_par = 0; w_par < OUT_W_PAR; w_par++) {
        TOutputStruct output_struct = out_stream[w_par].read();
        for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++) {
          flag &= (output_struct[ch_par] == (i + w_par) * IN_CH + ch + ch_par);
        }
      }
    }
  }

  return flag;
}

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t IN_W_PAR, size_t OUT_W_PAR, size_t CH_PAR, size_t PIPELINE_DEPTH = 1>
bool test_step_increaseWPAR() {

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(ap_uint<8> data) const {
      return static_cast<TOutput>(data);
    }
  };

  BandwidthAdjustIncreaseStreams<TInputStruct, TInput, TOutputStruct, TOutput,
                                 TruncQuantizer, IN_HEIGHT, IN_WIDTH, IN_CH,
                                 IN_W_PAR, OUT_W_PAR, CH_PAR, CH_PAR, PIPELINE_DEPTH>
      bandwidth_adjust;

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[IN_W_PAR];
  hls::stream<TOutputStruct> out_stream[OUT_W_PAR];

  // Check step function not progressing in case of no data
  ActorStatus actor_status = bandwidth_adjust.step(in_stream, out_stream);
  bool flag = actor_status.empty() && actor_status.get_current_index() == 0;

  TInputStruct input_struct;
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += IN_W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += CH_PAR) {
      for (size_t w_par = 0; w_par < IN_W_PAR; w_par++) {
        for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++) {
          // Each channel should have the average value of 1
          input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
        }
        in_stream[w_par].write(input_struct);
      }
    }
  }

  // Check pipeline delay
  for (size_t i = 1; i < PIPELINE_DEPTH; i++) {
    actor_status = bandwidth_adjust.step(in_stream, out_stream);
    flag &= actor_status.size() == i;
    flag &= actor_status.get_current_index() == i;
    for (size_t j = 0; j < OUT_W_PAR; j++) {
      flag &= out_stream[j].empty(); // No output yet
    }
  }

  // Run the operator
  for (size_t i = PIPELINE_DEPTH - 1; i < IN_HEIGHT * IN_WIDTH * IN_CH / (CH_PAR * IN_W_PAR);
       i++) {
    actor_status = bandwidth_adjust.step(in_stream, out_stream);
  }

  // Check restart of step function after all iterations
  actor_status = bandwidth_adjust.step(in_stream, out_stream);
  flag &= actor_status.get_current_index() == 0;

  // Flush the output stream
  TOutputStruct output_struct;
  for (size_t i = 0; i < OUT_W_PAR; i++) {
    while (out_stream[i].read_nb(output_struct))
      ;
  }

  return flag;
}

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t IN_W_PAR, size_t OUT_W_PAR, size_t CH_PAR>
bool test_run_decreaseWPAR() {

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(ap_uint<8> data) const {
      return static_cast<TOutput>(data);
    }
  };

  BandwidthAdjustDecreaseStreams<TInputStruct, TInput, TOutputStruct, TOutput,
                                 TruncQuantizer, IN_HEIGHT, IN_WIDTH, IN_CH,
                                 IN_W_PAR, OUT_W_PAR, CH_PAR, CH_PAR>
      bandwidth_adjust;

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[IN_W_PAR];
  hls::stream<TOutputStruct> out_stream[OUT_W_PAR];

  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += IN_W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += CH_PAR) {
      for (size_t w_par = 0; w_par < IN_W_PAR; w_par++) {
        TInputStruct input_struct;
        for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++) {
          // Each channel should have the average value of 1
          input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
        }
        in_stream[w_par].write(input_struct);
      }
    }
  }

  // Run the operator
  bandwidth_adjust.run(in_stream, out_stream);

  // Check output
  bool flag = true;
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += OUT_W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += CH_PAR) {
      for (size_t w_par = 0; w_par < OUT_W_PAR; w_par++) {
        TOutputStruct output_struct = out_stream[w_par].read();
        for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++) {
          flag &= (output_struct[ch_par] == (i + w_par) * IN_CH + ch + ch_par);
        }
      }
    }
  }

  return flag;
}

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t IN_W_PAR, size_t OUT_W_PAR, size_t CH_PAR, size_t PIPELINE_DEPTH = 1>
bool test_step_decreaseWPAR() {

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(ap_uint<8> data) const {
      return static_cast<TOutput>(data);
    }
  };

  BandwidthAdjustDecreaseStreams<TInputStruct, TInput, TOutputStruct, TOutput,
                                 TruncQuantizer, IN_HEIGHT, IN_WIDTH, IN_CH,
                                 IN_W_PAR, OUT_W_PAR, CH_PAR, CH_PAR, PIPELINE_DEPTH>
      bandwidth_adjust;

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[IN_W_PAR];
  hls::stream<TOutputStruct> out_stream[OUT_W_PAR];

  // Check step function not progressing in case of no data
  ActorStatus actor_status = bandwidth_adjust.step(in_stream, out_stream);
  bool flag = actor_status.empty() && actor_status.get_current_index() == 0;

  TInputStruct input_struct;
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += IN_W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += CH_PAR) {
      for (size_t w_par = 0; w_par < IN_W_PAR; w_par++) {
        for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++) {
          // Each channel should have the average value of 1
          input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
        }
        in_stream[w_par].write(input_struct);
      }
    }
  }

  // Check pipeline delay
  for (size_t i = 1; i < PIPELINE_DEPTH; i++) {
    actor_status = bandwidth_adjust.step(in_stream, out_stream);
    flag &= actor_status.size() == i;
    flag &= actor_status.get_current_index() == i;
    for (size_t j = 0; j < OUT_W_PAR; j++) {
      flag &= out_stream[j].empty(); // No output yet
    }
  }

  // Run the operator
  for (size_t i = PIPELINE_DEPTH - 1;
       i < IN_HEIGHT * IN_WIDTH * IN_CH / (CH_PAR * OUT_W_PAR); i++) {
    actor_status = bandwidth_adjust.step(in_stream, out_stream);
  }

  // Check restart of step function after all iterations
  actor_status = bandwidth_adjust.step(in_stream, out_stream);
  flag &= actor_status.get_current_index() == 0;

  // Flush the pipeline
  while (bandwidth_adjust.step(in_stream, out_stream).size() > 0) {
    // Continue stepping until all firings are processed
  }

  // Check firing condition.
  // The pipeline cannot start until OUT_W_PAR streams have data.
  for (size_t i = 0; i < OUT_W_PAR - 1; i++) {
    in_stream[i].write(input_struct);
  }
  actor_status = bandwidth_adjust.step(in_stream, out_stream);
  flag &= actor_status.empty() && actor_status.get_current_index() == 0;
  in_stream[OUT_W_PAR - 1].write(input_struct);
  actor_status = bandwidth_adjust.step(in_stream, out_stream);
  flag &= actor_status.get_current_index() == 1;

  // Flush the output stream again
  TOutputStruct output_struct;
  for (size_t i = 0; i < OUT_W_PAR; i++) {
    while (out_stream[i].read_nb(output_struct))
      ;
  }

  return flag;
}

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t W_PAR, size_t IN_CH_PAR, size_t OUT_CH_PAR>
bool test_run_increaseCHPAR() {

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(ap_uint<8> data) const {
      return static_cast<TOutput>(data);
    }
  };

  BandwidthAdjustIncreaseChannels<TInputStruct, TInput, TOutputStruct, TOutput,
                                  TruncQuantizer, IN_HEIGHT, IN_WIDTH, IN_CH,
                                  W_PAR, W_PAR, IN_CH_PAR, OUT_CH_PAR>
      bandwidth_adjust;

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[W_PAR];
  hls::stream<TOutputStruct> out_stream[W_PAR];

  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += IN_CH_PAR) {
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        TInputStruct input_struct;
        for (size_t ch_par = 0; ch_par < IN_CH_PAR; ch_par++) {
          // Each channel should have the average value of 1
          input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
        }
        in_stream[w_par].write(input_struct);
      }
    }
  }

  // Run the operator
  bandwidth_adjust.run(in_stream, out_stream);

  // Check output
  bool flag = true;
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += OUT_CH_PAR) {
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        TOutputStruct output_struct = out_stream[w_par].read();
        for (size_t ch_par = 0; ch_par < OUT_CH_PAR; ch_par++) {
          flag &= (output_struct[ch_par] == (i + w_par) * IN_CH + ch + ch_par);
        }
      }
    }
  }

  return flag;
}

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t W_PAR, size_t IN_CH_PAR, size_t OUT_CH_PAR, size_t PIPELINE_DEPTH = 1>
bool test_step_increaseCHPAR() {

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(ap_uint<8> data) const {
      return static_cast<TOutput>(data);
    }
  };

  BandwidthAdjustIncreaseChannels<TInputStruct, TInput, TOutputStruct, TOutput,
                                  TruncQuantizer, IN_HEIGHT, IN_WIDTH, IN_CH,
                                  W_PAR, W_PAR, IN_CH_PAR, OUT_CH_PAR, PIPELINE_DEPTH>
      bandwidth_adjust;

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[W_PAR];
  hls::stream<TOutputStruct> out_stream[W_PAR];

  // Check step function not progressing in case of no data
  ActorStatus actor_status = bandwidth_adjust.step(in_stream, out_stream);
  bool flag = actor_status.empty() && actor_status.get_current_index() == 0;

  TInputStruct input_struct;
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += IN_CH_PAR) {
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        for (size_t ch_par = 0; ch_par < IN_CH_PAR; ch_par++) {
          // Each channel should have the average value of 1
          input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
        }
        in_stream[w_par].write(input_struct);
      }
    }
  }

  // Run the operator
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH * IN_CH / (IN_CH_PAR * W_PAR);
       i++) {
    actor_status = bandwidth_adjust.step(in_stream, out_stream);
  }

  // Check restart of step function after all iterations
  actor_status = bandwidth_adjust.step(in_stream, out_stream);
  flag &= actor_status.get_current_index() == 0;

  // Flush the pipeline
  while (bandwidth_adjust.step(in_stream, out_stream).size() > 0) {
    // Continue stepping until all firings are processed
  }
  
  // Flush the output stream
  TOutputStruct output_struct;
  for (size_t i = 0; i < W_PAR; i++) {
    while (out_stream[i].read_nb(output_struct))
      ;
  }

  // Check pipeline progression without writing data.
  for (size_t ch_par = 0; ch_par < OUT_CH_PAR / IN_CH_PAR; ch_par++) {
    for (size_t i = 0; i < W_PAR; i++) {
      in_stream[i].write(input_struct);
    }
  }
  for (size_t i = 0; i < (OUT_CH_PAR / IN_CH_PAR) - 1; i++) {
    actor_status = bandwidth_adjust.step(in_stream, out_stream);
    flag &= actor_status.get_current_index() == i + 1;
    for (size_t j = 0; j < W_PAR; j++) {
      flag &= out_stream[j].empty(); // No output yet
    }
  }
  for (size_t i = 0; i < PIPELINE_DEPTH - 1; i++) {
    actor_status = bandwidth_adjust.step(in_stream, out_stream);
    for (size_t j = 0; j < W_PAR; j++) {
      flag &= out_stream[j].empty(); // No output yet
    }
  }
  actor_status = bandwidth_adjust.step(in_stream, out_stream);
  for (size_t j = 0; j < W_PAR; j++) {
    flag &= !out_stream[j].empty(); // Output should be available now
  }

  // Flush the output stream
  for (size_t i = 0; i < W_PAR; i++) {
    while (out_stream[i].read_nb(output_struct))
      ;
  }

  return flag;
}

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t W_PAR, size_t IN_CH_PAR, size_t OUT_CH_PAR>
bool test_run_decreaseCHPAR() {

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(ap_uint<8> data) const {
      return static_cast<TOutput>(data);
    }
  };

  BandwidthAdjustDecreaseChannels<TInputStruct, TInput, TOutputStruct, TOutput,
                                  TruncQuantizer, IN_HEIGHT, IN_WIDTH, IN_CH,
                                  W_PAR, W_PAR, IN_CH_PAR, OUT_CH_PAR>
      bandwidth_adjust;

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[W_PAR];
  hls::stream<TOutputStruct> out_stream[W_PAR];

  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += IN_CH_PAR) {
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        TInputStruct input_struct;
        for (size_t ch_par = 0; ch_par < IN_CH_PAR; ch_par++) {
          // Each channel should have the average value of 1
          input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
        }
        in_stream[w_par].write(input_struct);
      }
    }
  }

  // Run the operator
  bandwidth_adjust.run(in_stream, out_stream);

  // Check output
  bool flag = true;
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += OUT_CH_PAR) {
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        TOutputStruct output_struct = out_stream[w_par].read();
        for (size_t ch_par = 0; ch_par < OUT_CH_PAR; ch_par++) {
          flag &= (output_struct[ch_par] == (i + w_par) * IN_CH + ch + ch_par);
        }
      }
    }
  }

  return flag;
}

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t W_PAR, size_t IN_CH_PAR, size_t OUT_CH_PAR, size_t PIPELINE_DEPTH = 1>
bool test_step_decreaseCHPAR() {

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(ap_uint<8> data) const {
      return static_cast<TOutput>(data);
    }
  };

  BandwidthAdjustDecreaseChannels<TInputStruct, TInput, TOutputStruct, TOutput,
                                  TruncQuantizer, IN_HEIGHT, IN_WIDTH, IN_CH,
                                  W_PAR, W_PAR, IN_CH_PAR, OUT_CH_PAR, PIPELINE_DEPTH>
      bandwidth_adjust;

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[W_PAR];
  hls::stream<TOutputStruct> out_stream[W_PAR];

  // Check step function not progressing in case of no data
  ActorStatus actor_status = bandwidth_adjust.step(in_stream, out_stream);
  bool flag = actor_status.empty() && actor_status.get_current_index() == 0;

  TInputStruct input_struct;
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += W_PAR) {
    for (size_t ch = 0; ch < IN_CH; ch += IN_CH_PAR) {
      for (size_t w_par = 0; w_par < W_PAR; w_par++) {
        for (size_t ch_par = 0; ch_par < IN_CH_PAR; ch_par++) {
          // Each channel should have the average value of 1
          input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
        }
        in_stream[w_par].write(input_struct);
      }
    }
  }

  // Run the operator
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH * IN_CH / (OUT_CH_PAR * W_PAR);
       i++) {
    actor_status = bandwidth_adjust.step(in_stream, out_stream);
  }

  // Check restart of step function after all iterations
  actor_status = bandwidth_adjust.step(in_stream, out_stream);
  flag &= actor_status.get_current_index() == 0;

  // Flush the pipeline
  while (bandwidth_adjust.step(in_stream, out_stream).size() > 0) {
    // Continue stepping until all firings are processed
  }

  // Check pipeline progression without reading data.
  for (size_t i = 0; i < W_PAR; i++) {
    in_stream[i].write(input_struct);
  }
  for (size_t i = 0; i < IN_CH_PAR / OUT_CH_PAR; i++) {
    actor_status = bandwidth_adjust.step(in_stream, out_stream);
    flag &= actor_status.get_current_index() == i + 1;
  }

  // Flush the output stream
  TOutputStruct output_struct;
  for (size_t i = 0; i < W_PAR; i++) {
    while (out_stream[i].read_nb(output_struct))
      ;
  }

  return flag;
}

int main() {

  bool all_passed = true;

  // Test bandwidth adjustment from 2 to 4 streams
  all_passed &= test_run_increaseWPAR<std::array<ap_uint<8>, 2>, ap_uint<8>,
                                      std::array<ap_uint<8>, 2>, ap_uint<8>, 4,
                                      4, 4, 2, 4, 2>();

  // Test step when passing from 2 to 4 streams
  all_passed &= test_step_increaseWPAR<std::array<ap_uint<8>, 2>, ap_uint<8>,
                                       std::array<ap_uint<8>, 2>, ap_uint<8>, 4,
                                       4, 4, 2, 4, 2>();

  // Test step when passing from 2 to 4 streams, pipelined
  all_passed &= test_step_increaseWPAR<std::array<ap_uint<8>, 2>, ap_uint<8>,
                                       std::array<ap_uint<8>, 2>, ap_uint<8>, 4,
                                       4, 4, 2, 4, 2, 3>();

  // Test bandwidth adjustment from 4 to 2 streams
  all_passed &= test_run_decreaseWPAR<std::array<ap_uint<8>, 2>, ap_uint<8>,
                                      std::array<ap_uint<8>, 2>, ap_uint<8>, 4,
                                      4, 4, 4, 2, 2>();

  // Test step when passing from 4 to 2 streams
  all_passed &= test_step_decreaseWPAR<std::array<ap_uint<8>, 2>, ap_uint<8>,
                                       std::array<ap_uint<8>, 2>, ap_uint<8>, 4,
                                       4, 4, 4, 2, 2>();

  // Test step when passing from 4 to 2 streams, pipelined
  all_passed &= test_step_decreaseWPAR<std::array<ap_uint<8>, 2>, ap_uint<8>,
                                       std::array<ap_uint<8>, 2>, ap_uint<8>, 4,
                                       4, 4, 4, 2, 2, 7>();

  // Test bandwidth adjustment from 2 to 4 channels
  all_passed &= test_run_increaseCHPAR<std::array<ap_uint<8>, 2>, ap_uint<8>,
                                       std::array<ap_uint<8>, 4>, ap_uint<8>, 4,
                                       4, 4, 2, 2, 4>();

  all_passed &= test_step_increaseCHPAR<std::array<ap_uint<8>, 2>, ap_uint<8>,
                                        std::array<ap_uint<8>, 4>, ap_uint<8>,
                                        4, 4, 4, 2, 2, 4>();

  // Test step when passing from 2 to 4 channels, pipelined
  all_passed &= test_step_increaseCHPAR<std::array<ap_uint<8>, 2>, ap_uint<8>,
                                        std::array<ap_uint<8>, 4>, ap_uint<8>,
                                        4, 4, 4, 2, 2, 4, 4>();

  std::cout << "Test up to this point: " << (all_passed ? "True" : "False")
            << std::endl;
  // Test bandwidth adjustment from 4 to 2 channels
  all_passed &= test_run_decreaseCHPAR<std::array<ap_uint<8>, 4>, ap_uint<8>,
                                       std::array<ap_uint<8>, 2>, ap_uint<8>, 4,
                                       4, 4, 4, 4, 2>();

  all_passed &= test_step_decreaseCHPAR<std::array<ap_uint<8>, 4>, ap_uint<8>,
                                        std::array<ap_uint<8>, 2>, ap_uint<8>,
                                        4, 4, 4, 4, 4, 2>();

  // Test step when passing from 4 to 2 channels, pipelined
  all_passed &= test_step_decreaseCHPAR<std::array<ap_uint<8>, 4>, ap_uint<8>,
                                        std::array<ap_uint<8>, 2>, ap_uint<8>,
                                        4, 4, 4, 4, 4, 2, 2>();

  if (!all_passed) {
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  return all_passed ? 0 : 1;
}