#include "ConsumeStream.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include <array>
#include <cassert>
#include <iostream>

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, size_t DATA_PER_WORD, size_t BITS_PER_DATA,
          size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH, size_t IN_W_PAR,
          size_t IN_CH_PAR>
bool test_run() {
  // This function tests the run() method of ConsumeStream
  // with a simple input where each pixel has it's position in the HWC format.

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(ap_uint<8> data) const {
      return static_cast<TOutput>(data);
    }
  };

  // Instantiate the operator
  ConsumeStream<TInputStruct, TInput, TOutputStruct, TOutput, TruncQuantizer,
                DATA_PER_WORD, BITS_PER_DATA, IN_HEIGHT, IN_WIDTH, IN_CH,
                IN_W_PAR, IN_CH_PAR>
      consumer;

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[IN_W_PAR];
  hls::stream<TOutputStruct> out_stream;

  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += IN_W_PAR) {
    for (size_t c = 0; c < IN_CH; c += IN_CH_PAR) {
      for (size_t w_par = 0; w_par < IN_W_PAR; w_par++) {
        TInputStruct input_struct;
        for (size_t c_par = 0; c_par < IN_CH_PAR; c_par++) {
          // Each channel should have the average value of 1
          input_struct[c_par] = (i + w_par) * IN_CH + c + c_par;
        }
        in_stream[w_par].write(input_struct);
      }
    }
  }

  // Run consumer
  consumer.run(in_stream, out_stream);

  // Read and check output
  bool flag = true;
  size_t data_in_word = 0;
  TOutputStruct output_struct;
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i++) {
    for (size_t c = 0; c < IN_CH; c++) {
      if (data_in_word == 0) {
        // Read the output structure from the stream
        output_struct = out_stream.read();
      }

      // Fill the input structure with the value 1 for each channel
      flag &= (output_struct.data.range(BITS_PER_DATA * (data_in_word + 1) - 1,
                                        BITS_PER_DATA * data_in_word) ==
               i * IN_CH + c);
      data_in_word++;

      if (data_in_word >= DATA_PER_WORD) {
        data_in_word = 0;
      }
    }
  }

  return flag;
}

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, size_t DATA_PER_WORD, size_t BITS_PER_DATA,
          size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH, size_t IN_W_PAR,
          size_t IN_CH_PAR>
bool test_step() {
  // This function tests the step() method of ConsumeStream
  // with a simple square input where each pixel has it's position in the HWC
  // format.

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(ap_uint<8> data) const {
      return static_cast<TOutput>(data);
    }
  };

  // Instantiate the operator
  ConsumeStream<TInputStruct, TInput, TOutputStruct, TOutput, TruncQuantizer,
                DATA_PER_WORD, BITS_PER_DATA, IN_HEIGHT, IN_WIDTH, IN_CH,
                IN_W_PAR, IN_CH_PAR>
      consumer;

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[IN_W_PAR];
  hls::stream<TOutputStruct> out_stream;

  // Check step function not progressing before any input
  ActorStatus actor_status = consumer.step(in_stream, out_stream);
  bool flag = actor_status.empty() && actor_status.get_current_index() == 0;

  // Prepare input data: fill every pixel with a counter following HWC format
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += IN_W_PAR) {
    for (size_t c = 0; c < IN_CH; c += IN_CH_PAR) {
      for (size_t w_par = 0; w_par < IN_W_PAR; w_par++) {
        TInputStruct input_struct;
        for (size_t c_par = 0; c_par < IN_CH_PAR; c_par++) {
          // Each channel should have the average value of 1
          input_struct[c_par] = (i + w_par) * IN_CH + c + c_par;
        }
        in_stream[w_par].write(input_struct);
      }
    }
  }

  // Step through consumer
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH * IN_CH / (IN_CH_PAR * IN_W_PAR);
       i++) {
    actor_status = consumer.step(in_stream, out_stream);
  }

  // Empty the output streams
  TOutputStruct output_struct;
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH * IN_CH / DATA_PER_WORD; i++) {
    // Read the output structure from the stream
    output_struct = out_stream.read();
  }

  // Check step function not progressing after all input
  actor_status = consumer.step(in_stream, out_stream);
  flag &= actor_status.empty() && actor_status.get_current_index() == 0;

  return flag;
}

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, size_t DATA_PER_WORD, size_t BITS_PER_DATA,
          size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH, size_t IN_W_PAR,
          size_t IN_CH_PAR>
bool test_step_pipelined(size_t PIPELINE_DEPTH) {
  // This function tests the step() method of ConsumeStream
  // with a simple square input where each pixel has its position in the HWC
  // format, and checks the pipeline behaviour.

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(ap_uint<8> data) const {
      return static_cast<TOutput>(data);
    }
  };

  // Instantiate the operator
  ConsumeStream<TInputStruct, TInput, TOutputStruct, TOutput, TruncQuantizer,
                DATA_PER_WORD, BITS_PER_DATA, IN_HEIGHT, IN_WIDTH, IN_CH,
                IN_W_PAR, IN_CH_PAR>
      consumer(PIPELINE_DEPTH);

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[IN_W_PAR];
  hls::stream<TOutputStruct> out_stream;

  // Check step function not progressing before any input
  ActorStatus actor_status = consumer.step(in_stream, out_stream);
  bool flag = actor_status.empty() && actor_status.get_current_index() == 0;

  for (size_t i = 0; i < DATA_PER_WORD; i += IN_CH_PAR * IN_W_PAR) {
    for (size_t c = 0; c < IN_CH; c += IN_CH_PAR) {
      for (size_t w_par = 0; w_par < IN_W_PAR; w_par++) {
        TInputStruct input_struct;
        for (size_t c_par = 0; c_par < IN_CH_PAR; c_par++) {
          // Each channel should have the average value of 1
          input_struct[c_par] = (i + w_par) * IN_CH + c + c_par;
        }
        in_stream[w_par].write(input_struct);

        // Check that the actor does not fire before all the input is ready
        if (IN_W_PAR > 1 && w_par != IN_W_PAR - 1) {
          actor_status = consumer.step(in_stream, out_stream);
          flag &= actor_status.get_current_index() == 0;
        }
      }
    }
  }

  // Execute the step function until an output is produced
  for (size_t i = 0; i < (DATA_PER_WORD / (IN_CH_PAR * IN_W_PAR)) - 1; i++) {
    actor_status = consumer.step(in_stream, out_stream);
    flag &= out_stream.empty(); // Should not have output yet
    flag &= actor_status.get_current_index() == i + 1;
  }

  for (size_t i = 0; i < PIPELINE_DEPTH - 1; i++) {
    actor_status = consumer.step(in_stream, out_stream);
    flag &= out_stream.empty(); // Should still not have output
  }
  actor_status = consumer.step(in_stream, out_stream);
  flag &= !out_stream.empty(); // Should have output

  // Flush the output streams
  TOutputStruct output_struct;
  output_struct = out_stream.read();

  return flag;
}

int main() {

  bool all_passed = true;

  // Test 4x4x4 input with 128-bit data width
  // and no padding, 2 channels processed in parallel.
  all_passed &=
      test_run<std::array<ap_uint<8>, 2>, ap_uint<8>, ap_axiu<128, 0, 0, 0>,
               ap_uint<128>, 16, 8, 4, 4, 4, 1, 2>();

  // Test 4x4x3 input with 64-bit data width
  // with padding, 2 channels and 2 pixels processed in parallel.
  all_passed &=
      test_run<std::array<ap_uint<8>, 3>, ap_uint<8>, ap_axiu<64, 0, 0, 0>,
               ap_uint<64>, 6, 8, 4, 4, 3, 2, 3>();

  // Test step function with 4x4x12 input with 64-bit data width
  //  with padding, 6 channels and 1 pixels processed in parallel.
  //  channels of the tensor not fitting in a single word.
  all_passed &=
      test_run<std::array<ap_uint<8>, 6>, ap_uint<8>, ap_axiu<64, 0, 0, 0>,
               ap_uint<64>, 6, 8, 4, 4, 12, 1, 6>();

  // Test step function with 128-bit data width
  // and no padding, 2 channels processed in parallel.
  all_passed &=
      test_step<std::array<ap_uint<8>, 2>, ap_uint<8>, ap_axiu<128, 0, 0, 0>,
                ap_uint<128>, 16, 8, 4, 4, 4, 1, 2>();

  // Test step function with 64-bit data width
  // and padding, 2 channels and 2 pixels processed in parallel.
  all_passed &=
      test_step<std::array<ap_uint<8>, 3>, ap_uint<8>, ap_axiu<64, 0, 0, 0>,
                ap_uint<64>, 6, 8, 4, 4, 3, 2, 3>();

  // Test step function with 64-bit data width
  // with padding, 6 channels and 1 pixel processed in parallel,
  // channels of the tensor not fitting in a single word, pipelined.
  all_passed &= test_step_pipelined<std::array<ap_uint<8>, 6>, ap_uint<8>,
                                    ap_axiu<64, 0, 0, 0>, ap_uint<64>, 6, 8, 4,
                                    4, 12, 1, 6>(4);

  // Test step function with 64-bit data width
  // with padding, 3 channels and 2 pixels processed in parallel,
  // pipelined.
  all_passed &= test_step_pipelined<std::array<ap_uint<8>, 3>, ap_uint<8>,
                                    ap_axiu<64, 0, 0, 0>, ap_uint<64>, 6, 8, 4,
                                    4, 3, 2, 3>(5);

  if (!all_passed) {
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  return all_passed ? 0 : 1;
}
