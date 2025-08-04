#include "StreamingGlobalAveragePool.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include <array>
#include <cassert>
#include <iostream>

bool test_run_simple_square() {
  // This function tests the run() method of StreamingGlobalAveragePool
  // with a simple square input where each channel has the same value.
  // The expected output is the average of the input values.

  constexpr size_t IN_HEIGHT = 4;
  constexpr size_t IN_WIDTH = 4;
  constexpr size_t OUT_CH = 4;
  constexpr size_t OUT_CH_PAR = 2;

  using TInput = ap_uint<8>;
  using TOutput = ap_uint<8>;
  using TAcc = ap_int<32>;
  using TDiv = ap_uint<16>;

  using TInputStruct = std::array<TInput, OUT_CH_PAR>;
  using TOutputStruct = std::array<TOutput, OUT_CH_PAR>;

  // Simple quantizer: truncates accumulator
  struct TruncQuantizer {
    TOutput operator()(TAcc acc) const { return static_cast<TOutput>(acc); }
  };

  // Instantiate the operator
  StreamingGlobalAveragePool<TInputStruct, TInput, TOutputStruct, TOutput, TAcc,
                             TDiv, TruncQuantizer, IN_HEIGHT, IN_WIDTH, OUT_CH,
                             OUT_CH_PAR>
      pool;

  // Prepare input and output streams
  hls::stream<TInputStruct> in_stream[1];
  hls::stream<TOutputStruct> out_stream[1];

  // Prepare input data: fill every channel with 1, expect sum = 4 (2x2 window)
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i++) {
    for (size_t c = 0; c < OUT_CH; c += OUT_CH_PAR) {
      TInputStruct input_struct;
      for (size_t j = 0; j < OUT_CH_PAR; j++) {
        input_struct[j] = 1; // Fill each channel with value 1
      }
      in_stream[0].write(input_struct);
    }
  }

  // Run pooling
  pool.run(in_stream, out_stream);

  // Read and check output
  bool flag = true;
  for (size_t c = 0; c < OUT_CH; c += OUT_CH_PAR) {
    TOutputStruct output_struct = out_stream[0].read();
    for (size_t j = 0; j < OUT_CH_PAR; j++) {
      // Each channel should have the average value of 1
      flag &= (output_struct[j] == 1);
    }
  }

  return flag;
}

bool test_step_simple_square(size_t PIPELINE_DEPTH = 1) {
  // This function tests the step() method of StreamingGlobalAveragePool
  // with a simple square input where each channel has the same value.
  // The expected output is the average of the input values.

  constexpr size_t IN_HEIGHT = 4;
  constexpr size_t IN_WIDTH = 4;
  constexpr size_t OUT_CH = 4;
  constexpr size_t OUT_CH_PAR = 2;

  using TInput = ap_uint<8>;
  using TOutput = ap_uint<8>;
  using TAcc = ap_int<32>;
  using TDiv = ap_uint<16>;

  using TInputStruct = std::array<TInput, OUT_CH_PAR>;
  using TOutputStruct = std::array<TOutput, OUT_CH_PAR>;

  hls::stream<TInputStruct> in_stream[1];
  hls::stream<TOutputStruct> out_stream[1];

  // Simple quantizer: truncates accumulator.
  struct TruncQuantizer {
    TOutput operator()(TAcc acc) const { return static_cast<TOutput>(acc); }
  };

  // Instantiate the operator
  StreamingGlobalAveragePool<TInputStruct, TInput, TOutputStruct, TOutput, TAcc,
                             TDiv, TruncQuantizer, IN_HEIGHT, IN_WIDTH, OUT_CH,
                             OUT_CH_PAR>
      pool(PIPELINE_DEPTH);

  // Check step function not progressing before any input
  ActorStatus actor_status = pool.step(in_stream, out_stream);
  bool flag = actor_status.empty() && actor_status.get_current_index() == 0;

  // Prepare input data.
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i++) {
    for (size_t c = 0; c < OUT_CH; c += OUT_CH_PAR) {
      TInputStruct input_struct;
      for (size_t j = 0; j < OUT_CH_PAR; j++) {
        input_struct[j] = 1; // Fill each channel with value 1
      }
      in_stream[0].write(input_struct);
    }
  }

  // Step through pooling
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH * OUT_CH / OUT_CH_PAR; i++) {
    actor_status = pool.step(in_stream, out_stream);
  }

  // Check step function restarted correctly
  actor_status = pool.step(in_stream, out_stream);
  flag &= actor_status.empty() && actor_status.get_current_index() == 0;

  // Flush the output stream and check results.
  TOutputStruct output_struct;
  while (out_stream[0].read_nb(output_struct)) {
    for (size_t j = 0; j < OUT_CH_PAR; j++) {
      // Each channel should have the average value of 1
      flag &= (output_struct[j] == 1);
    }
  }

  return flag;
}

bool test_step_simple_square_pipelined() {
  // This function tests the step() method of StreamingGlobalAveragePool
  // with a simple square input where each channel has the same value.
  // The expected output is the average of the input values. The operator
  // is pipelined, so the output is delayed by PIPELINE_DEPTH cycles.

  constexpr size_t IN_HEIGHT = 4;
  constexpr size_t IN_WIDTH = 4;
  constexpr size_t OUT_CH = 4;
  constexpr size_t OUT_CH_PAR = 2;
  constexpr size_t PIPELINE_DEPTH = 4;

  using TInput = ap_uint<8>;
  using TOutput = ap_uint<8>;
  using TAcc = ap_int<32>;
  using TDiv = ap_uint<16>;

  using TInputStruct = std::array<TInput, OUT_CH_PAR>;
  using TOutputStruct = std::array<TOutput, OUT_CH_PAR>;

  hls::stream<TInputStruct> in_stream[1];
  hls::stream<TOutputStruct> out_stream[1];

  // Simple quantizer: truncates accumulator.
  struct TruncQuantizer {
    TOutput operator()(TAcc acc) const { return static_cast<TOutput>(acc); }
  };

  StreamingGlobalAveragePool<TInputStruct, TInput, TOutputStruct, TOutput, TAcc,
                             TDiv, TruncQuantizer, IN_HEIGHT, IN_WIDTH, OUT_CH,
                             OUT_CH_PAR>
      pool_with_delay(PIPELINE_DEPTH);

  // Check step function not progressing before any input
  ActorStatus actor_status =
      pool_with_delay.step(in_stream, out_stream);
  bool flag = actor_status.empty() && actor_status.get_current_index() == 0;

  std::cout << "Check step function not progressing before any input: " << flag << std::endl;

  // Prepare input data.
  for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i++) {
    for (size_t c = 0; c < OUT_CH; c += OUT_CH_PAR) {
      TInputStruct input_struct;
      for (size_t j = 0; j < OUT_CH_PAR; j++) {
        input_struct[j] = 1; // Fill each channel with value 1
      }
      in_stream[0].write(input_struct);
    }
  }

  // Check behaviour at the start of the pipeline
  for (size_t i = 1; i < PIPELINE_DEPTH; i++) {
    actor_status = pool_with_delay.step(in_stream, out_stream);
    flag &= actor_status.size() == i;
    flag &= actor_status.get_current_index() == i;
    flag &= out_stream[0].empty(); // No output yet
  }

  std::cout << "Check behaviour at the start of the pipeline: " << flag << std::endl;

  // Step through pooling
  for (size_t i = PIPELINE_DEPTH;
       i < IN_HEIGHT * IN_WIDTH * OUT_CH / OUT_CH_PAR; i++) {
    actor_status = pool_with_delay.step(in_stream, out_stream);
    flag &= actor_status.size() == PIPELINE_DEPTH - 1;
    flag &= out_stream[0].empty();
  }

  std::cout << "Step through pooling: " << flag << std::endl;

  // Check behaviour at the end of the pipeline
  actor_status = pool_with_delay.step(in_stream, out_stream);
  flag &= actor_status.size() == PIPELINE_DEPTH - 1;
  flag &= actor_status.get_current_index() == 0; // Should have completed one full iteration
  flag &= out_stream[0].empty();    // Should still not have output yet.

  std::cout << "Check behaviour at the end of the pipeline: " << flag << std::endl;

  actor_status = pool_with_delay.step(in_stream, out_stream);
  flag &= actor_status.size() == PIPELINE_DEPTH - 2;
  flag &= actor_status.get_current_index() == 0; // Should have completed one full iteration
  flag &= out_stream[0].empty();    // Should still not have output yet.

  std::cout << "Check behaviour at the end of the pipeline after two steps: " << flag << std::endl;

  actor_status = pool_with_delay.step(in_stream, out_stream);
  flag &= actor_status.size() == 1;
  flag &= actor_status.get_current_index() == 0; // Should have completed one full iteration
  flag &= !out_stream[0].empty();   // First output should be available now
  TOutputStruct output_struct = out_stream[0].read();

  std::cout << "Check behaviour at the end of the pipeline after three steps: " << flag << std::endl;

  actor_status = pool_with_delay.step(in_stream, out_stream);
  flag &= actor_status.size() == 0;
  flag &= actor_status.get_current_index() == 0; // Should have completed one full iteration
  flag &= !out_stream[0].empty();   // Second output should be available now
  output_struct = out_stream[0].read();

  std::cout << "Check behaviour at the end of the pipeline after four steps: " << flag << std::endl;

  // Check step function after processing all but the last input
  actor_status = pool_with_delay.step(in_stream, out_stream);
  flag &= out_stream[0].empty(); // Should not have output now

  std::cout << "Check step function after processing all but the last input: " << flag << std::endl;
  // Loop through the remaining inputs.
  for (size_t i = 0; i < (OUT_CH / OUT_CH_PAR) - 1; i++) {
    actor_status = pool_with_delay.step(in_stream, out_stream);
  }

  // Check step function restarted correctly
  actor_status = pool_with_delay.step(in_stream, out_stream);
  flag &= actor_status.empty() && actor_status.get_current_index() == 0;

  // Flush the output stream and check results.
  while (out_stream[0].read_nb(output_struct)) {
    for (size_t j = 0; j < OUT_CH_PAR; j++) {
      // Each channel should have the average value of 1
      flag &= (output_struct[j] == 1);
    }
  }

  return flag;
}

int main() {

  bool all_passed = true;

  all_passed &= test_run_simple_square();
  all_passed &= test_step_simple_square();
  all_passed &= test_step_simple_square_pipelined();
  if (!all_passed) {
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  return all_passed ? 0 : 1;
}
