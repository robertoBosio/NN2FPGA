#pragma once
#include "ap_int.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, typename Quantizer, size_t DATA_PER_WORD,
          size_t BITS_PER_DATA, size_t IN_HEIGHT, size_t IN_WIDTH, size_t IN_CH,
          size_t IN_W_PAR, size_t IN_CH_PAR, size_t PIPELINE_DEPTH = 1>
class ConsumeStream {
public:
  static_assert(DATA_PER_WORD % (IN_W_PAR * IN_CH_PAR) == 0,
                "DATA_PER_WORD must be a multiple of IN_CH_PAR * IN_W_PAR");
  static_assert(IN_W_PAR == 1 || IN_CH == IN_CH_PAR,
                "IN_CH must be equal to IN_CH_PAR when IN_W_PAR > 1");
  static_assert(IN_CH % IN_CH_PAR == 0,
                "IN_CH must be a multiple of IN_CH_PAR");
  static_assert(IN_WIDTH % IN_W_PAR == 0,
                "IN_WIDTH must be a multiple of IN_W_PAR");
  static_assert(DATA_PER_WORD * BITS_PER_DATA % 8 == 0,
                "DATA_PER_WORD * BITS_PER_DATA must be a multiple of a byte");

  ConsumeStream() {
    // Initialize the step state variables.
    STEP_output_data.data = 0; // Initialize the output data structure.
    STEP_output_data.keep =
        (1UL << ((DATA_PER_WORD * BITS_PER_DATA) >> 3)) -
        1; // Set the keep field for the useful bytes per word.
    STEP_output_data.strb =
        STEP_output_data.keep; // Set the strb field to the same value as keep.
    STEP_output_data.last = false; // Initialize the last signal to false.
    STEP_i_par = 0;                // Initialize the parallel index to zero.
    STEP_i_word = 0;               // Initialize the word index to zero.
  }

  void run(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
           hls::stream<TOutputStruct> &output_data_stream) {
    TOutputStruct output_data; // Output data structure to hold the results.

    // Loop through the input height and width.
    for (size_t i_word = 0; i_word < ITER; i_word += DATA_PER_WORD) {
      // Loop through the output channels, with a step size equal to the number
      // of channels processed in parallel.
      for (size_t i_par = 0; i_par < DATA_PER_WORD;
           i_par += IN_CH_PAR * IN_W_PAR) {
#pragma HLS pipeline style = stp II = 1
        ConsumeStream::pipeline_body(input_data_stream, output_data_stream,
                                     output_data, i_word, i_par);
      }
    }
  }

  ActorStatus step(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
                   hls::stream<TOutputStruct> &output_data_stream) {

    // Compute firing condition.
    bool firing_condition = true;
    for (size_t i_w_par = 0; i_w_par < IN_W_PAR; i_w_par++) {
      if (input_data_stream[i_w_par].empty()) {
        firing_condition = false;
      }
    }

    if (firing_condition) {
      hls::stream<TOutputStruct> instant_output_stream;
      ConsumeStream::pipeline_body(input_data_stream, instant_output_stream,
                                   STEP_output_data, STEP_i_word, STEP_i_par);
      STEP_i_par += IN_CH_PAR * IN_W_PAR;
      if (STEP_i_par >= DATA_PER_WORD) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        STEP_i_par = 0;
        STEP_i_word += DATA_PER_WORD;
      }
      if (STEP_i_word >= ITER) {
        STEP_i_word =
            0; // Reset the word index if we have processed all iterations.
      }

      STEP_actor_status.fire(); // Fire the actor status.

      // Add the output to the delayed output stream.
      if (!instant_output_stream.empty()) {
        STEP_delayed_output.push(instant_output_stream.read(), true);
      } else {
        STEP_delayed_output.push(TOutputStruct(),
                                 false); // Placeholder, ignored
      }
    } else {
      // If the firing condition is not met, push a placeholder to maintain the
      // pipeline depth.
      STEP_delayed_output.push(TOutputStruct(), false);
    }

    // Advance the actor status.
    STEP_actor_status.advance();

    // Write the output data to the output stream.
    TOutputStruct out;
    if (STEP_delayed_output.pop(out)) {
      output_data_stream.write(out);
    }

    return STEP_actor_status; // Return the current actor status.
  }

private:
  static const size_t ITER =
      IN_HEIGHT * IN_WIDTH *
      IN_CH; // Total number of iterations based on input height and width.

  // State variables for step execution
  size_t STEP_i_word;             // Current word index
  size_t STEP_i_par;              // Current parallel index
  TOutputStruct STEP_output_data; // Output data structure for the current step

  // CSDFG state variables
  ActorStatus STEP_actor_status{PIPELINE_DEPTH, ITER / (IN_CH_PAR * IN_W_PAR)};
  PipelineDelayBuffer<TOutputStruct>
      STEP_delayed_output{PIPELINE_DEPTH}; // Delayed output buffer

  static void
  pipeline_body(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
                hls::stream<TOutputStruct> &output_data_stream,
                TOutputStruct &output_data, size_t i_word, size_t i_par) {
#pragma HLS inline
    Quantizer quantizer; // Quantizer instance for quantization.

    // Loop through the pixels processed in parallel.
    for (size_t i_w_par = 0; i_w_par < IN_W_PAR; i_w_par++) {
      TInputStruct s_input_struct =
          input_data_stream[i_w_par]
              .read(); // Read the input data structure from the stream.
      for (size_t i_och_par = 0; i_och_par < IN_CH_PAR; i_och_par++) {
        // Calculate the starting bit index for the current output channel and
        // pixel. The formula maps each (i_w_par, i_och_par, i_par) to its
        // corresponding bit range in the output word.
        size_t bit_idx = (i_w_par * IN_CH_PAR) + i_och_par + i_par;
        // Write the data for the current pixel output channel.
        output_data.data.range(BITS_PER_DATA * (bit_idx + 1) - 1,
                               BITS_PER_DATA * bit_idx) =
            quantizer(s_input_struct[i_och_par]);
      }
    }

    if (i_par == DATA_PER_WORD - IN_CH_PAR * IN_W_PAR) {
      if (i_word == ITER - DATA_PER_WORD) {
        // If we are at the end of the tensor, assert the last.
        // In the future we could also think about setting the last at each
        // line.
        output_data.last = true;
      } else {
        // Otherwise, set the last signal to false.
        output_data.last = false;
      }

      // Write the output data structure to the output stream.
      output_data.keep = (1UL << ((DATA_PER_WORD * BITS_PER_DATA) >> 3)) -
                         1; // Set the keep field for the useful bytes per word.
      output_data.strb =
          output_data.keep; // Set the strb field to the same value as keep.
      output_data_stream.write(output_data);
    }
  }
};