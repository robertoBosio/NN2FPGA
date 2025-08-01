#pragma once
#include "ap_int.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, typename Quantizer, size_t DATA_PER_WORD,
          size_t BITS_PER_DATA, size_t IN_HEIGHT, size_t IN_WIDTH,
          size_t OUT_CH, size_t OUT_W_PAR, size_t OUT_CH_PAR,
          size_t PIPELINE_DEPTH = 1>
class ProduceStream {
public:
  static_assert(DATA_PER_WORD % (OUT_CH_PAR * OUT_W_PAR) == 0,
                "DATA_PER_WORD must be a multiple of OUT_CH_PAR * OUT_W_PAR");
  static_assert(OUT_W_PAR == 1 || OUT_CH == OUT_CH_PAR,
                "OUT_CH must be equal to OUT_CH_PAR when OUT_W_PAR > 1");
  static_assert(OUT_CH % OUT_CH_PAR == 0,
                "OUT_CH must be a multiple of OUT_CH_PAR");
  static_assert(IN_WIDTH % OUT_W_PAR == 0,
                "IN_WIDTH must be a multiple of OUT_W_PAR");
  static_assert(PIPELINE_DEPTH > 0, "PIPELINE_DEPTH must be greater than 0");

  ProduceStream() {
    STEP_i_par = 0;  // Initialize the parallel index to zero.
    STEP_i_word = 0; // Initialize the word index to zero.
    for (size_t i = 0; i < OUT_W_PAR; ++i) {
      STEP_delayed_output[i] =
          PipelineDelayBuffer<TOutputStruct>(PIPELINE_DEPTH);
    }
  }

  void run(hls::stream<TInputStruct> &input_data_stream,
           hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {
    TInputStruct
        input_data; // Read the input data structure from the input stream.

    // Loop through the word packets of the input tensor.
    for (size_t i_word = 0; i_word < ITER; i_word += DATA_PER_WORD) {
      // Loop through the parallel data within each word packet.
      for (size_t i_par = 0; i_par < DATA_PER_WORD;
           i_par += OUT_CH_PAR * OUT_W_PAR) {
#pragma HLS pipeline style = stp II = 1
        ProduceStream::pipeline_body(input_data_stream, output_data_stream,
                                     input_data, i_par);
      }
    }
  }

  ActorStatus step(hls::stream<TInputStruct> &input_data_stream,
                   hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {

    if (STEP_i_par == 0 && input_data_stream.empty()) {
      for (size_t i = 0; i < OUT_W_PAR; ++i) {
        STEP_delayed_output[i].push(TOutputStruct(), false);
      }
    } else {
      hls::stream<TOutputStruct> output_stream[OUT_W_PAR];
      ProduceStream::pipeline_body(input_data_stream, output_stream,
                                   STEP_input_data, STEP_i_par);
      STEP_i_par += OUT_CH_PAR * OUT_W_PAR;
      if (STEP_i_par >= DATA_PER_WORD) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        STEP_i_par = 0;
        STEP_i_word += DATA_PER_WORD;
      }
      if (STEP_i_word >= ITER) {
        STEP_i_word = 0;
      }

      // Insert the firing status for the current step.
      STEP_actor_status.fire();

      // Add the output to the delayed output stream.
      for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
        if (!output_stream[i_w_par].empty()) {
          STEP_delayed_output[i_w_par].push(output_stream[i_w_par].read(),
                                            true);
        } else {
          STEP_delayed_output[i_w_par].push(TOutputStruct(),
                                            false); // Placeholder, ignored
        }
      }
    }

    // Advance the state of the actor firings.
    STEP_actor_status.advance();

    // Write the output data to the output stream.
    TOutputStruct out;
    for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
      if (STEP_delayed_output[i_w_par].pop(out)) {
        output_data_stream[i_w_par].write(out);
      }
    }

    // Return the current firing iteration index.
    return STEP_actor_status;
  }

private:
  const size_t ITER =
      IN_HEIGHT * IN_WIDTH *
      OUT_CH; // Total number of iterations based on input height and width.

  // State variables for step execution
  size_t STEP_i_word = 0;       // Current word index
  size_t STEP_i_par = 0;        // Current parallel index
  TInputStruct STEP_input_data; // Input data structure for the current step

  // CSDFG state variables
  ActorStatus STEP_actor_status{PIPELINE_DEPTH,
                                ITER / (OUT_CH_PAR * OUT_W_PAR)};
  PipelineDelayBuffer<TOutputStruct>
      STEP_delayed_output[OUT_W_PAR]; // Delayed output buffers to maintain
                                      // pipeline depth for each parallel output

  static void
  pipeline_body(hls::stream<TInputStruct> &input_data_stream,
                hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
                TInputStruct &input_data, size_t i_par) {
#pragma HLS inline
    Quantizer quantizer; // Quantizer instance for quantization.

    if (i_par == 0) {
      // Read the input data structure from the input stream.
      input_data = input_data_stream.read();
    }

    // Loop through the pixels processed in parallel.
    for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++) {
      TOutputStruct s_output_struct; // Output structure to hold the results.
      for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {
        ap_uint<BITS_PER_DATA> s_output_data =
            0; // Output data for the current channel.

        // Extract the data for the current pixel output channel.
        s_output_data.range(BITS_PER_DATA - 1, 0) = input_data.data.range(
            BITS_PER_DATA * ((i_w_par * OUT_CH_PAR) + i_och_par + i_par + 1) -
                1,
            BITS_PER_DATA * ((i_w_par * OUT_CH_PAR) + i_och_par + i_par));

        s_output_struct[i_och_par] = quantizer(s_output_data);
      }
      // Write the output structure to the output stream.
      output_data_stream[i_w_par].write(s_output_struct);
    }
  }
};