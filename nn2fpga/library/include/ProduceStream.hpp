#pragma once
#include "ap_int.h"
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>

/**
 * @class ProduceStream
 * @brief ProduceStream is a templated class for streaming data into neural
 * network data, converting AXI streams to the nn2FPGA data format based on the
 * parallelism selected.
 *
 * This class reads AXI input data structures from an input stream, processes
 * them in parallel according to the specified template parameters, and writes
 * the results to output streams. The main point is to transform a AXI stream
 * into the nn2FPGA data format. In self-timed execution, it behaves like a
 * fixed throughput producer, producing data at a fixed rate defined by the
 * neural network's iteration interval, which means it does not flood the
 * network with data but produces it at the same rate as the neural network is
 * expected to consume it.
 *
 * @tparam TInputStruct    The structure type of the input data (AXI).
 * @tparam TInput          The type of the input data (AXI).
 * @tparam TOutputStruct   The structure type of the output data.
 * @tparam TOutput         The type of the output data.
 * @tparam Quantizer       The quantizer functor used for quantization.
 * @tparam DATA_PER_WORD   Number of data elements per input word.
 * @tparam BITS_PER_DATA   Number of bits per data element.
 * @tparam IN_HEIGHT       Input tensor height.
 * @tparam IN_WIDTH        Input tensor width.
 * @tparam OUT_CH          Number of output channels.
 * @tparam OUT_W_PAR       Output width parallelism (number of parallel outputs
 * in width).
 * @tparam OUT_CH_PAR      Output channel parallelism (number of parallel
 * outputs in channels).
 *
 * @note
 * - DATA_PER_WORD must be a multiple of OUT_CH_PAR * OUT_W_PAR.
 * - If OUT_W_PAR > 1, OUT_CH must be equal to OUT_CH_PAR, this is to preserve
 * the correct order of the data flowing.
 * - OUT_CH must be a multiple of OUT_CH_PAR.
 * - IN_WIDTH must be a multiple of OUT_W_PAR.
 *
 * @section Usage
 * - Use the run() method for functional verification and for synthesis.
 * - Use the step() and step_fixedthroughput() method for self-timed execution
 * with actor status tracking, which is needed for fifo depth estimation.
 *
 * @section Pipeline
 * The class manages pipeline depth and delayed output buffers to ensure correct
 * data propagation and timing in hardware pipelines. This information is only 
 * required for self-timed execution and is not used in hardware synthesis.
 *
 * @section Parallelism
 * The class supports parallel processing of output channels and width, as
 * specified by OUT_CH_PAR and OUT_W_PAR, respectively.
 *
 * @section Quantization
 * The Quantizer template parameter is used to quantize the extracted data
 * before writing to the output stream.
 */

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, typename Quantizer, size_t DATA_PER_WORD,
          size_t BITS_PER_DATA, size_t IN_HEIGHT, size_t IN_WIDTH,
          size_t OUT_CH, size_t OUT_W_PAR, size_t OUT_CH_PAR>
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

  ProduceStream()
      : ProduceStream(1, IN_HEIGHT * IN_WIDTH * OUT_CH /
                             (OUT_CH_PAR * OUT_W_PAR)) {}

  ProduceStream(size_t pipeline_depth, size_t nnII)
      : STEP_pipeline_depth(pipeline_depth), STEP_i_par(0), STEP_i_word(0),
        STEP_nnII(nnII),
        STEP_fixed_counter(nnII),
        STEP_actor_status(pipeline_depth, (IN_HEIGHT * IN_WIDTH * OUT_CH) /
                                              (OUT_CH_PAR * OUT_W_PAR)) {
    for (size_t i = 0; i < OUT_W_PAR; ++i) {
      STEP_delayed_output[i] =
          PipelineDelayBuffer<TOutputStruct>(pipeline_depth);
    }
  }

  void run(hls::stream<TInputStruct> &input_data_stream,
           hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {
    // Read the input data structure from the input stream.
    TInputStruct input_data;
    // Total number of iterations based on input height and width.
    const size_t ITER = IN_HEIGHT * IN_WIDTH * OUT_CH;

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

  ActorStatus step(hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {
    // Step the ProduceStream actor. It is composed by two different nodes:
    // 1. The produce node, which reads the input data stream and produces
    //    output data streams (the actual one).
    // 2. The fixed throughput node, which writes data at a fixed rate defined
    //    by the neural network iteration interval (nnII).

    ActorStatus actor_status =
        step_produce(STEP_input_data_stream, output_data_stream);

    step_fixedthroughput(STEP_input_data_stream);

    // Return the current actor status.
    return actor_status;

  }

private:

  // State variables for step execution
  size_t STEP_i_word = 0;         // Current word index
  size_t STEP_i_par = 0;          // Current parallel index
  size_t STEP_pipeline_depth = 1; // Pipeline depth
  size_t STEP_nnII = 1;           // Number of iterations for fixed throughput
  TInputStruct STEP_input_data;   // Input data structure for the current step
  hls::stream<TInputStruct> STEP_input_data_stream;

  // CSDFG state variables
  ActorStatus STEP_actor_status;
  PipelineDelayBuffer<TOutputStruct>
      STEP_delayed_output[OUT_W_PAR]; // Delayed output buffers to maintain
                                      // pipeline depth for each parallel output

  // Fixed throughput state variables
  size_t STEP_fixed_counter;

  ActorStatus
  step_produce(hls::stream<TInputStruct> &input_data_stream,
               hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR]) {

    const size_t ITER = IN_HEIGHT * IN_WIDTH * OUT_CH;

    bool firing_condition = true;
    if (STEP_i_par == 0 && input_data_stream.empty()) {
      firing_condition = false;
    }

    if (firing_condition) {
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
    } else {

      for (size_t i = 0; i < OUT_W_PAR; ++i) {
        STEP_delayed_output[i].push(TOutputStruct(), false);
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

  void step_fixedthroughput(hls::stream<TInputStruct> &output_data_stream) {
    // Write data with a specific throughput, defined by the ratio between
    // the number of data to produce and the neural network iteration
    // interval (nnII).

    // Increment the step counter by the number of data to produce.
    size_t words_per_tensor = IN_HEIGHT * IN_WIDTH * OUT_CH / DATA_PER_WORD;
    if (STEP_fixed_counter >= STEP_nnII) {
      // If we have reached the bottleneck cycles, reset the counter.
      STEP_fixed_counter -= STEP_nnII;

      // Write a token.
      TInputStruct output_data;
      output_data.data = 0;
      output_data_stream.write(output_data);
    }
    STEP_fixed_counter += words_per_tensor;
  }

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