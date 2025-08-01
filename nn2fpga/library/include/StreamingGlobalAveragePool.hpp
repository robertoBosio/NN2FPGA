#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>

/*
 * StreamingGlobalAveragePool implements a global average pooling operation done
 * in a streaming fashion. Data in input is in HWC format, thus an accumulator
 * is used to sum the values across the height and width dimensions for each
 * channel.
 */

template <typename TInputStruct, typename TInput, typename TOutputStruct,
          typename TOutput, typename TAcc, typename TDiv, typename Quantizer,
          size_t IN_HEIGHT, size_t IN_WIDTH, size_t OUT_CH, size_t OUT_CH_PAR,
          size_t PIPELINE_DEPTH = 1>
class StreamingGlobalAveragePool {
public:
  static_assert(OUT_CH % OUT_CH_PAR == 0,
                "OUT_CH must be a multiple of OUT_CH_PAR");
  static_assert(OUT_CH_PAR > 0, "OUT_CH_PAR must be greater than 0");
  static_assert(IN_HEIGHT > 0 && IN_WIDTH > 0,
                "IN_HEIGHT and IN_WIDTH must be greater than 0");
  static_assert(PIPELINE_DEPTH > 0, "PIPELINE_DEPTH must be greater than 0");

  StreamingGlobalAveragePool() {
    STEP_i_hw = 0;  // Initialize the height and width index to zero.
    STEP_i_och = 0; // Initialize the output channel index to zero.
  }

  void run(hls::stream<TInputStruct> i_data[1],
           hls::stream<TOutputStruct> o_data[1]) {
    TAcc s_acc_buff[OUT_CH]; // Accumulator buffer for each output channel.

    // Loop through the input height and width.
    for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH; i_hw++) {
      // Loop through the output channels, with a step size equal to the number
      // of channels processed in parallel.
      for (size_t i_och = 0; i_och < OUT_CH; i_och += OUT_CH_PAR) {
#pragma HLS pipeline style = stp II = 1
        StreamingGlobalAveragePool::pipeline_body(i_data, o_data, s_acc_buff,
                                                  i_hw, i_och);
      }
    }
  }

  ActorStatus
  step(hls::stream<TInputStruct> i_data[1],
       hls::stream<TOutputStruct> o_data[1]) {

    if (!i_data[0].empty()) {

      // If there is data in the input stream, process it.
      hls::stream<TOutputStruct> output_stream[1];
      StreamingGlobalAveragePool::pipeline_body(
          i_data, output_stream, STEP_s_acc_buff, STEP_i_hw, STEP_i_och);

      // Insert new firing status into the multiset.
      STEP_actor_status.fire();

      // Add the output to the delayed output stream.
      if (!output_stream[0].empty()) {
        STEP_delayed_output.push(output_stream[0].read(), true);
      } else {
        STEP_delayed_output.push(TOutputStruct(),
                                 false); // Placeholder, ignored
      }

      // Update the counters.
      STEP_i_och += OUT_CH_PAR;
      if (STEP_i_och >= OUT_CH) {
        // If we have processed all output channels, reset the index and
        // increment the height/width index.
        STEP_i_och = 0;
        STEP_i_hw++;
      }
      if (STEP_i_hw >= IN_HEIGHT * IN_WIDTH) {
        STEP_i_hw = 0; // Reset the height/width index if we have processed all
                       // iterations.
      }

    } else {
      // If there is no data in the input stream, push a delay slot.
      STEP_delayed_output.push(TOutputStruct(), false);
    }

    // Advance the state of the actor firings.
    STEP_actor_status.advance();

    // Write the output data to the output stream.
    TOutputStruct out;
    if (STEP_delayed_output.pop(out)) {
      o_data[0].write(out);
    }

    // Return the actor status.
    return STEP_actor_status;
  }

private:
  // State variables for step execution
  TAcc STEP_s_acc_buff[OUT_CH]; // Accumulator buffer for each output channel
  size_t STEP_i_hw;             // Current height and width index
  size_t STEP_i_och;            // Current output channel index

  // CSDFG actor state variables
  ActorStatus STEP_actor_status{PIPELINE_DEPTH, IN_HEIGHT * IN_WIDTH * OUT_CH / OUT_CH_PAR};

  // Pipeline state variables
  PipelineDelayBuffer<TOutputStruct> STEP_delayed_output{
      PIPELINE_DEPTH}; // Delayed output buffer to maintain pipeline depth

  static void pipeline_body(hls::stream<TInputStruct> i_data[1],
                            hls::stream<TOutputStruct> o_data[1],
                            TAcc s_acc_buff[OUT_CH], size_t i_hw,
                            size_t i_och) {
#pragma HLS inline
    TOutputStruct s_output_struct; // Output structure to hold the results.
    TInputStruct
        s_input_struct;  // Input structure to read data from the input stream.
    Quantizer quantizer; // Quantizer instance for quantization.

    // Loop through the channels processed in parallel.
    for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++) {
      unsigned int current_och =
          i_och + i_och_par; // Current output channel index.

      // Initializing the accumulator for each window.
      if (i_hw == 0) {
        s_acc_buff[current_och] = 0;
      }

      // Reading packets of OUT_CH_PAR channels.
      if (i_och_par == 0) {
        s_input_struct = i_data[0].read();
      }

      // Accumulating the input data for the current output channel.
      s_acc_buff[current_och] += s_input_struct[i_och_par];

      // Writing the output at the end of the window
      if (i_hw == (IN_HEIGHT * IN_WIDTH - 1)) {
        TDiv divisor =
            IN_HEIGHT * IN_WIDTH; // Divisor for the average calculation.

        // Round the accumulated value to the nearest integer.
        // This is not strictly correct, as ties should be rounded to the
        // nearest even number, but it requires the use of a modulo operation,
        // which is quite expensive. Instead, we are rounding ties up.
        TAcc rounded_value = s_acc_buff[current_och] + (divisor >> 1);
        TAcc result = rounded_value / divisor; // Calculate the average.

        // Potential logic for a rounding to the nearest even number
        // TAcc quotient = acc / div;
        // TDiv remainder = acc % div;
        // TDiv half = div >> 1;
        // if (remainder > half || (remainder == half && quotient[0])) {
        //   quotient += 1;
        // }
        // TAcc result = quotient;

        s_output_struct[i_och_par] = quantizer(result);
        if (i_och_par == (OUT_CH_PAR - 1)) {
          o_data[0].write(s_output_struct);
        }
      }
    }
  }
};