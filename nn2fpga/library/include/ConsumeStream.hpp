#pragma once
#include <cstddef>
#include "hls_stream.h"

template <typename TInputStruct,
          typename TInput,
          typename TOutputStruct,
          typename TOutput,
          typename Quantizer,
          size_t DATA_PER_WORD,
          size_t BITS_PER_DATA,
          size_t IN_HEIGHT,
          size_t IN_WIDTH,
          size_t IN_CH,
          size_t IN_W_PAR,
          size_t IN_CH_PAR>
class ConsumeStream
{
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
    
    ConsumeStream()
    {
#pragma HLS inline
        // Initialize the step state variables.
        STEP_output_data.data = 0; // Initialize the output data structure.
        STEP_output_data.keep = (1UL << ((DATA_PER_WORD * BITS_PER_DATA) >> 3)) - 1; // Set the keep field for the useful bytes per word. 
        STEP_output_data.strb = STEP_output_data.keep; // Set the strb field to the same value as keep.
        STEP_output_data.last = false; // Initialize the last signal to false.
        STEP_i_par = 0; // Initialize the parallel index to zero.
        STEP_i_word = 0; // Initialize the word index to zero.
    }

    void run(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
             hls::stream<bool> &input_last_stream,
             hls::stream<TOutputStruct> output_data_stream[1])
    {
        TOutputStruct output_data; // Output data structure to hold the results.

        // Loop through the input height and width.
        for (size_t i_word = 0; i_word < ITER; i_word += DATA_PER_WORD)
        {
            // Loop through the output channels, with a step size equal to the number of channels processed in parallel.
            for (size_t i_par = 0; i_par < DATA_PER_WORD; i_par += IN_CH_PAR * IN_W_PAR)
            {
#pragma HLS pipeline style = stp II = 1
                ConsumeStream::pipeline_body(input_data_stream, input_last_stream, output_data_stream, output_data, i_word, i_par);
            }
        }
    }

    bool step(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
              hls::stream<bool> &input_last_stream,
              hls::stream<TOutputStruct> output_data_stream[1])
    {
        if (STEP_i_word >= ITER)
        {
            // If we have processed all data, return false to indicate no more data.
            return false;
        }
        for (size_t i_w_par = 0; i_w_par < IN_W_PAR; i_w_par++)
        {
            if (input_data_stream[i_w_par].empty())
            {
                // If there is no data in the input stream, return false.
                return false;
            }
        }
        ConsumeStream::pipeline_body(input_data_stream, input_last_stream, output_data_stream, STEP_output_data, STEP_i_word, STEP_i_par);
        STEP_i_par += IN_CH_PAR * IN_W_PAR;
        if (STEP_i_par >= DATA_PER_WORD)
        {
            // If we have processed all output channels, reset the index and increment the height/width index.
            STEP_i_par = 0;
            STEP_i_word += DATA_PER_WORD;
        }

        return true; // Indicate that the step was successful and more data can be processed.
    }

private:
        
    static const size_t ITER = IN_HEIGHT * IN_WIDTH * IN_CH; // Total number of iterations based on input height and width.

    // State variables for step execution
    size_t STEP_i_word;             // Current word index
    size_t STEP_i_par;              // Current parallel index
    TOutputStruct STEP_output_data; // Output data structure for the current step

    static void pipeline_body(
        hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
        hls::stream<bool> &input_last_stream,
        hls::stream<TOutputStruct> output_data_stream[1],
        TOutputStruct &output_data,
        size_t i_word,
        size_t i_par)
    {
#pragma HLS inline
        Quantizer quantizer; // Quantizer instance for quantization.

        // Loop through the pixels processed in parallel.
        for (size_t i_w_par = 0; i_w_par < IN_W_PAR; i_w_par++)
        {
            TInputStruct s_input_struct = input_data_stream[i_w_par].read(); // Read the input data structure from the stream.
            for (size_t i_och_par = 0; i_och_par < IN_CH_PAR; i_och_par++)
            {
                // Calculate the starting bit index for the current output channel and pixel.
                // The formula maps each (i_w_par, i_och_par, i_par) to its corresponding bit range in the output word.
                size_t bit_idx = (i_w_par * IN_CH_PAR) + i_och_par + i_par;
                // Write the data for the current pixel output channel.
                output_data.data.range(BITS_PER_DATA * (bit_idx + 1) - 1,
                                       BITS_PER_DATA * bit_idx) = quantizer(s_input_struct[i_och_par]);

            }
        }

        if (i_par == DATA_PER_WORD - IN_CH_PAR * IN_W_PAR)
        {
            if (i_word == ITER - DATA_PER_WORD)
            {
                // If we are at the last word and parallel index, read the last signal.
                output_data.last = input_last_stream.read();
            } else
            {
                // Otherwise, set the last signal to false.
                output_data.last = false;
            }

            // Write the output data structure to the output stream.
            output_data_stream[0].write(output_data);
        }
    }
};