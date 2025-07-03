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
          size_t OUT_CH,
          size_t OUT_W_PAR,
          size_t OUT_CH_PAR>
class ProduceStream
{
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
    {
        STEP_i_par = 0; // Initialize the parallel index to zero.
        STEP_i_word = 0; // Initialize the word index to zero.
    }

    void run(hls::stream<TInputStruct> input_data_stream[1],
             hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR])
    {
        TInputStruct input_data; // Read the input data structure from the input stream.

        // Loop through the input height and width.
        for (size_t i_word = 0; i_word < ITER; i_word += DATA_PER_WORD)
        {
            // Loop through the output channels, with a step size equal to the number of channels processed in parallel.
            for (size_t i_par = 0; i_par < DATA_PER_WORD; i_par += OUT_CH_PAR * OUT_W_PAR)
            {
#pragma HLS pipeline style = stp II = 1
                ProduceStream::pipeline_body(input_data_stream, output_data_stream, input_data, i_par);
            }
        }
    }

    bool step(hls::stream<TInputStruct> input_data_stream[1],
              hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR])
    {
        if (STEP_i_word >= ITER)
        {
            // If we have processed all data, return false to indicate no more data.
            return false;
        }
        if (STEP_i_par == 0 && input_data_stream[0].empty())
        {
            // If there is no data in the input stream, return false.
            return false;
        }
        ProduceStream::pipeline_body(input_data_stream, output_data_stream, STEP_input_data, STEP_i_par);
        STEP_i_par += OUT_CH_PAR * OUT_W_PAR;
        if (STEP_i_par >= DATA_PER_WORD)
        {
            // If we have processed all output channels, reset the index and increment the height/width index.
            STEP_i_par = 0;
            STEP_i_word += DATA_PER_WORD;
        }

        return true; // Indicate that the step was successful and more data can be processed.
    }

private:
        
    const size_t ITER = IN_HEIGHT * IN_WIDTH * OUT_CH; // Total number of iterations based on input height and width. 
    
    // State variables for step execution
    size_t STEP_i_word = 0;       // Current word index
    size_t STEP_i_par = 0;            // Current parallel index
    TInputStruct STEP_input_data; // Input data structure for the current step

    static void pipeline_body(
        hls::stream<TInputStruct> input_data_stream[1],
        hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
        TInputStruct &input_data,
        size_t i_par)
    {
#pragma HLS inline
        Quantizer quantizer; // Quantizer instance for quantization.
        
        if (i_par == 0)
        {
            // Read the input data structure from the input stream.
            input_data = input_data_stream[0].read();
        }

        // Loop through the pixels processed in parallel.
        for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++)
        {
            TOutputStruct s_output_struct; // Output structure to hold the results.
            for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++)
            {
                ap_uint<BITS_PER_DATA> s_output_data = 0; // Output data for the current channel.

                // Extract the data for the current pixel output channel.
                s_output_data.range(BITS_PER_DATA - 1, 0) =
                    input_data.data.range(BITS_PER_DATA * ((i_w_par * OUT_CH_PAR) + i_och_par + i_par + 1) - 1,
                                          BITS_PER_DATA * ((i_w_par * OUT_CH_PAR) + i_och_par + i_par));

                s_output_struct[i_och_par] = quantizer(s_output_data);
            }
            // Write the output structure to the output stream.
            output_data_stream[i_w_par].write(s_output_struct);
        }
    }
};