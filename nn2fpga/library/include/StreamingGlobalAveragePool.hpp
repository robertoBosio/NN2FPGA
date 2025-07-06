#pragma once
#include <cstddef>
#include "hls_stream.h"

/*
 * StreamingGlobalAveragePool implements a global average pooling operation done in a streaming fashion.
 * Data in input is in HWC format, thus an accumulator is used to sum the values across the height and width dimensions for each channel.
 */

template <typename TInputStruct,
          typename TInput,
          typename TOutputStruct,
          typename TOutput,
          typename TAcc,
          typename TDiv,
          typename Quantizer,
          size_t IN_HEIGHT,
          size_t IN_WIDTH,
          size_t OUT_CH,
          size_t OUT_CH_PAR>
class StreamingGlobalAveragePool
{
public:
    static_assert(OUT_CH % OUT_CH_PAR == 0, "OUT_CH must be a multiple of OUT_CH_PAR");
    static_assert(OUT_CH_PAR > 0, "OUT_CH_PAR must be greater than 0");

    StreamingGlobalAveragePool()
    {
        STEP_i_hw = 0;  // Initialize the height and width index to zero.
        STEP_i_och = 0; // Initialize the output channel index to zero.
    }

    void run(hls::stream<TInputStruct> i_data[1],
             hls::stream<TOutputStruct> o_data[1])
    {
        TAcc s_acc_buff[OUT_CH]; // Accumulator buffer for each output channel.

        // Loop through the input height and width.
        for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH; i_hw++)
        {
            // Loop through the output channels, with a step size equal to the number of channels processed in parallel.
            for (size_t i_och = 0; i_och < OUT_CH; i_och += OUT_CH_PAR)
            {
#pragma HLS pipeline style = stp II = 1
                StreamingGlobalAveragePool::pipeline_body(i_data, o_data, s_acc_buff, i_hw, i_och);
            }
        }
    }

    bool step(hls::stream<TInputStruct> i_data[1],
              hls::stream<TOutputStruct> o_data[1])
    {
        if (STEP_i_hw >= IN_HEIGHT * IN_WIDTH)
        {
            // If we have processed all height and width, return false to indicate no more data.
            return false;
        }
        if (i_data[0].empty())
        {
            // If there is no data in the input stream, return false.
            return false;
        }
        StreamingGlobalAveragePool::pipeline_body(i_data, o_data, STEP_s_acc_buff, STEP_i_hw, STEP_i_och);
        STEP_i_och += OUT_CH_PAR;
        if (STEP_i_och >= OUT_CH)
        {
            // If we have processed all output channels, reset the index and increment the height/width index.
            STEP_i_och = 0;
            STEP_i_hw++;
        }

        return true; // Indicate that the step was successful and more data can be processed.
    }

private:
    // State variables for step execution
    TAcc STEP_s_acc_buff[OUT_CH]; // Accumulator buffer for each output channel
    size_t STEP_i_hw;             // Current height and width index
    size_t STEP_i_och;            // Current output channel index

    static void pipeline_body(
        hls::stream<TInputStruct> i_data[1],
        hls::stream<TOutputStruct> o_data[1],
        TAcc s_acc_buff[OUT_CH],
        size_t i_hw,
        size_t i_och)
    {
#pragma HLS inline
        TOutputStruct s_output_struct; // Output structure to hold the results.
        TInputStruct s_input_struct;   // Input structure to read data from the input stream.
        Quantizer quantizer;           // Quantizer instance for quantization.

        // Loop through the channels processed in parallel.
        for (size_t i_och_par = 0; i_och_par < OUT_CH_PAR; i_och_par++)
        {
            unsigned int current_och = i_och + i_och_par; // Current output channel index.

            // Initializing the accumulator for each window.
            if (i_hw == 0)
            {
                s_acc_buff[current_och] = 0;
            }

            // Reading packets of OUT_CH_PAR channels.
            if (i_och_par == 0)
            {
                s_input_struct = i_data[0].read();
            }

            // Accumulating the input data for the current output channel.
            s_acc_buff[current_och] += s_input_struct[i_och_par];

            // Writing the output at the end of the window
            if (i_hw == (IN_HEIGHT * IN_WIDTH - 1))
            {
                TDiv divisor = IN_HEIGHT * IN_WIDTH; // Divisor for the average calculation.
                s_output_struct[i_och_par] = quantizer(s_acc_buff[current_och] / divisor);
                if (i_och_par == (OUT_CH_PAR - 1))
                {
                    o_data[0].write(s_output_struct);
                }
            }
        }
    }
};