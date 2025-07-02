#pragma once
#include <cstddef>
#include "hls_stream.h"

template <typename TInputStruct,
          typename TInput,
          typename TOutputStruct,
          typename TOutput,
          typename Quantizer,
          size_t IN_HEIGHT,
          size_t IN_WIDTH,
          size_t IN_CH,
          size_t IN_W_PAR,
          size_t OUT_W_PAR,
          size_t CH_PAR>
class BandwidthAdjustIncreaseStreams
{
public:
    static_assert(IN_W_PAR > 0, "IN_W_PAR must be greater than 0");
    static_assert(OUT_W_PAR > 0, "OUT_W_PAR must be greater than 0");
    static_assert(IN_W_PAR < OUT_W_PAR, "IN_W_PAR must be less than OUT_W_PAR");
    static_assert(OUT_W_PAR % IN_W_PAR == 0, "OUT_W_PAR must be a multiple of IN_W_PAR");
    static_assert(IN_WIDTH % IN_W_PAR == 0, "IN_WIDTH must be a multiple of IN_W_PAR");
    static_assert(IN_WIDTH % OUT_W_PAR == 0, "IN_WIDTH must be a multiple of OUT_W_PAR");
    static_assert(IN_CH % CH_PAR == 0, "IN_CH must be a multiple of CH_PAR");

    BandwidthAdjustIncreaseStreams()
        : STEP_i_hw(0), STEP_i_out_stream(0), STEP_i_ch(0) // Initialize indices to zero.
    {
    }

    void run(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
             hls::stream<bool> &input_last_stream,
             hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
             hls::stream<bool> &output_last_stream)
    {
        for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH; i_hw += OUT_W_PAR)
        {
            for (size_t i_out_stream = 0; i_out_stream < OUT_W_PAR; i_out_stream += IN_W_PAR)
            {
                for (size_t i_ch = 0; i_ch < IN_CH; i_ch += CH_PAR)
                {
#pragma HLS pipeline style = stp II = 1
                    BandwidthAdjustIncreaseStreams::pipeline_body(input_data_stream, output_data_stream, i_out_stream);
                }
            }
        }
        output_last_stream.write(input_last_stream.read());
    }

    bool step(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
              hls::stream<bool> &input_last_stream,
              hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
              hls::stream<bool> &output_last_stream)
    {
        if (STEP_i_hw >= IN_HEIGHT * IN_WIDTH)
        {
            // If we have processed all height and width, return false to indicate no more data.
            return false;
        }
        for (size_t i_in_stream = 0; i_in_stream < IN_W_PAR; i_in_stream++)
        {
            if (input_data_stream[i_in_stream].empty())
            {
                // If there is no data in the input stream, return false.
                return false;
            }
        }
        BandwidthAdjustIncreaseStreams::pipeline_body(input_data_stream, output_data_stream, STEP_i_out_stream);
        STEP_i_ch += CH_PAR;
        if (STEP_i_ch >= IN_CH)
        {
            // If we have processed all input channels, reset the index and increment the height/width index.
            STEP_i_ch = 0;
            STEP_i_out_stream += IN_W_PAR;
        }
        if (STEP_i_out_stream >= OUT_W_PAR)
        {
            // If we have processed all output streams, reset the index and increment the height/width index.
            STEP_i_out_stream = 0;
            STEP_i_hw += OUT_W_PAR;
        }
        if (STEP_i_hw == IN_HEIGHT * IN_WIDTH && STEP_i_out_stream == 0 && STEP_i_ch == 0)
        {
            // If we have processed all height, width, and output streams, propagate the last signal.
            output_last_stream.write(input_last_stream.read());
        }
        return true; // Return true to indicate that there is more data to process.
    }

private:
    size_t STEP_i_hw;         // Index for height and width.
    size_t STEP_i_out_stream; // Index for output stream.
    size_t STEP_i_ch;         // Index for channels.

    static void pipeline_body(
        hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
        hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
        size_t i_out_stream)
    {
#pragma HLS inline
        TInputStruct s_input_struct;   // Input structure to read data from the input stream.
        TOutputStruct s_output_struct; // Output structure to hold the results.
        Quantizer quantizer;           // Quantizer instance for quantization.

        for (size_t i_w_par = 0; i_w_par < IN_W_PAR; i_w_par++)
        {
            // Read the input data structure from the input stream.
            s_input_struct = input_data_stream[i_w_par].read();

            for (size_t i_och_par = 0; i_och_par < CH_PAR; i_och_par++)
            {
                // Extract the data for the current pixel output channel.
                TInput s_input_data = s_input_struct[i_och_par];

                // Quantize the input data.
                TOutput s_output_data = quantizer(s_input_data);

                // Store the quantized data in the output structure.
                s_output_struct[i_och_par] = s_output_data;
            }
            output_data_stream[i_out_stream + i_w_par].write(s_output_struct);
        }
    }
};

template <typename TInputStruct,
          typename TInput,
          typename TOutputStruct,
          typename TOutput,
          typename Quantizer,
          size_t IN_HEIGHT,
          size_t IN_WIDTH,
          size_t IN_CH,
          size_t IN_W_PAR,
          size_t OUT_W_PAR,
          size_t CH_PAR>
class BandwidthAdjustDecreaseStreams
{
public:
    static_assert(IN_W_PAR > 0, "IN_W_PAR must be greater than 0");
    static_assert(OUT_W_PAR > 0, "OUT_W_PAR must be greater than 0");
    static_assert(IN_W_PAR > OUT_W_PAR, "OUT_W_PAR must be less than IN_W_PAR");
    static_assert(IN_W_PAR % OUT_W_PAR == 0, "OUT_W_PAR must be a multiple of IN_W_PAR");
    static_assert(IN_WIDTH % IN_W_PAR == 0, "IN_WIDTH must be a multiple of IN_W_PAR");
    static_assert(IN_WIDTH % OUT_W_PAR == 0, "IN_WIDTH must be a multiple of OUT_W_PAR");
    static_assert(IN_CH % CH_PAR == 0, "IN_CH must be a multiple of CH_PAR");

    BandwidthAdjustDecreaseStreams()
        : STEP_i_hw(0), STEP_i_in_stream(0), STEP_i_ch(0) // Initialize indices to zero.
    {
    }

    void run(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
             hls::stream<bool> &input_last_stream,
             hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
             hls::stream<bool> &output_last_stream)
    {
        for (size_t i_hw = 0; i_hw < IN_HEIGHT * IN_WIDTH; i_hw += IN_W_PAR)
        {
            for (size_t i_in_stream = 0; i_in_stream < IN_W_PAR; i_in_stream += OUT_W_PAR)
            {
                for (size_t i_ch = 0; i_ch < IN_CH; i_ch += CH_PAR)
                {
#pragma HLS pipeline style = stp II = 1
                    BandwidthAdjustDecreaseStreams::pipeline_body(input_data_stream, output_data_stream, i_in_stream);
                }
            }
        }
        output_last_stream.write(input_last_stream.read());
    }

    bool step(hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
              hls::stream<bool> &input_last_stream,
              hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
              hls::stream<bool> &output_last_stream)
    {
        // If we have processed all height and width, return false to indicate no more data.
        if (STEP_i_hw >= IN_HEIGHT * IN_WIDTH)
        {
            std::cout << "No more data to process." << std::endl;
            return false;
        }

        // Check if there is data in the input streams considered in this step.
        for (size_t i_in_stream = 0; i_in_stream < OUT_W_PAR; i_in_stream++)
        {
            if (input_data_stream[STEP_i_in_stream + i_in_stream].empty())
            {
                // If there is no data in the input stream, return false.
                std::cout << "Input stream " << i_in_stream << " is empty." << std::endl;
                return false;
            }
        }
        BandwidthAdjustDecreaseStreams::pipeline_body(input_data_stream, output_data_stream, STEP_i_in_stream);
        STEP_i_ch += CH_PAR;
        if (STEP_i_ch >= IN_CH)
        {
            // If we have processed all input channels, reset the index and increment the height/width index.
            STEP_i_ch = 0;
            STEP_i_in_stream += OUT_W_PAR;
        }
        if (STEP_i_in_stream >= IN_W_PAR)
        {
            // If we have processed all output streams, reset the index and increment the height/width index.
            STEP_i_in_stream = 0;
            STEP_i_hw += IN_W_PAR;
        }
        if (STEP_i_hw == IN_HEIGHT * IN_WIDTH && STEP_i_in_stream == 0 && STEP_i_ch == 0)
        {
            // If we have processed all height, width, and output streams, propagate the last signal.
            output_last_stream.write(input_last_stream.read());
            std::cout << "Last signal written." << std::endl;
        }
        std::cout << "Processed height/width: " << STEP_i_hw << ", output stream index: " << STEP_i_in_stream << ", channel index: " << STEP_i_ch << std::endl;
        return true; // Return true to indicate that there is more data to process.
    }

private:
    size_t STEP_i_hw;        // Index for height and width.
    size_t STEP_i_in_stream; // Index for input stream.
    size_t STEP_i_ch;        // Index for channels.

    static void pipeline_body(
        hls::stream<TInputStruct> input_data_stream[IN_W_PAR],
        hls::stream<TOutputStruct> output_data_stream[OUT_W_PAR],
        size_t i_in_stream)
    {
#pragma HLS inline
        TInputStruct s_input_struct;   // Input structure to read data from the input stream.
        TOutputStruct s_output_struct; // Output structure to hold the results.
        Quantizer quantizer;           // Quantizer instance for quantization.

        for (size_t i_w_par = 0; i_w_par < OUT_W_PAR; i_w_par++)
        {
            // Read the input data structure from the input stream.
            s_input_struct = input_data_stream[i_in_stream + i_w_par].read();

            for (size_t i_och_par = 0; i_och_par < CH_PAR; i_och_par++)
            {
                // Extract the data for the current pixel output channel.
                TInput s_input_data = s_input_struct[i_och_par];

                // Quantize the input data.
                TOutput s_output_data = quantizer(s_input_data);

                // Store the quantized data in the output structure.
                s_output_struct[i_och_par] = s_output_data;
            }
            output_data_stream[i_w_par].write(s_output_struct);
        }
    }
};