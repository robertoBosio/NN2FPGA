#include <array>
#include <iostream>
#include <cassert>
#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ProduceStream.hpp"

template <typename TInputStruct,
          typename TInput,
          typename TOutputStruct,
          typename TOutput,
          size_t DATA_PER_WORD,
          size_t BITS_PER_DATA,
          size_t IN_HEIGHT,
          size_t IN_WIDTH,
          size_t OUT_CH,
          size_t OUT_W_PAR,
          size_t OUT_CH_PAR>
bool test_run() {
    // This function tests the run() method of ProduceStream
    // with a simple input where each pixel has it's position in the HWC format.

    // Simple quantizer: truncates accumulator
    struct TruncQuantizer
    {
        TOutput operator()(ap_uint<8> data) const
        {
            return static_cast<TOutput>(data);
        }
    };

    // Instantiate the operator
    ProduceStream<
        TInputStruct, TInput,
        TOutputStruct, TOutput,
        TruncQuantizer,
        DATA_PER_WORD, BITS_PER_DATA,
        IN_HEIGHT, IN_WIDTH,
        OUT_CH, OUT_W_PAR, OUT_CH_PAR>
        producer;

    // Prepare input and output streams
    hls::stream<TInputStruct> in_stream[1];
    hls::stream<TOutputStruct> out_stream[OUT_W_PAR];
    hls::stream<bool> out_last;

    // Prepare input data: fill every pixel with a counter following HWC format
    size_t data_in_word = 0;
    TInputStruct input_struct;
    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i++)
    {
        for (size_t c = 0; c < OUT_CH; c++)
        {
            // Fill the input structure with the value 1 for each channel
            input_struct.data.range(BITS_PER_DATA * (data_in_word + 1) - 1, BITS_PER_DATA * data_in_word) = i * OUT_CH + c;
            data_in_word++;
            
            if (data_in_word >= DATA_PER_WORD)
            {
                // If we have filled the current word, write it to the stream
                if (i == IN_HEIGHT * IN_WIDTH - 1 && c == OUT_CH - 1) {
                    // If this is the last input, set the last signal
                    input_struct.last = true;
                } else {
                    input_struct.last = false;
                }
                in_stream[0].write(input_struct);
                
                data_in_word = 0;
                input_struct.data = 0; // Reset for next word
            }
        }
    }

    // Run producer
    producer.run(in_stream, out_stream, out_last);

    // Read and check output
    bool flag = true;
    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += OUT_W_PAR)
    {
        for (size_t c = 0; c < OUT_CH; c += OUT_CH_PAR)
        {
            for (size_t w_par = 0; w_par < OUT_W_PAR; w_par++)
            {
                TOutputStruct output_struct = out_stream[w_par].read();
                for (size_t c_par = 0; c_par < OUT_CH_PAR; c_par++)
                {
                    // Each channel should have the average value of 1
                    flag &= (output_struct[c_par] == (i + w_par) * OUT_CH + c + c_par);
                }
            }
        }
    }

    // Check last
    flag &= (out_last.read() == true);
    return flag;
}

template <typename TInputStruct,
          typename TInput,
          typename TOutputStruct,
          typename TOutput,
          size_t DATA_PER_WORD,
          size_t BITS_PER_DATA,
          size_t IN_HEIGHT,
          size_t IN_WIDTH,
          size_t OUT_CH,
          size_t OUT_W_PAR,
          size_t OUT_CH_PAR>
bool test_step() {
    // This function tests the step() method of ProduceStream
    // with a simple square input where each pixel has it's position in the HWC format.

    // Simple quantizer: truncates accumulator
    struct TruncQuantizer
    {
        TOutput operator()(ap_uint<8> data) const
        {
            return static_cast<TOutput>(data);
        }
    };

    // Instantiate the operator
    ProduceStream<
        TInputStruct, TInput,
        TOutputStruct, TOutput,
        TruncQuantizer,
        DATA_PER_WORD, BITS_PER_DATA,
        IN_HEIGHT, IN_WIDTH,
        OUT_CH, OUT_W_PAR, OUT_CH_PAR>
        producer;

    // Prepare input and output streams
    hls::stream<TInputStruct> in_stream[1];
    hls::stream<TOutputStruct> out_stream[OUT_W_PAR];
    hls::stream<bool> out_last;

    // Check step function not progressing before any input
    bool flag = (producer.step(in_stream, out_stream, out_last) == false);

    // Prepare input data: fill every pixel with a counter following HWC format
    size_t data_in_word = 0;
    TInputStruct input_struct;
    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i++)
    {
        for (size_t c = 0; c < OUT_CH; c++)
        {
            // Fill the input structure with the value 1 for each channel
            input_struct.data.range(BITS_PER_DATA * (data_in_word + 1) - 1, BITS_PER_DATA * data_in_word) = i * OUT_CH + c;
            data_in_word++;
            
            if (data_in_word >= DATA_PER_WORD)
            {
                // If we have filled the current word, write it to the stream
                if (i == IN_HEIGHT * IN_WIDTH - 1 && c == OUT_CH - 1) {
                    // If this is the last input, set the last signal
                    input_struct.last = true;
                } else {
                    input_struct.last = false;
                }
                in_stream[0].write(input_struct);
                
                data_in_word = 0;
                input_struct.data = 0; // Reset for next word
            }
        }
    }

    // Step through producer
    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH * OUT_CH / (OUT_CH_PAR * OUT_W_PAR); i++) {
        flag &= producer.step(in_stream, out_stream, out_last);
    }
    
    // Check last
    flag &= (out_last.read() == true);

    // Check step function not progressing after all input
    flag &= (producer.step(in_stream, out_stream, out_last) == false);
    
    return flag;
}

int main() {

    bool all_passed = true;

    // Test square input with 128-bit data width
    // and no padding, 2 channels processed in parallel.
    all_passed &= test_run<
        ap_axiu<128, 0, 0, 0>, ap_uint<128>,
        std::array<ap_uint<8>, 2>, ap_uint<8>,
        16, 8, 4, 4, 4, 1, 2>();
    
    // Test square input with 64-bit data width
    // with padding, 2 channels and 2 pixels processed in parallel.
    all_passed &= test_run<
        ap_axiu<64, 0, 0, 0>, ap_uint<64>,
        std::array<ap_uint<8>, 3>, ap_uint<8>,
        6, 8, 4, 4, 3, 2, 3>();

    // Test step function with 128-bit data width
    // and no padding, 2 channels processed in parallel.
    all_passed &= test_step<
        ap_axiu<128, 0, 0, 0>, ap_uint<128>,
        std::array<ap_uint<8>, 2>, ap_uint<8>,
        16, 8, 4, 4, 4, 1, 2>();

    // Test step function with 64-bit data width
    // and padding, 2 channels and 2 pixels processed in parallel.
    all_passed &= test_step<
        ap_axiu<64, 0, 0, 0>, ap_uint<64>,
        std::array<ap_uint<8>, 3>, ap_uint<8>,
        6, 8, 4, 4, 3, 2, 3>();

    if (!all_passed) {
        std::cout << "Failed." << std::endl;
    } else {
        std::cout << "Passed." << std::endl;
    }

    return all_passed ? 0 : 1;
}
