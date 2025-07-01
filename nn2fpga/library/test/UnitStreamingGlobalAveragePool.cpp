#include <array>
#include <iostream>
#include <cassert>
#include "ap_int.h"
#include "hls_stream.h"
#include "StreamingGlobalAveragePool.hpp"

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

    // Simple quantizer: truncates accumulator >> 2
    struct TruncQuantizer
    {
        TOutput operator()(TAcc acc) const
        {
            return static_cast<TOutput>(acc);
        }
    };

    // Instantiate the operator
    StreamingGlobalAveragePool<
        TInputStruct, TInput,
        TOutputStruct, TOutput,
        TAcc, TDiv,
        TruncQuantizer,
        IN_HEIGHT, IN_WIDTH,
        OUT_CH, OUT_CH_PAR
    > pool;

    // Prepare input and output streams
    hls::stream<TInputStruct> in_stream[1];
    hls::stream<TOutputStruct> out_stream[1];
    hls::stream<bool> in_last;
    hls::stream<bool> out_last;

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
    in_last.write(true);

    // Run pooling
    pool.run(in_stream, in_last, out_stream, out_last);

    // Read and check output
    bool flag = true;
    for (size_t c = 0; c < OUT_CH; c += OUT_CH_PAR) {
        TOutputStruct output_struct = out_stream[0].read();
        for (size_t j = 0; j < OUT_CH_PAR; j++) {
            // Each channel should have the average value of 1
            flag &= (output_struct[j] == 1);
        }
    }

    // Check last
    flag &= (out_last.read() == true);
    return flag;
}

int main() {

    bool all_passed = true;

    all_passed &= test_run_simple_square();
    if (!all_passed) {
        std::cout << "Failed." << std::endl;
    } else {
        std::cout << "Passed." << std::endl;
    }

    return all_passed ? 0 : 1;
}
