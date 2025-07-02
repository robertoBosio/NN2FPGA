#include <array>
#include <iostream>
#include <cassert>
#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "BandwidthAdjust.hpp"

template <typename TInputStruct,
          typename TInput,
          typename TOutputStruct,
          typename TOutput,
          size_t IN_HEIGHT,
          size_t IN_WIDTH,
          size_t IN_CH,
          size_t IN_W_PAR,
          size_t OUT_W_PAR,
          size_t CH_PAR>
bool test_run_increaseWPAR() {
    
    // Simple quantizer: truncates accumulator
    struct TruncQuantizer
    {
        TOutput operator()(ap_uint<8> data) const
        {
            return static_cast<TOutput>(data);
        }
    };

    BandwidthAdjustIncreaseStreams<
        TInputStruct, TInput,
        TOutputStruct, TOutput,
        TruncQuantizer,
        IN_HEIGHT, IN_WIDTH,
        IN_CH, IN_W_PAR, OUT_W_PAR, CH_PAR>
        bandwidth_adjust;

    // Prepare input and output streams
    hls::stream<TInputStruct> in_stream[IN_W_PAR];
    hls::stream<TOutputStruct> out_stream[OUT_W_PAR];
    hls::stream<bool> in_last;
    hls::stream<bool> out_last;

    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += IN_W_PAR)
    {
        for (size_t ch = 0; ch < IN_CH; ch += CH_PAR)
        {
            for (size_t w_par = 0; w_par < IN_W_PAR; w_par++)
            {
                TInputStruct input_struct;
                for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++)
                {
                    // Each channel should have the average value of 1
                    input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
                }
                in_stream[w_par].write(input_struct);
            }
        }
    }

    in_last.write(true); // Set last signal for the input stream

    // Run the operator
    bandwidth_adjust.run(in_stream, in_last, out_stream, out_last);

    // Check output
    bool flag = true;
    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += OUT_W_PAR)
    {
        for (size_t ch = 0; ch < IN_CH; ch += CH_PAR)
        {
            for (size_t w_par = 0; w_par < OUT_W_PAR; w_par++)
            {
                TOutputStruct output_struct = out_stream[w_par].read();
                for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++)
                {
                    flag &= (output_struct[ch_par] == (i + w_par) * IN_CH + ch + ch_par);
                }
            }
        }
    }

    // Check last signal
    flag &= (out_last.read() == true);

    return flag;
}

template <typename TInputStruct,
          typename TInput,
          typename TOutputStruct,
          typename TOutput,
          size_t IN_HEIGHT,
          size_t IN_WIDTH,
          size_t IN_CH,
          size_t IN_W_PAR,
          size_t OUT_W_PAR,
          size_t CH_PAR>
bool test_step_increaseWPAR() {
    
    // Simple quantizer: truncates accumulator
    struct TruncQuantizer
    {
        TOutput operator()(ap_uint<8> data) const
        {
            return static_cast<TOutput>(data);
        }
    };

    BandwidthAdjustIncreaseStreams<
        TInputStruct, TInput,
        TOutputStruct, TOutput,
        TruncQuantizer,
        IN_HEIGHT, IN_WIDTH,
        IN_CH, IN_W_PAR, OUT_W_PAR, CH_PAR>
        bandwidth_adjust;

    // Prepare input and output streams
    hls::stream<TInputStruct> in_stream[IN_W_PAR];
    hls::stream<TOutputStruct> out_stream[OUT_W_PAR];
    hls::stream<bool> in_last;
    hls::stream<bool> out_last;

    // Check step function not progressing in case of no data
    bool flag = (bandwidth_adjust.step(in_stream, in_last, out_stream, out_last) == false);

    TInputStruct input_struct;
    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += IN_W_PAR)
    {
        for (size_t ch = 0; ch < IN_CH; ch += CH_PAR)
        {
            for (size_t w_par = 0; w_par < IN_W_PAR; w_par++)
            {
                for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++)
                {
                    // Each channel should have the average value of 1
                    input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
                }
                in_stream[w_par].write(input_struct);
            }
        }
    }

    in_last.write(true); // Set last signal for the input stream

    // Run the operator
    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH * IN_CH / (CH_PAR * IN_W_PAR); i++)
    {
        flag &= bandwidth_adjust.step(in_stream, in_last, out_stream, out_last);
    } 

    // Check last signal
    flag &= (out_last.read() == true);

    // Check step function not progresssing in case of completed iterations.
    for (size_t i = 0; i < IN_W_PAR; i++)
    {
        in_stream[i].write(input_struct);
    }
    flag &= (bandwidth_adjust.step(in_stream, in_last, out_stream, out_last) == false);

    return flag;
}

template <typename TInputStruct,
          typename TInput,
          typename TOutputStruct,
          typename TOutput,
          size_t IN_HEIGHT,
          size_t IN_WIDTH,
          size_t IN_CH,
          size_t IN_W_PAR,
          size_t OUT_W_PAR,
          size_t CH_PAR>
bool test_run_decreaseWPAR() {
    
    // Simple quantizer: truncates accumulator
    struct TruncQuantizer
    {
        TOutput operator()(ap_uint<8> data) const
        {
            return static_cast<TOutput>(data);
        }
    };

    BandwidthAdjustDecreaseStreams<
        TInputStruct, TInput,
        TOutputStruct, TOutput,
        TruncQuantizer,
        IN_HEIGHT, IN_WIDTH,
        IN_CH, IN_W_PAR, OUT_W_PAR, CH_PAR>
        bandwidth_adjust;

    // Prepare input and output streams
    hls::stream<TInputStruct> in_stream[IN_W_PAR];
    hls::stream<TOutputStruct> out_stream[OUT_W_PAR];
    hls::stream<bool> in_last;
    hls::stream<bool> out_last;

    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += IN_W_PAR)
    {
        for (size_t ch = 0; ch < IN_CH; ch += CH_PAR)
        {
            for (size_t w_par = 0; w_par < IN_W_PAR; w_par++)
            {
                TInputStruct input_struct;
                for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++)
                {
                    // Each channel should have the average value of 1
                    input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
                }
                in_stream[w_par].write(input_struct);
            }
        }
    }

    in_last.write(true); // Set last signal for the input stream

    // Run the operator
    bandwidth_adjust.run(in_stream, in_last, out_stream, out_last);

    // Check output
    bool flag = true;
    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += OUT_W_PAR)
    {
        for (size_t ch = 0; ch < IN_CH; ch += CH_PAR)
        {
            for (size_t w_par = 0; w_par < OUT_W_PAR; w_par++)
            {
                TOutputStruct output_struct = out_stream[w_par].read();
                for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++)
                {
                    flag &= (output_struct[ch_par] == (i + w_par) * IN_CH + ch + ch_par);
                }
            }
        }
    }

    // Check last signal
    flag &= (out_last.read() == true);

    return flag;
}

template <typename TInputStruct,
          typename TInput,
          typename TOutputStruct,
          typename TOutput,
          size_t IN_HEIGHT,
          size_t IN_WIDTH,
          size_t IN_CH,
          size_t IN_W_PAR,
          size_t OUT_W_PAR,
          size_t CH_PAR>
bool test_step_decreaseWPAR() {
    
    // Simple quantizer: truncates accumulator
    struct TruncQuantizer
    {
        TOutput operator()(ap_uint<8> data) const
        {
            return static_cast<TOutput>(data);
        }
    };

    BandwidthAdjustDecreaseStreams<
        TInputStruct, TInput,
        TOutputStruct, TOutput,
        TruncQuantizer,
        IN_HEIGHT, IN_WIDTH,
        IN_CH, IN_W_PAR, OUT_W_PAR, CH_PAR>
        bandwidth_adjust;

    // Prepare input and output streams
    hls::stream<TInputStruct> in_stream[IN_W_PAR];
    hls::stream<TOutputStruct> out_stream[OUT_W_PAR];
    hls::stream<bool> in_last;
    hls::stream<bool> out_last;

    // Check step function not progressing in case of no data
    bool flag = (bandwidth_adjust.step(in_stream, in_last, out_stream, out_last) == false);

    TInputStruct input_struct;
    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH; i += IN_W_PAR)
    {
        for (size_t ch = 0; ch < IN_CH; ch += CH_PAR)
        {
            for (size_t w_par = 0; w_par < IN_W_PAR; w_par++)
            {
                for (size_t ch_par = 0; ch_par < CH_PAR; ch_par++)
                {
                    // Each channel should have the average value of 1
                    input_struct[ch_par] = (i + w_par) * IN_CH + ch + ch_par;
                }
                in_stream[w_par].write(input_struct);
            }
        }
    }

    in_last.write(true); // Set last signal for the input stream

    // Run the operator
    for (size_t i = 0; i < IN_HEIGHT * IN_WIDTH * IN_CH / (CH_PAR * OUT_W_PAR); i++)
    {
        flag &= bandwidth_adjust.step(in_stream, in_last, out_stream, out_last);
    } 

    // Check last signal
    flag &= (out_last.read() == true);

    // Check step function not progresssing in case of completed iterations.
    for (size_t i = 0; i < IN_W_PAR; i++)
    {
        in_stream[i].write(input_struct);
    }
    flag &= (bandwidth_adjust.step(in_stream, in_last, out_stream, out_last) == false);

    return flag;
}

int main() {

    bool all_passed = true;

    // Test bandwidth adjustment from 2 to 4 streams
    all_passed &= test_run_increaseWPAR<std::array<ap_uint<8>, 2>,
                           ap_uint<8>,
                           std::array<ap_uint<8>, 2>,
                           ap_uint<8>,
                           4, 4, 4, 2, 4, 2>();
    
    // Test step when passing from 2 to 4 streams
    all_passed &= test_step_increaseWPAR<std::array<ap_uint<8>, 2>,
                           ap_uint<8>,
                           std::array<ap_uint<8>, 2>,
                           ap_uint<8>,
                           4, 4, 4, 2, 4, 2>();
    
    // Test bandwidth adjustment from 4 to 2 streams
    all_passed &= test_run_decreaseWPAR<std::array<ap_uint<8>, 2>,
                           ap_uint<8>,
                           std::array<ap_uint<8>, 2>,
                           ap_uint<8>,
                           4, 4, 4, 4, 2, 2>();

    // Test step when passing from 4 to 2 streams
    all_passed &= test_step_decreaseWPAR<std::array<ap_uint<8>, 2>,
                           ap_uint<8>,
                           std::array<ap_uint<8>, 2>,
                           ap_uint<8>,
                           4, 4, 4, 4, 2, 2>();

    if (!all_passed) {
        std::cout << "Failed." << std::endl;
    } else {
        std::cout << "Passed." << std::endl;
    }

    return all_passed ? 0 : 1;
}