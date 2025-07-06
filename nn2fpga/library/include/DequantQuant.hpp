#pragma once
#include "ap_int.h"

template <
    int Shift,
    int ACC_WIDTH,
    int OUT_WIDTH>
struct DequantQuantPo2
{
    ap_int<OUT_WIDTH> operator()(ap_int<ACC_WIDTH> acc) const
    {
#pragma HLS inline

        const ap_int<OUT_WIDTH> max_val = (ap_int<OUT_WIDTH>(1) << (OUT_WIDTH - 1)) - 1;
        const ap_int<OUT_WIDTH> min_val = (-ap_int<OUT_WIDTH>(1)) << (OUT_WIDTH - 1);

        ap_int<ACC_WIDTH> shifted = acc >> Shift;

        // Round to nearest even
        if (Shift > 0)
        {
            const ap_int<ACC_WIDTH> half = ap_int<ACC_WIDTH>(1) << (Shift - 1);
            // If the remainder equals half, it's a tie; 
            // break ties by rounding to the nearest even value (LSB of shifted)
            if (acc.range(Shift - 1, 0) == half)
            {
                if (shifted[0] != 0)
                {
                    shifted += (acc >= 0) ? 1 : -1;
                }
            }
            else if (acc.range(Shift - 1, 0) > half)
            {
                shifted += 1;
            }
        }
        
        // Clamp to output range
        if (shifted > max_val)
            shifted = max_val;
        if (shifted < min_val)
            shifted = min_val;

        return ap_int<OUT_WIDTH>(shifted);
    }
};

template <
    int Shift,
    typename TAcc,
    typename TOut>
struct DequantQuantPo2Types
{
    TOut operator()(TAcc acc) const
    {
#pragma HLS inline

        // Compute max and min values based on the output type
        const TOut max_val = TOut((TOut(1) << (TOut::width - 1)) - 1);
        const TOut min_val = TOut(0) - (TOut(1) << (TOut::width - 1));

        TAcc shifted = acc >> Shift;

        // Round to nearest even
        if (Shift > 0)
        {
            const TAcc half = TAcc(1) << (Shift - 1);
            TAcc remainder = acc.range(Shift - 1, 0);

            if (remainder == half)
            {
                if (shifted[0] != 0)
                {
                    shifted += (acc >= 0) ? 1 : -1;
                }
            }
            else if (remainder > half)
            {
                shifted += 1;
            }
        }

        // Clamp to output range
        if (shifted > max_val)
            shifted = max_val;
        if (shifted < min_val)
            shifted = min_val;

        return TOut(shifted);
    }
};


template <typename T>
struct DequantQuantEqual
{
    T operator()(T acc) const
    {
#pragma HLS inline
        return acc;
    }
};
