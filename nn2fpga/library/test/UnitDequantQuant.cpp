#include "DequantQuant.hpp"
#include "ap_int.h"

constexpr int ACC_WIDTH = 16;
constexpr int OUT_WIDTH = 8;

bool test_basic_shift() {
    constexpr int Shift = 2;
    DequantQuantPo2<Shift, ACC_WIDTH, OUT_WIDTH> dq;
    return dq(64) == 16;
}

bool test_negative_shift() {
    constexpr int Shift = -2;
    DequantQuantPo2<Shift, ACC_WIDTH, OUT_WIDTH> dq;
    return dq(16) == 64;
}

bool test_zero_shift() {
    constexpr int Shift = 0;
    DequantQuantPo2<Shift, ACC_WIDTH, OUT_WIDTH> dq;
    return dq(64) == 64 && dq(-64) == -64;
}

bool test_round_to_even() {
    constexpr int Shift = 3;
    DequantQuantPo2<Shift, ACC_WIDTH, OUT_WIDTH> dq;
    return dq(27) == 3 && dq(28) == 4 && dq(-12) == -2 && dq(-11) == -1;
}

bool test_clamp_positive() {
    constexpr int Shift = 0;
    DequantQuantPo2<Shift, ACC_WIDTH, OUT_WIDTH> dq;
    return dq(200) == 127;
}

bool test_clamp_negative() {
    constexpr int Shift = 0;
    DequantQuantPo2<Shift, ACC_WIDTH, OUT_WIDTH> dq;
    return dq(-200) == -128;
}

bool test_negative_rounding() {
    constexpr int Shift = 2;
    DequantQuantPo2<Shift, ACC_WIDTH, OUT_WIDTH> dq;
    return dq(-19) == -5;
}

int main() {
    bool all_passed = true;

    all_passed &= test_basic_shift();
    all_passed &= test_zero_shift();
    all_passed &= test_negative_shift();
    all_passed &= test_round_to_even();
    all_passed &= test_clamp_positive();
    all_passed &= test_clamp_negative();
    all_passed &= test_negative_rounding();
    if (!all_passed) {
        std::cout << "Failed." << std::endl;
    } else {
        std::cout << "Passed." << std::endl;
    }

    return all_passed ? 0 : 1;
}
