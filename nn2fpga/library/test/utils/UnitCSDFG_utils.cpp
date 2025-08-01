#include <iostream>
#include <cassert>
#include <vector>
#include "utils/CSDFG_utils.hpp"

int main() {
    bool all_passed = true;

    CSDFGState state1;
    state1.tokens = {5, 3, 2};
    state1.actor_firings = {
        {{2, 0}, {1, 1}},
        {{3, 0}},
        {{4, 0}, {2, 1}}
    };
    state1.actor_positions = {0, 1, 2};

    CSDFGState state2;
    state2.tokens = {5, 3, 2};
    state2.actor_firings = {
        {{2, 0}, {1, 1}},
        {{3, 0}},
        {{4, 0}, {2, 1}}
    };
    state2.actor_positions = {0, 1, 2};

    all_passed &= (state1 == state2);
    
    CSDFGStateHasher hasher;
    std::size_t hash_value = hasher(state1);
    std::size_t expected_hash_value = hasher(state2);
    all_passed &= (hash_value == expected_hash_value);

    // Test changing order in the actor_firings
    state2.actor_firings = {
        {{3, 0}},
        {{2, 0}, {1, 1}},
        {{4, 0}, {2, 1}}
    };

    all_passed &= !(state1 == state2);
    state2 = state1; // Reset to original state

    // Test changing tokens
    state2.tokens[0] = 6;
    all_passed &= !(state1 == state2);
    state2 = state1; // Reset to original state

    // Test changing actor_positions
    state2.actor_positions[0] = 1;
    all_passed &= !(state1 == state2);
    state2 = state1; // Reset to original state

    if (!all_passed) {
        std::cout << "Failed." << std::endl;
    } else {
        std::cout << "Passed." << std::endl;
    }

    return all_passed ? 0 : 1;
}
