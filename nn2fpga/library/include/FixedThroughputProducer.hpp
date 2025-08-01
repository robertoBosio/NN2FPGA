#pragma once
#include "hls_stream.h"
#include "utils/CSDFG_utils.hpp"
#include <cstddef>

// Template class for producing data streams with fixed throughput.
template <typename TOutputStruct, typename TOutput, size_t BOTTLENECK_CYCLES,
          size_t DATA_TO_PRODUCE>
class FixedThroughputProducer {
public:
  ActorStatus step(hls::stream<TOutputStruct> &output_stream) {

    STEP_counter += DATA_TO_PRODUCE; // Increment the step counter by the number
                                     // of data to produce.
    if (STEP_counter >= BOTTLENECK_CYCLES) {
      // If we have reached the bottleneck cycles, reset the counter.
      STEP_counter -= BOTTLENECK_CYCLES;

      // Write a token.
      TOutputStruct output_data;
      output_data.data = 0;
      output_stream.write(output_data);
    }

    // Fire the actor.
    STEP_actor_status.fire();

    // Advance the actor status.
    STEP_actor_status.advance();

    // Return the current actor status.
    return STEP_actor_status;
  }

private:
  size_t STEP_counter = 0; // Counter to track the number of steps taken.

  ActorStatus STEP_actor_status{1, BOTTLENECK_CYCLES};
};
