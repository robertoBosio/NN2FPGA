#include "StreamToNHWC.hpp"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include <array>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include "utils/CSDFG_utils.hpp"
#include "test_config.hpp"

bool test_run() {
  // This function tests the run() method of StreamToNHWC.
  using TInputWord = std::array<test_config::TInput, test_config::IN_CH_PAR>;

  // Instantiate the operator
  StreamToNHWC<TInputWord, test_config::TInput, test_config::TOutputWord,
               test_config::TOutput, test_config::Quantizer,
               test_config::DATA_PER_WORD, test_config::HEIGHT,
               test_config::WIDTH, test_config::CH, test_config::IN_W_PAR,
               test_config::IN_CH_PAR>
      consumer;

  // Prepare input and output streams
  hls::stream<TInputWord> in_stream[test_config::IN_W_PAR];
  hls::stream<test_config::TOutputWord> out_stream;

  for (size_t i = 0; i < test_config::HEIGHT * test_config::WIDTH;
       i += test_config::IN_W_PAR) {
    for (size_t c = 0; c < test_config::CH; c += test_config::IN_CH_PAR) {
      for (size_t w_par = 0; w_par < test_config::IN_W_PAR; w_par++) {
        TInputWord input_struct;
        for (size_t c_par = 0; c_par < test_config::IN_CH_PAR; c_par++) {
          input_struct[c_par] = (i + w_par) * test_config::CH + c + c_par;
        }
        in_stream[w_par].write(input_struct);
      }
    }
  }

  // Run consumer
  consumer.run(in_stream, out_stream);

  // Read and check output
  bool flag = true;
  size_t data_in_word = 0;
  test_config::TOutputWord output_struct;
  for (size_t i = 0; i < test_config::HEIGHT * test_config::WIDTH; i++) {
    for (size_t c = 0; c < test_config::CH; c++) {
      if (data_in_word == 0) {
        // Read the output structure from the stream
        output_struct = out_stream.read();
      }

      // Fill the input structure with the value 1 for each channel
      flag &= (output_struct.data.range(
                   test_config::TInput::width * (data_in_word + 1) - 1,
                   test_config::TInput::width * data_in_word) ==
               i * test_config::CH + c);
      data_in_word++;

      if (data_in_word >= test_config::DATA_PER_WORD) {
        data_in_word = 0;
      }
    }
  }

  return flag;
}

bool test_step() {
  // This function tests the step() method of StreamToNHWC

  using TInputWord = std::array<test_config::TInput, test_config::IN_CH_PAR>;
  static constexpr size_t expectedII =
      test_config::HEIGHT * test_config::WIDTH * test_config::CH /
      (test_config::IN_CH_PAR * test_config::IN_W_PAR);

  // Instantiate the operator
  StreamToNHWC<TInputWord, test_config::TInput, test_config::TOutputWord,
               test_config::TOutput, test_config::Quantizer,
               test_config::DATA_PER_WORD, test_config::HEIGHT,
               test_config::WIDTH, test_config::CH, test_config::IN_W_PAR,
               test_config::IN_CH_PAR>
      consumer(test_config::PIPELINE_DEPTH);

  // Prepare input and output streams
  hls::stream<TInputWord> in_stream[test_config::IN_W_PAR];
  hls::stream<test_config::TOutputWord> out_stream;

  std::unordered_map<CSDFGState, size_t, CSDFGStateHasher> visited_states;
  CSDFGState current_state;
  size_t clock_cycles = 0;
  size_t II = 0;
  while (true) {
    TInputWord input_data;
    for (size_t i_w_par = 0; i_w_par < test_config::IN_W_PAR; i_w_par++) {
      in_stream[i_w_par].write(input_data);
    }
    ActorStatus actor_status = consumer.step(in_stream, out_stream);
    std::vector<ActorStatus> actor_statuses;
    std::vector<size_t> channel_quantities;
    actor_statuses.push_back(actor_status);
    channel_quantities.push_back(0);
    current_state = CSDFGState(actor_statuses, channel_quantities);
    if (visited_states.find(current_state) != visited_states.end()) {
      II = clock_cycles - visited_states[current_state];
      break;
    }
    visited_states.emplace(current_state, clock_cycles);

    // Prevent infinite loops in case of errors
    clock_cycles++;
    assert(clock_cycles < 10 * expectedII);
  }

  // Flush the output stream.
  test_config::TOutputWord output_word;
  while (out_stream.read_nb(output_word))
    ;

  bool flag = (II == expectedII);
  std::cout << "Expected II: " << expectedII << ", Measured II: " << II
            << std::endl;
  return flag;
}

int main() {

  bool all_passed = true;
  all_passed &= test_run();
  all_passed &= test_step();

  if (!all_passed) {
    std::cout << "Failed." << std::endl;
  } else {
    std::cout << "Passed." << std::endl;
  }

  return all_passed ? 0 : 1;
}
