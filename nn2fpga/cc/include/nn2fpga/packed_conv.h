#ifndef NN2FPGA_PACKED_CONV_H_
#define NN2FPGA_PACKED_CONV_H_

#include "ap_int.h"
#include "hls_stream.h"
#include "nn2fpga/debug.h"
#include "nn2fpga/activations_utils.h"
#include "nn2fpga/stream_utils.h"
#include "nn2fpga/quantisation.h"
#include <type_traits>

namespace nn2fpga {

////////////////////////////////////////////////////////////////////////////////


// Write template for the conv_pipe function.
template <class t_input, class t_weight, class t_bias, class t_add_struct,
          class t_input_mod, class t_acc_struct, class t_acc,
          int c_reuse, int c_index,
          int c_ops, int c_ich, int c_och>
void conv_pipe(
    t_input i_input[c_reuse][c_index],
    t_weight i_weight[c_index],
    t_bias i_bias,
    uint32_t ops,
    uint32_t och,
    uint32_t ich,
    uint32_t reuse,
    bool last,
    t_add_struct i_add,
    t_acc i_acc_buff[c_reuse][c_och],
    hls::stream<t_acc_struct> o_acc_stream[c_ops]) {
#pragma HLS inline
  t_acc s_acc;
  t_acc s_acc_base;
  t_acc_struct s_acc_struct;
  s_acc_struct.last = last & (och == (c_och - 1));

  if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
    if (ich == 0)
      s_acc = i_bias[ops];
    else
      s_acc = 0;
  } else {
    s_acc = 0;
  }

  if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false) {
    if (ich == och) {
      s_acc += t_acc(i_add.data);
    }
  }

  s_acc_base = i_acc_buff[reuse][och];

  for (auto s_index = 0; s_index < c_index; s_index++) {
    t_input s_data = i_input[reuse][s_index];
    if constexpr(std::is_same<t_input_mod, std::nullptr_t>::value == false)
      s_data = t_input_mod(s_data);
    s_acc += s_data * i_weight[s_index][ops];
  }

  if (ich != 0) s_acc += s_acc_base;

  s_acc_struct.data = s_acc;

  if (ich == (c_ich - 1)) {
    o_acc_stream[ops].write(s_acc_struct);
  } else {
    i_acc_buff[reuse][och] = s_acc;
  }
}

////////////////////////////////////////////////////////////////////////////////
template <class t_input_struct, class t_input, class t_weight, class t_bias,
          class t_add_struct, class t_forward_struct, class t_input_mod, class t_input_1x1,
          class t_weight_1x1, class t_bias_1x1, class t_acc_struct, class t_acc,
          class t_acc_1x1_struct, class t_acc_1x1, int c_ich, int c_och,
          int c_oh, int c_ow, int c_index, int c_str, int c_ops, int c_reuse>
void conv_comp(hls::stream<t_input_struct> i_input[c_index],
               hls::stream<t_weight> i_weights[c_index],
               hls::stream<t_bias> i_bias[1],
               hls::stream<t_weight_1x1> i_weights_1x1[1],
               hls::stream<t_bias_1x1> i_bias_1x1[1],
               hls::stream<t_add_struct> i_add[1],
               hls::stream<t_forward_struct> o_forward[1],
               hls::stream<t_acc_struct> o_acc_stream[c_ops],
               hls::stream<t_acc_1x1_struct> o_acc_1x1_stream[c_ops]) {
  /* #pragma HLS inline */
  // Generic Convolution Computation

  const auto c_o_index = c_oh * c_ow / c_reuse;
  const auto c_num_och = c_och / c_ops;
  const auto c_iter = c_reuse * c_num_och;

  t_acc s_acc_buff[c_reuse][c_och];
#pragma HLS array_partition variable = s_acc_buff type = cyclic factor = c_ops dim = 2
  t_acc_1x1 s_acc_1x1_buff[c_reuse][c_och];
#pragma HLS array_partition variable = s_acc_1x1_buff type = cyclic factor = c_ops dim = 2
  t_input s_input[c_reuse][c_index];
#pragma HLS array_partition variable = s_input type = complete dim = 2
  t_input s_input_1x1[c_reuse][1];
#pragma HLS array_partition variable = s_input_1x1 type = complete
  bool s_last = false;
  t_weight s_weight[c_index];
#pragma HLS array_partition variable = s_weight type = complete
  t_weight_1x1 s_weight_1x1[1];
#pragma HLS array_partition variable = s_weight_1x1 type = complete
  t_bias s_bias;
#pragma HLS array_partition variable = s_bias type = complete
  t_bias_1x1 s_bias_1x1;
#pragma HLS array_partition variable = s_bias_1x1 type = complete
  t_add_struct s_add;

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_ich = 0; s_ich < c_ich; s_ich++) {
      for (auto s_iter = 0; s_iter < c_iter; s_iter++) {
#pragma HLS pipeline style = stp

        auto s_reuse = s_iter % c_reuse;
        auto s_num_och = s_iter / c_reuse;

        if (s_num_och == 0) {
          for (auto s_index = 0; s_index < c_index; s_index++) {
            t_input_struct s_input_struct = i_input[s_index].read();
            s_input[s_reuse][s_index] = s_input_struct.data;
            /* Sending last only at the bottom right data */
            if (s_index == 0) s_last = s_input_struct.last;
          }
        }

        /* Buffering to speed up computations */
        /* TODO: Adjust for generic bit quantizations */
        if (s_reuse == 0) {
          for (auto s_index = 0; s_index < c_index; s_index++) {
            s_weight[s_index] = i_weights[s_index].read();
          }

          // If it is the first reuse iteration, read the 1x1 weights
          if constexpr(std::is_same<t_weight_1x1, std::nullptr_t>::value == false)
            s_weight_1x1[0] = i_weights_1x1[0].read();

          if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
            if (s_ich == 0) s_bias = i_bias[0].read();
          }

          if constexpr(std::is_same<t_bias_1x1, std::nullptr_t>::value == false) {
            if (s_ich == 0) s_bias_1x1 = i_bias_1x1[0].read();
          }
        }

        // Done to avoid partitioning of the stream and resource wasting
        // Theoretically with the add the input and output dimensions should
        // be equal, so we could perform the addition in different iterations
        if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false)
          if (s_iter == 0) s_add = i_add[0].read();

      COMPUTE:
        for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
          auto s_och = s_num_och * c_ops + s_ops;
          conv_pipe<
            t_input,
            t_weight,
            t_bias,
            t_add_struct,
            t_input_mod,
            t_acc_struct,
            t_acc,
            c_reuse,
            c_index,
            c_ops,
            c_ich,
            c_och
          > (
            s_input,
            s_weight,
            s_bias,
            s_ops,
            s_och,
            s_ich,
            s_reuse,
            s_last,
            s_add,
            s_acc_buff,
            o_acc_stream
          );

          if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
            s_input_1x1[s_reuse][0] = s_input[s_reuse][c_index / 2];
            conv_pipe<
              t_input,
              t_weight_1x1,
              t_bias_1x1,
              std::nullptr_t,
              t_input_1x1,
              t_acc_1x1_struct,
              t_acc_1x1,
              c_reuse,
              1,
              c_ops,
              c_ich,
              c_och
            > (
              s_input_1x1,
              s_weight_1x1,
              s_bias_1x1,
              s_ops,
              s_och,
              s_ich,
              s_reuse,
              s_last,
              nullptr,
              s_acc_1x1_buff,
              o_acc_1x1_stream
            );
          }
        }
        if constexpr(std::is_same<t_forward_struct, std::nullptr_t>::value == false) {
          if (s_num_och == (c_num_och - 1)) {
            t_forward_struct s_forward;
            s_forward.data = s_input[s_reuse][c_index / 2];
            s_forward.last = false;
            o_forward[0].write(s_forward);
          }
        }
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////

template <class t_output_struct, class t_output, class t_output_clip,
          class t_output_mask, class t_acc_struct,
          class t_acc, int c_ich, int c_och, int c_oh, int c_ow, int c_index,
          int c_ops, int c_relu>
void quant_stream(t_acc_struct i_acc, hls::stream<t_output_struct> o_data[1]) {
#pragma HLS inline
  t_acc_struct s_acc_struct = i_acc;
  t_acc s_acc = s_acc_struct.data;

  t_output_struct s_output;

  if (c_relu == 1) {
    s_acc = relu_op<t_acc>(s_acc);
  }
  if constexpr(std::is_same<t_output_clip, std::nullptr_t>::value == false) {
    s_acc = t_output_clip(s_acc);
  }
  if constexpr(std::is_same<t_output_mask, std::nullptr_t>::value == false) {
    s_acc = t_output_mask(s_acc);
  }
  s_output.data = t_output(s_acc);
  // std::cout << s_output.data << " ";
  s_output.last = s_acc_struct.last;

  o_data[0].write(s_output);
}

template <class t_output_struct, class t_output, class t_output_clip, class t_output_mask,
          class t_output_1x1_struct, class t_output_1x1, class t_acc_struct, class t_acc,
          class t_acc_1x1_struct, class t_acc_1x1, int c_ich, int c_och,
          int c_oh, int c_ow, int c_index, int c_ops, int c_relu, int c_stride>
void stream_output(hls::stream<t_acc_struct> i_acc[c_ops],
                   hls::stream<t_acc_1x1_struct> i_acc_1x1[c_ops],
                   hls::stream<t_output_struct> o_data[1],
                   hls::stream<t_output_1x1_struct> o_data_1x1[1]) {
  /* #pragma HLS inline */

  const auto c_num_comp = c_oh * c_ow * c_och;
  const auto c_pipe_iter = c_num_comp;
  const auto c_num_och = c_och / c_ops;

  t_acc_struct s_acc[c_och];
  t_acc_1x1_struct s_acc_1x1[c_och];

  for (auto s_pipe_iter = 0; s_pipe_iter < c_pipe_iter; s_pipe_iter++) {
#pragma HLS pipeline style = stp
    auto s_ops = s_pipe_iter % c_ops;
    auto s_num_och = s_pipe_iter % c_num_och;
    auto s_och = s_pipe_iter % c_och;

    if (s_och < c_num_och) {
      for (auto s_r_ops = 0; s_r_ops < c_ops; s_r_ops++) {
        auto s_r_och = s_num_och * c_ops + s_r_ops;
        s_acc[s_r_och] = i_acc[s_r_ops].read();
        if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
          s_acc_1x1[s_r_och] = i_acc_1x1[s_r_ops].read();
        }
      }
    }

    quant_stream<t_output_struct, t_output, t_output_clip, t_output_mask, t_acc_struct, t_acc, c_ich, c_och,
                 c_oh, c_ow, c_index, c_ops, c_relu>(s_acc[s_och], o_data);

    if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
      quant_stream<t_output_1x1_struct, t_output_1x1, std::nullptr_t, std::nullptr_t, t_acc_1x1_struct, t_acc_1x1,
                  c_ich, c_och, c_oh, c_ow, 1, 1, 0>(s_acc_1x1[s_och], o_data_1x1);
    }
  }
}

}  // namespace nn2fpga

#endif // NN2FPGA_PACKED_CONV_H_
