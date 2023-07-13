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
          int c_reuse, int c_ws, int c_index,
          int c_ops, int c_ich, int c_och>
void conv_pipe(
    t_input i_input[c_index],
    t_weight i_weight[c_index],
    t_bias i_bias,
    uint32_t ops,
    uint32_t och,
    uint32_t ich,
    uint32_t reuse,
    bool last,
    t_add_struct i_add[c_ws],
    t_acc i_acc_buff[c_reuse][c_och*c_ws],
    t_acc_struct s_acc_struct[c_ws]) {
#pragma HLS inline

  if constexpr(c_ws > 1) {
    
    t_acc s_acc[c_ws];
    t_acc s_acc_base[c_ws];
    for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
      s_acc_struct[s_ws].last = last & (och == (c_och - 1));
    }

    const auto c_bits = t_input::width;
    // TODO: use t_weight which right now is an hls::vector to extract the weight
    // const auto c_pad_bits = t_weight::width*t_input::width+2;
    const auto c_pad_bits = t_input::width*t_input::width+2;
    // round to the higher integer
    const auto c_simd = c_index / 7 + (c_index % 7 != 0);
    // round to the higher log2 c_simd
    const auto c_simd_bits = int(log2(c_simd)) + (c_simd != (1 << int(log2(c_simd))));
    // mask on c_simd bits
    const auto c_mask = (1 << c_simd_bits) - 1;

    typedef ap_fixed<c_simd_bits+t_input::width, c_simd_bits+t_input::iwidth> t_acc_simd;

    if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
      for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
        if (ich == 0) {
          s_acc[s_ws] = i_bias[ops];
        } else {
          s_acc[s_ws] = 0;
        }
      }
    } else {
      for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
        s_acc[s_ws] = 0;
      }
    }

    if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false) {
      if (ich == och) {
        for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
          s_acc[s_ws] += i_add[s_ws].data;
        }
      }
    }
    if (ich == 0) {
      for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
        s_acc_base[s_ws] = i_acc_buff[och*c_ws+s_ws];
      }
    }
    ap_uint<48> s_acc_simd[c_simd] = {0};
    for (auto s_index = 0; s_index < c_index; s_index++) {
      ap_uint<27> s_data = 0;
      for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
        if constexpr(std::is_same<t_input_mod, std::nullptr_t>::value == false)
          s_data.range(c_pad_bits*s_ws+c_bits-1, c_pad_bits*s_ws) = t_input_mod(i_input[s_index]).range(t_input::width-1, 0);
        else
          s_data.range(c_pad_bits*s_ws+c_bits-1, c_pad_bits*s_ws) = i_input[s_index].range(t_input::width-1, 0);
      }
      s_acc_simd[s_index & c_mask] += s_data * i_weight[s_index][ops];
    }

    for (auto s_simd = 0; s_simd < c_simd; s_simd++) {
      for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
        t_acc_simd s_acc_simd_value = s_acc_simd[s_simd];
        s_acc[s_ws] += s_acc_simd_value;
      }
    }

    if (ich != 0) {
      for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
        s_acc[s_ws] += s_acc_base[s_ws];
      }
    }
    for (auto s_ws = 0; s_ws < c_ws - 1; s_ws++) {
      i_acc_buff[reuse][och*c_ws+s_ws] = s_acc[s_ws];
    }

    for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
      s_acc_struct[s_ws].data = s_acc[s_ws];
    }
    // return s_acc_struct;
  }


  if constexpr(c_ws == 1) {

    t_acc s_acc;
    t_acc s_acc_base;
    s_acc_struct[0].last = last & (och == (c_och - 1));

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
        s_acc += i_add[0].data;
      }
    }
    s_acc_base = i_acc_buff[reuse][och];

    for (auto s_index = 0; s_index < c_index; s_index++) {
      t_input s_data = i_input[s_index];
      if constexpr(std::is_same<t_input_mod, std::nullptr_t>::value == false)
        s_data = t_input_mod(s_data);
      s_acc += s_data * i_weight[s_index][ops];
    }
    if (ich != 0) s_acc += s_acc_base;

    i_acc_buff[reuse][och] = s_acc;

    s_acc_struct[0].data = s_acc;

    // return s_acc_struct;

  }

}

////////////////////////////////////////////////////////////////////////////////
template <class t_input_struct, class t_input, class t_weight, class t_bias,
          class t_add_struct, class t_forward_struct, class t_input_mod, class t_input_1x1,
          class t_weight_1x1, class t_bias_1x1, class t_acc_struct, class t_acc,
          class t_acc_1x1_struct, class t_acc_1x1, int c_ich, int c_och,
          int c_oh, int c_ow, int c_index, int c_str, int c_ops, int c_reuse, int c_ws>
void conv_comp(hls::stream<t_input_struct> i_input[c_index+c_reuse-1],
               hls::stream<t_weight> i_weights[c_index],
               hls::stream<t_bias> i_bias[1],
               hls::stream<t_weight_1x1> i_weights_1x1[1],
               hls::stream<t_bias_1x1> i_bias_1x1[1],
               hls::stream<t_add_struct> i_add[1],
               hls::stream<t_forward_struct> o_forward[1],
               hls::stream<t_acc_struct> o_acc_stream[c_ops*c_ws],
               hls::stream<t_acc_1x1_struct> o_acc_1x1_stream[c_ops*c_ws]) {
  /* #pragma HLS inline */
  // Generic Convolution Computation

  const auto c_o_index = c_oh * c_ow / c_reuse;
  const auto c_reuse_iter = c_reuse / c_ws;
  const auto c_num_och = c_och / c_ops;
  const auto c_iter = c_reuse_iter * c_num_och;

  t_acc s_acc_buff[c_reuse_iter][c_och*c_ws];
#pragma HLS array_partition variable = s_acc_buff type = cyclic factor = c_ops*c_ws dim = 2
  t_acc_1x1 s_acc_1x1_buff[c_reuse_iter][c_och*c_ws];
#pragma HLS array_partition variable = s_acc_1x1_buff type = cyclic factor = c_ops*c_ws dim = 2
  t_input s_input[c_index+c_reuse-1];
#pragma HLS array_partition variable = s_input type = complete dim = 2
  t_input s_input_1x1[c_reuse];
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
  t_add_struct s_add[c_ws];
  t_acc_struct s_acc_struct[c_ops*c_ws];
#pragma HLS array_partition variable = s_acc_struct type = complete
  t_acc_1x1_struct s_acc_1x1_struct[c_ops*c_ws];
#pragma HLS array_partition variable = s_acc_1x1_struct type = complete

  std::cout << "conv_comp: " << c_index << " " << c_reuse << " " << c_ops << " " << c_ws << "\n";
  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_ich = 0; s_ich < c_ich; s_ich++) {
      for (auto s_iter = 0; s_iter < c_iter; s_iter++) {
#pragma HLS pipeline style = stp
        auto s_reuse = s_iter % c_reuse_iter;
        auto s_num_och = s_iter / c_reuse_iter;

        if (s_num_och == 0) {
          for (auto s_index = 0; s_index < (c_index+c_reuse-1); s_index++) {
            t_input_struct s_input_struct = i_input[s_index].read();
            s_input[s_index] = s_input_struct.data;
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
          if (s_iter == 0) s_add[0] = i_add[0].read();

        COMPUTE:
        for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
          auto s_och = s_num_och * c_ops + s_ops;
          t_acc_struct s_acc_temp[c_ws];

          conv_pipe<
            t_input,
            t_weight,
            t_bias,
            t_add_struct,
            t_input_mod,
            t_acc_struct,
            t_acc,
            c_reuse_iter,
            c_ws,
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
            s_acc_temp
          );

          for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
            s_acc_struct[s_ops*c_ws+s_ws] = s_acc_temp[s_ws];
          }

          if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
            for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
              s_input_1x1[s_ws] = s_input[c_index / 2 + s_ws*c_str];
            }
            t_acc_1x1_struct s_acc_temp_1x1[c_ws];
            conv_pipe<
              t_input,
              t_weight_1x1,
              t_bias_1x1,
              std::nullptr_t,
              t_input_1x1,
              t_acc_1x1_struct,
              t_acc_1x1,
              c_reuse_iter,
              c_ws,
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
              s_acc_temp_1x1
            );
            for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
              s_acc_1x1_struct[s_ops*c_ws+s_ws] = s_acc_temp_1x1[s_ws];
            }
          }
          if (s_ich == (c_ich-1)) {
            for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
              o_acc_stream[s_ops*c_ws+s_ws].write(s_acc_struct[s_ops*c_ws+s_ws]);
              if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false)
                o_acc_1x1_stream[s_ops*c_ws+s_ws].write(s_acc_1x1_struct[s_ops*c_ws+s_ws]);
            }
          }
        }
        if constexpr(std::is_same<t_forward_struct, std::nullptr_t>::value == false) {
          for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
            if (s_num_och == (c_num_och - 1)) {
              t_forward_struct s_forward;
              s_forward.data = s_input[c_index / 2 + s_ws*c_str];
              s_forward.last = false;
              o_forward[s_ws].write(s_forward);
            }
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
          int c_ops, int c_relu, int c_ws>
void quant_stream(t_acc_struct i_acc, int s_ws, hls::stream<t_output_struct> o_data[c_ws]) {
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

  o_data[s_ws].write(s_output);
}

template <class t_output_struct, class t_output, class t_output_clip, class t_output_mask,
          class t_output_1x1_struct, class t_output_1x1, class t_acc_struct, class t_acc,
          class t_acc_1x1_struct, class t_acc_1x1, int c_ich, int c_och,
          int c_oh, int c_ow, int c_index, int c_ops, int c_relu, int c_stride, int c_ws>
void stream_output(hls::stream<t_acc_struct> i_acc[c_ops*c_ws],
                   hls::stream<t_acc_1x1_struct> i_acc_1x1[c_ops*c_ws],
                   hls::stream<t_output_struct> o_data[c_ws],
                   hls::stream<t_output_1x1_struct> o_data_1x1[c_ws]) {
  /* #pragma HLS inline */

  // std::cout << "stream_output" << std::endl;
  const auto c_num_comp = (c_oh * c_ow * c_och) / c_ws;
  const auto c_pipe_iter = c_num_comp;
  const auto c_num_och = c_och / (c_ops*c_ws);

  t_acc_struct s_acc[c_ws][c_och];
  t_acc_1x1_struct s_acc_1x1[c_ws][c_och];

  std::cout << "stream_output: " << c_num_comp << " " << c_pipe_iter << " " << c_num_och << std::endl;
  for (auto s_pipe_iter = 0; s_pipe_iter < c_pipe_iter; s_pipe_iter++) {
#pragma HLS pipeline style = stp
    for (auto s_ws = 0; s_ws < c_ws; s_ws++) {

      auto s_ops = s_pipe_iter % (c_ops*c_ws);
      auto s_num_och = s_pipe_iter % c_num_och;
      auto s_och = s_pipe_iter % c_och;

      if (s_och < c_num_och) {
        for (auto s_r_ops = 0; s_r_ops < (c_ops*c_ws); s_r_ops++) {
          auto s_r_och = s_num_och * (c_ops*c_ws) + s_r_ops;
          s_acc[s_ws][s_r_och] = i_acc[s_r_ops].read();
          if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
            s_acc_1x1[s_ws][s_r_och] = i_acc_1x1[s_r_ops].read();
          }
        }
      }
      // std::cout << s_acc[s_ws][s_och].data << " ";

      quant_stream<t_output_struct, t_output, t_output_clip, t_output_mask, t_acc_struct, t_acc, c_ich, c_och,
                  c_oh, c_ow, c_index, c_ops, c_relu, c_ws>(s_acc[s_ws][s_och], s_ws, o_data);

      // if (s_pipe_iter%c_och == (c_och-1))
      //   std::cout << std::endl;

      if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
        quant_stream<t_output_1x1_struct, t_output_1x1, std::nullptr_t, std::nullptr_t, t_acc_1x1_struct, t_acc_1x1,
                    c_ich, c_och, c_oh, c_ow, 1, 1, 0, c_ws>(s_acc_1x1[s_ws][s_och], s_ws, o_data_1x1);
      }

    }
  }
  // std::cout << std::endl;
}

}  // namespace nn2fpga

#endif // NN2FPGA_PACKED_CONV_H_
