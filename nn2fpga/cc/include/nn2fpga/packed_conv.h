#ifndef NN2FPGA_PACKED_CONV_H_
#define NN2FPGA_PACKED_CONV_H_

#include "ap_int.h"
#include "hls_stream.h"
#include "nn2fpga/debug.h"
#include "nn2fpga/activations_utils.h"
#include "nn2fpga/stream_utils.h"
#include "nn2fpga/quantisation.h"
#ifdef SIMD_DSP
  #include "nn2fpga/black_box/mac/mac_simd.h"
#endif
#include <cstddef>
#include <type_traits>
#include <math.h>

namespace nn2fpga {

////////////////////////////////////////////////////////////////////////////////

template <class t_output, class t_output_clip, class t_output_mask,
          class t_acc, int c_relu>
t_output quant_stream(t_acc i_acc) {
#pragma HLS inline

  t_acc s_acc = i_acc;

  if (c_relu == 1) {
    s_acc = relu_op<t_acc>(s_acc);
  }
  if constexpr(std::is_same<t_output_clip, std::nullptr_t>::value == false) {
    s_acc = t_output_clip(s_acc);
  }
  if constexpr(std::is_same<t_output_mask, std::nullptr_t>::value == false) {
    s_acc = t_output_mask(s_acc);
  }

  return t_output(s_acc);

}

// Write template for the conv_pipe function.
template <class t_input, class t_weight, class t_weight_st, class t_bias, class t_add_struct,
          class t_input_mod, class t_acc_struct, class t_acc, class t_acc_simd,
          class t_output_struct, class t_output, class t_output_clip, 
          class t_output_mask, int c_reuse, int c_ws, int c_fh, int c_fw, int c_index, int c_str,
          int c_ops, int c_in_ops, int c_relu, int c_ich, int c_och, int c_bits, int c_simd_bits,
          int c_simd, int c_pad_bits, int c_int_pad_bits, int c_mask, int c_w_bits>
void conv_pipe(
    t_input i_input,
    t_weight i_weight[c_index],
    t_bias i_bias,
    uint32_t ops,
    uint32_t och,
    uint32_t ich,
    uint32_t ich_idx,
    uint32_t ich_idx_add,
    uint32_t reuse,
    bool last,
    t_add_struct i_add[c_ws],
    t_acc i_acc_buff[c_reuse][c_och*c_ws],
    t_output_struct s_output_struct[c_ws]) {
#pragma HLS inline

  if constexpr(c_ws > 1) {
    
    t_acc s_acc[c_ws];
    t_acc s_acc_base[c_ws];
    const int FW = (c_fw+(c_ws-1)*c_str);

    ap_uint<48> s_acc_simd[c_simd];

    if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
      if (ich == 0) {
        auto s_bias = i_bias[ops];
        for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
          s_acc[s_ws] = s_bias;
        }
      } else {
        for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
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
        // TODO: Add support for multiple inputs (vector type)
        for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
          s_acc[s_ws] += i_add[s_ws].data[0][ich_idx_add];
        }
      }
    }

    for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
      s_acc_base[s_ws] = i_acc_buff[reuse][och*c_ws+s_ws];
    }

    for (auto s_simd = 0; s_simd < c_simd; s_simd++) {
      s_acc_simd[s_simd] = 0;
    }

    for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
      for (auto s_fw = 0; s_fw < c_fw; s_fw++) {
        ap_int<27> s_data = 0;
        ap_int<18> s_weight = 0;

        ap_int<27> s_input_ext[c_ws];

        for (auto s_ws = 0; s_ws < c_ws; s_ws++) {

          auto s_index = s_fh*FW+s_fw+(c_ws-s_ws-1)*c_str;

          s_input_ext[s_ws] = 0;

          if constexpr(std::is_same<t_input_mod, std::nullptr_t>::value == false)
            s_input_ext[s_ws].range(c_pad_bits*s_ws+c_bits-1, c_pad_bits*s_ws) = t_input_mod(i_input[s_index][ich_idx]).range(c_bits-1, 0);
          else
            s_input_ext[s_ws].range(c_pad_bits*s_ws+c_bits-1, c_pad_bits*s_ws) = i_input[s_index][ich_idx].range(c_bits-1, 0);

          for (auto pos = c_pad_bits*s_ws+c_bits; pos < 27; pos++) {
            s_input_ext[s_ws].range(pos,pos) = s_input_ext[s_ws].range(c_pad_bits*s_ws+c_bits-1, c_pad_bits*s_ws+c_bits-1);
          }

          #ifndef SIMD_DSP
            s_data += s_input_ext[s_ws];
          #endif

        }
        auto s_index = s_fh*c_fw+s_fw;

        s_weight.range(c_w_bits - 1, 0) = i_weight[s_index][ops].range(c_w_bits - 1, 0);
        for (auto pos = c_w_bits; pos < 18; pos++) {
          s_weight.range(pos,pos) = s_weight.range(c_w_bits - 1, c_w_bits - 1);
        }

        #ifdef SIMD_DSP
          auto s_simd_in1 = s_input_ext[0];
          auto s_simd_in2 = s_input_ext[1];

          s_acc_simd[s_index & c_mask] = mac_simd(s_simd_in1, s_weight, s_acc_simd[s_index & c_mask], s_simd_in2);
        #else
          s_acc_simd[s_index & c_mask] += s_data * s_weight;
        #endif

      }
    }

    for (auto s_simd = 0; s_simd < c_simd; s_simd++) {
      for (auto s_ws = 0; s_ws < c_ws; s_ws++) {

        t_acc_simd s_acc_simd_value = 0;
        t_acc_simd s_acc_adj = 0;
        if (s_ws > 0)
          s_acc_adj.range(0,0) = s_acc_simd[s_simd].range(c_pad_bits*(s_ws)-1, c_pad_bits*(s_ws)-1);
        s_acc_simd_value.range(c_pad_bits-1, 0) = s_acc_simd[s_simd].range(c_pad_bits*(s_ws+1)-1, c_pad_bits*s_ws);
        s_acc[s_ws] += s_acc_simd_value + s_acc_adj;
      }
    }

    if (ich != 0) {
      for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
        s_acc[s_ws] += s_acc_base[s_ws];
      }
    }

    for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
      i_acc_buff[reuse][och*c_ws+s_ws] = s_acc[s_ws];
    }


    if (ich == c_ich-1) {
      for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
        s_output_struct[s_ws].data[0][ops] = quant_stream<
          t_output, t_output_clip, t_output_mask, t_acc, c_relu
        >(s_acc[s_ws]);
        s_output_struct[s_ws].last = last;
      }
    }

    // return s_acc_struct;
  }


  if constexpr(c_ws == 1) {

    t_acc s_acc;
    t_acc s_acc_base;

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
        s_acc += i_add[0].data[0][ich_idx_add];
      }
    }

    s_acc_base = i_acc_buff[reuse][och];

    for (auto s_index = 0; s_index < c_index; s_index++) {
      auto s_data = i_input[s_index][ich_idx];
      if constexpr(std::is_same<t_input_mod, std::nullptr_t>::value == false)
        s_data = t_input_mod(s_data);
      s_acc += s_data * i_weight[s_index][ops];
    }

    if (ich != 0) s_acc += s_acc_base;

    i_acc_buff[reuse][och] = s_acc;

    if (ich == c_ich-1) {
      s_output_struct[0].data[0][ops] = quant_stream<
        t_output, t_output_clip, t_output_mask, t_acc, c_relu
      >(s_acc);
      s_output_struct[0].last = last;
    }
    // return s_acc_struct;

  }

}

////////////////////////////////////////////////////////////////////////////////
template <class t_input_struct, class t_input, class t_input_data, class t_weight, class t_weight_st, class t_bias,
          class t_add_struct, class t_add, class t_forward_struct, class t_input_mod, class t_input_1x1,
          class t_weight_1x1, class t_weight_1x1_st, class t_bias_1x1, class t_acc_struct, class t_acc,
          class t_acc_1x1_struct, class t_acc_1x1, 
          class t_output_struct, class t_output, class t_output_clip, class t_output_mask, 
          class t_output_struct_1x1, class t_output_1x1,
          int c_ich, int c_och, int c_oh, int c_ow, int c_fh, int c_fw, int c_index, 
          int c_str, int c_ops, int c_in_ops, int c_add_ops, int c_relu, int c_reuse, int c_ws, int c_ws_out, int c_in_bits, int c_in_ibits, 
          int c_w_bits, int c_w_ibits, int c_simd_bits, int c_simd, int c_mask,
          int c_in_bits_1x1, int c_in_ibits_1x1, int c_w_bits_1x1, int c_w_ibits_1x1,
          int c_simd_bits_1x1, int c_simd_1x1, int c_mask_1x1, int c_depth>
void conv_comp(hls::stream<t_input_struct> i_input[1],
               hls::stream<t_weight> i_weights[c_index],
               hls::stream<t_bias> i_bias[1],
               hls::stream<t_weight_1x1> i_weights_1x1[1],
               hls::stream<t_bias_1x1> i_bias_1x1[1],
               hls::stream<t_add_struct> i_add[c_ws],
               hls::stream<t_forward_struct> o_forward[c_ws],
               hls::stream<t_output_struct> o_output[c_ws],
               hls::stream<t_output_struct_1x1> o_output_1x1[1]) {
  /* #pragma HLS inline */
  // Generic Convolution Computation

  const auto c_och_depth = (c_depth == 1) ? 1 : c_och;
  const auto c_o_index = c_oh * c_ow / c_reuse;
  const auto c_reuse_iter = c_reuse / c_ws;
  const auto c_num_och = c_och_depth / c_ops;
  const auto c_iter = c_reuse_iter * c_num_och;
  // constexpr int FW = (c_fw);
  constexpr int FW = (c_fw+(c_ws-1)*c_str);
  constexpr int MO = (c_fh*FW)/2;

  t_acc s_acc_buff[c_reuse_iter][c_och_depth*c_ws];
#pragma HLS array_partition variable = s_acc_buff type = cyclic factor = c_ops*c_ws dim = 2
  t_acc_1x1 s_acc_1x1_buff[c_reuse_iter][c_och_depth*c_ws];
#pragma HLS array_partition variable = s_acc_1x1_buff type = cyclic factor = c_ops*c_ws dim = 2
  t_input s_input;
// #pragma HLS array_partition variable = s_input type = complete dim = 0
#pragma HLS array_partition variable = s_input type = complete
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
  t_output_struct s_output_struct[c_ws];
#pragma HLS array_partition variable = s_output_struct type = complete
  t_output_struct_1x1 s_output_1x1_struct[c_ws];
#pragma HLS array_partition variable = s_output_1x1_struct type = complete

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the first pipeline
  // round to the higher log2 c_simd
  const int c_pad_bits = c_in_bits+c_w_bits+c_simd_bits;
  const int c_int_pad_bits = c_simd_bits+c_in_ibits+c_w_ibits;
  // mask on c_simd bits
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the second pipeline
  // mask on c_simd bits

  const int c_pad_bits_1x1 = c_in_bits_1x1+c_w_bits_1x1+c_simd_bits_1x1;
  const int c_int_pad_bits_1x1 = c_simd_bits+c_in_ibits_1x1+c_w_ibits_1x1;

  ////////////////////////////////////////////////////////////////////////////

  // TODO: the numbre of integer bits of the weights type must be considered
  typedef ap_fixed<c_pad_bits, c_int_pad_bits, AP_RND, AP_WRAP> t_acc_simd;
  typedef ap_fixed<c_pad_bits_1x1, c_int_pad_bits_1x1, AP_RND, AP_WRAP> t_acc_simd_1x1;

  auto s_ich_idx_add = 0;

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_ich = 0; s_ich < c_ich; s_ich++) {
      for (auto s_iter = 0; s_iter < c_iter; s_iter++) {
#pragma HLS pipeline style = stp II=1
        auto s_reuse = s_iter % c_reuse_iter;
        auto s_num_och = s_iter / c_reuse_iter;
        auto s_ich_idx = s_ich % c_in_ops;
        if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false)
          s_ich_idx_add = s_ich % c_add_ops;

        if ((s_num_och == 0) && ((s_ich_idx) == 0)) {
          t_input_struct s_input_struct = i_input[0].read();
          s_input = s_input_struct.data;
          /* Sending last only at the bottom right data */
          s_last = s_input_struct.last;
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
            if ((s_ich == 0) | (c_depth == 1)) s_bias = i_bias[0].read();
          }

          if constexpr(std::is_same<t_bias_1x1, std::nullptr_t>::value == false) {
            if ((s_ich == 0) | (c_depth == 1)) s_bias_1x1 = i_bias_1x1[0].read();
          }
        }

        // Done to avoid partitioning of the stream and resource wasting
        // Theoretically with the add the input and output dimensions should
        // be equal, so we could perform the addition in different iterations
        if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false){
          for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
            if ((s_iter == 0) && (s_ich_idx_add == 0)){
              s_add[s_ws] = i_add[s_ws].read();   
            }
          }
        }

        COMPUTE:
        for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
          auto s_och = s_num_och * c_ops + s_ops;

          conv_pipe<
            t_input,
            t_weight,
            t_weight_st,
            t_bias,
            t_add_struct,
            t_input_mod,
            t_acc_struct,
            t_acc,
            t_acc_simd,
            t_output_struct,
            t_output,
            t_output_clip,
            t_output_mask,
            c_reuse_iter,
            c_ws,
            c_fh,
            c_fw,
            c_index,
            c_str,
            c_ops,
            c_in_ops,
            c_relu,
            c_ich,
            c_och_depth,
            c_in_bits,
            c_simd_bits,
            c_simd,
            c_pad_bits,
            c_int_pad_bits,
            c_mask,
            c_w_bits
          > (
            s_input,
            s_weight,
            s_bias,
            s_ops,
            s_och,
            s_ich,
            s_ich_idx,
            s_ich_idx_add,
            s_reuse,
            s_last,
            s_add,
            s_acc_buff,
            s_output_struct
          );

          if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
            std::array<t_input_data, c_ws> s_input_1x1;
          #pragma HLS array_partition variable = s_input_1x1 type = complete
            for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
              // s_input_1x1[s_ws] = s_input[MO + MO%c_str - s_ws*c_str];
              s_input_1x1[c_ws - s_ws - 1] = s_input[MO + MO%c_str - s_ws*c_str];
            }
            conv_pipe<
              std::array<t_input_data, c_ws>,
              t_weight_1x1,
              t_weight_1x1_st,
              t_bias_1x1,
              std::nullptr_t,
              t_input_1x1,
              t_acc_1x1_struct,
              t_acc_1x1,
              t_acc_simd_1x1,
              t_output_struct_1x1,
              t_output_1x1,
              std::nullptr_t,
              std::nullptr_t,
              c_reuse_iter,
              c_ws,
              1,
              1,
              1,
              1,
              c_ops,
              c_in_ops,
              0,
              c_ich,
              c_och_depth,
              c_in_bits_1x1,
              c_simd_bits,
              c_simd_1x1,
              c_pad_bits,
              c_int_pad_bits,
              c_mask_1x1,
              c_w_bits_1x1
            > (
              s_input_1x1,
              s_weight_1x1,
              s_bias_1x1,
              s_ops,
              s_och,
              s_ich,
              s_ich_idx,
              s_ich_idx_add,
              s_reuse,
              s_last,
              nullptr,
              s_acc_1x1_buff,
              s_output_1x1_struct
            );
          }
        }
        if ((s_ich == (c_ich-1)) | (c_depth == 1)) {
          for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
            o_output[s_ws % c_ws_out].write(s_output_struct[s_ws]);
            if constexpr(std::is_same<t_output_struct_1x1, std::nullptr_t>::value == false)
              o_output_1x1[s_ws % c_ws_out].write(s_output_1x1_struct[s_ws]);
          }
        }
        if constexpr(std::is_same<t_forward_struct, std::nullptr_t>::value == false) {
          for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
            if ((s_num_och == (c_num_och - 1)) && (s_ich_idx == 0)) {
              t_forward_struct s_forward;
              s_forward.data[0] = s_input[MO + MO%c_str - s_ws*c_str];
              s_forward.last = false;
              o_forward[s_ws % c_ws_out].write(s_forward);
            }
          }
        }
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////

}  // namespace nn2fpga

#endif // NN2FPGA_PACKED_CONV_H_
