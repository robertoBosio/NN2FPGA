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

#pragma GCC diagnostic push
// #pragma GCC diagnostic error "-Wpedantic"
// #pragma GCC diagnostic error "-Wall"
// #pragma GCC diagnostic error "-Wextra"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wc++11-compat"

namespace nn2fpga {

////////////////////////////////////////////////////////////////////////////////

template<class t_output,
         class t_output_clip,
         class t_output_mask,
         class t_acc,
         int c_relu,
         int c_leakyrelu>
t_output
quant_stream(t_acc i_acc)
{
#pragma HLS inline

  t_acc s_acc = i_acc;
  t_output s_leakyrelu_in, s_leakyrelu_out, s_output;

  if constexpr (std::is_same<t_output_clip, std::nullptr_t>::value == false) {
    s_acc = t_output_clip(s_acc);
  }

  if constexpr (std::is_same<t_output_mask, std::nullptr_t>::value == false) {
    s_acc = t_output_mask(s_acc);
  }
  
  if constexpr (c_relu == 1) {
    s_acc = relu_op<t_acc>(s_acc);
    s_output = t_output(s_acc);
  }
  else if constexpr (c_leakyrelu == 1){
    s_leakyrelu_in = t_output(s_acc);
    s_leakyrelu_out = s_leakyrelu_in;
    s_leakyrelu_out = leakyrelu <t_output, t_output, t_output> (s_leakyrelu_in);
    s_output = t_output(s_leakyrelu_out);
  }
  else 
    s_output = t_output(s_acc);

  return s_output;
}

template<class t_output,
         class t_output_clip,
         class t_output_mask,
         class t_add,
         class t_acc,
         int c_relu,
         int c_leakyrelu>
t_output
quant_and_add_stream(t_acc i_acc, t_add i_add)
{
#pragma HLS inline
  
  t_acc s_acc = i_acc;
  t_output s_leakyrelu_in, s_leakyrelu_out, s_output;

  // Post convolution quantization
  if constexpr (std::is_same<t_output_clip, std::nullptr_t>::value == false) {
    s_acc = t_output_clip(s_acc);
  }
  
  s_acc += i_add;

  // Post addition quantization
  if constexpr (std::is_same<t_output_mask, std::nullptr_t>::value == false) {
    s_acc = t_output_mask(s_acc);
  }
  
  if constexpr (c_relu == 1) {
    s_acc = relu_op<t_acc>(s_acc);
    s_output = t_output(s_acc);
  }
  else if constexpr (c_leakyrelu == 1){
    s_leakyrelu_in = t_output(s_acc);
    s_leakyrelu_out = leakyrelu <t_output, t_output, t_output> (s_leakyrelu_in);
    s_output = t_output(s_leakyrelu_out);
  }
  else 
    s_output = t_output(s_acc); 
  // Post activation quantization
  return s_output;
}

template<int pad_bits>
ap_uint<48>
DSP48E2_ADDMULADD_PREADD(ap_int<27> A, ap_int<18> B, ap_int<27> D, ap_uint<48> C)
{
#pragma HLS inline 

  return ((A + D) * B) + C;
}

template<typename t_act,
         typename t_act_ext,
         typename t_act_pad_ext,
         typename t_weight,
         typename t_weight_ext,
         typename t_res,
         int pad_bits>
ap_uint<48>
double_packing(t_act input0, t_act input1, t_weight weight, t_res res)
{
#pragma HLS inline
  static_assert(t_act::width <= 8, "width of the inputs <= 8");
  static_assert(t_weight::width <= 8, "width of the weights <= 8");

  /* The activations must be extened with the proper sign */
  t_act_ext input0_ext = 0;
  t_act_pad_ext input1_ext = 0;

  /* The weight must be extened with the proper sign */
  t_weight_ext weight_ext = 0;

  // std::cout << "input0 " << input0 << " input1 " << input1 << " weight " <<
  // weight << std::endl;
  input0_ext.range(t_act::width - 1, 0) = input0.range(t_act::width - 1, 0);

  input1_ext.range(t_act::width + pad_bits - 1, pad_bits) =
    input1.range(t_act::width - 1, 0);

  weight_ext.range(t_weight::width - 1, 0) =
    weight.range(t_weight::width - 1, 0);

  // std::cout << "A " << input0_ext << " B " << input1_ext << " C " <<
  // weight_ext << " D " << res << std::endl;
  return DSP48E2_ADDMULADD_PREADD<pad_bits>(
    input0_ext, weight_ext, input1_ext, res);
}

template<typename t_act,
         typename t_weight,
         typename t_res,
         int pad_bits>
ap_uint<48>
double_packing_wrap(t_act input0, t_act input1, t_weight weight, t_res res)
{
  #pragma HLS inline
  if constexpr (std::is_same<typename t_act::Base::Base,
                             _AP_ROOT_TYPE<t_act::Base::width, true>>::value) {
    if constexpr (std::is_same<
                    typename t_weight::Base::Base,
                    _AP_ROOT_TYPE<t_weight::Base::width, true>>::value) {
      /* Activations and weight are signed*/
      return double_packing<t_act,
                            ap_int<t_act::width>,
                            ap_int<t_act::width + pad_bits>,
                            t_weight,
                            ap_int<t_weight::width>,
                            t_res,
                            pad_bits>(input0, input1, weight, res);
    } else {
      /* Activations are signed, the weight is unsigned*/
      return double_packing<t_act,
                            ap_int<t_act::width>,
                            ap_int<t_act::width + pad_bits>,
                            t_weight,
                            ap_uint<t_weight::width>,
                            t_res,
                            pad_bits>(input0, input1, weight, res);
    }
  } else {
    if constexpr (std::is_same<
                    typename t_weight::Base::Base,
                    _AP_ROOT_TYPE<t_weight::Base::width, true>>::value) {
      /* Activations are unsigned, the weight is signed*/
      return double_packing<t_act,
                            ap_uint<t_act::width>,
                            ap_uint<t_act::width + pad_bits>,
                            t_weight,
                            ap_int<t_weight::width>,
                            t_res,
                            pad_bits>(input0, input1, weight, res);
    } else {
      /* Activations and weight are unsigned*/
      return double_packing<t_act,
                            ap_uint<t_act::width>,
                            ap_uint<t_act::width + pad_bits>,
                            t_weight,
                            ap_uint<t_weight::width>,
                            t_res,
                            pad_bits>(input0, input1, weight, res);
    }
  }
}

template<typename t_act, typename t_weight, typename t_res, int pad_bits>
void
double_packing_debug(t_act input0, t_act input1, t_weight weight, t_res res)
{
#pragma HLS inline
  static_assert(t_act::width <= 8, "width of the inputs <= 8");
  static_assert(t_weight::width <= 8, "width of the weights <= 8");

  std::cout << "input0 " << input0 << " input1 " << input1 << " weight " << weight << std::endl;
  ap_int<t_act::width> input0_ext = 0;
  input0_ext.range(t_act::width - 1, 0) = input0.range(t_act::width - 1, 0);

  ap_int<t_act::width + pad_bits> input1_ext = 0;
  input1_ext.range(t_act::width + pad_bits - 1, pad_bits) =
    input1.range(t_act::width - 1, 0);

  ap_int<t_weight::width> weight_ext = 0;
  weight_ext.range(t_weight::width - 1, 0) = weight.range(t_weight::width - 1, 0);

  std::cout << "A " << input0_ext << " B " << input1_ext << " C " << weight_ext << " D " << res << std::endl;
}

// Template for the conv_pipe function.
template<class t_input,
         class t_weight,
         class t_weight_st,
         class t_bias,
         class t_add_vector,
         class t_add,
         class t_input_mod,
         class t_acc_struct,
         class t_acc,
         class t_acc_simd,
         class t_output,
         class t_output_clip,
         class t_output_mask,
         int c_reuse,
         int c_fh,
         int c_fw,
         int c_index,
         int c_str,
         int c_ops,
         int c_iter_ops_out,
         int c_in_ops,
         int c_add_ops,
         int c_ow_ops,
         int c_ow_pack,
         int c_och_pack,
         int c_relu,
         int c_leakyrelu,
         int c_ich,
         int c_och,
         int c_bits,
         int c_simd_bits,
         int c_simd,
         int c_pad_bits,
         int c_int_pad_bits,
         int c_pad_acc_bits,
         int c_mask,
         int c_w_bits,
         int c_depth>
void
conv_pipe(const t_input i_input,
          const t_weight i_weight[c_index],
          const t_bias i_bias,
          const uint32_t ops,
          const uint32_t num_ich,
          const uint32_t s_ow_ops,
          const uint32_t ich_idx_add,
          const t_add_vector i_add[c_ow_ops],
          t_acc i_acc_buff[c_ops * c_ow_ops],
          t_output s_output_struct[c_ow_ops][c_iter_ops_out])
{
#pragma HLS inline

  const int FW = (c_fw + (c_ow_ops - 1) * c_str);

  if constexpr (c_och_pack > 1) {
    // Depthwise convolutions cannot be packed over the output channels.

    t_acc s_acc[c_ow_pack * c_och_pack];
    t_acc s_acc_base[c_ow_pack * c_och_pack];

    ap_uint<48> s_acc_simd[c_simd];

    for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
      for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
        auto s_w_index = s_och_pack * c_ow_pack + s_ow_pack;
        s_acc[s_w_index] = 0;
        if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
          if (num_ich == 0) {
            #ifndef __SYNTHESIS__
              #ifdef DEBUG_CONV
                std::cout << "B" << " " << i_bias[0][ops+s_och_pack] << std::endl;
              #endif
            #endif
            s_acc[s_w_index] = i_bias[0][ops+s_och_pack];
          }
        }
      }
    }

    for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
      for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
        auto s_w_index = s_och_pack * c_ow_pack + s_ow_pack;
        auto s_r_index = (ops + s_och_pack) * c_ow_ops + s_ow_pack + s_ow_ops;
        s_acc_base[s_w_index] = i_acc_buff[s_r_index];
      }
    }

    for (auto s_simd = 0; s_simd < c_simd; s_simd++) {
      s_acc_simd[s_simd] = 0;
    }

    ICH_OPS_LOOP:
    for (auto ich_idx = 0; ich_idx < c_in_ops; ich_idx++) {
      FH_LOOP:
      for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
        FW_LOOP:
        for (auto s_fw = 0; s_fw < c_fw; s_fw++) {
          ap_int<27> s_data = 0;
          ap_int<18> s_b_ext = 0;

          ap_int<27> s_a_d_ext[c_och_pack];

          for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {

            auto s_index = s_fh*c_fw+s_fw;

            s_a_d_ext[s_och_pack] = 0;

            s_a_d_ext[s_och_pack].range(c_pad_bits*s_och_pack+c_w_bits-1, c_pad_bits*s_och_pack) = i_weight[s_index][ich_idx][ops+s_och_pack].range(c_w_bits - 1, 0);

            #ifndef __SYNTHESIS__
              #ifdef DEBUG_CONV
                std::cout << "W" << s_index << " " << i_weight[s_index][ich_idx][ops+s_och_pack] << std::endl;
              #endif
            #endif

            if constexpr(std::is_same<typename t_weight_st::Base::Base, _AP_ROOT_TYPE<t_weight_st::Base::width, true>>::value) {
              for (auto pos = c_pad_bits*s_och_pack+c_w_bits; pos < 27; pos++) {
                s_a_d_ext[s_och_pack].range(pos,pos) = s_a_d_ext[s_och_pack].range(c_pad_bits*s_och_pack+c_w_bits-1, c_pad_bits*s_och_pack+c_w_bits-1);
              }
            }

            s_data += s_a_d_ext[s_och_pack];

          }

          for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {

            auto s_index = s_fh * FW + s_fw +
                           (c_ow_ops - (s_ow_pack + s_ow_ops) - 1) * c_str;
            auto s_w_index = s_ow_pack * (c_pad_acc_bits);

            s_b_ext.range(s_w_index + c_bits - 1, s_w_index) = t_input_mod(i_input[s_index][ich_idx]).range(c_bits-1, 0);
            if constexpr(std::is_same<typename t_input_mod::Base::Base, _AP_ROOT_TYPE<t_input_mod::Base::width, true>>::value) {
              for (auto pos = s_w_index+c_bits; pos < 18; pos++) {
                s_b_ext.range(pos,pos) = s_b_ext.range(s_w_index + c_bits - 1, s_w_index + c_bits - 1);
              }
            }

            #ifndef __SYNTHESIS__
              #ifdef DEBUG_CONV
                std::cout << "A" << s_index << " " << i_input[s_index][ich_idx] << std::endl;
              #endif
            #endif

          }

          auto acc_group =
            (ich_idx * (c_fh * c_fw) + s_fh * c_fw + s_fw) & c_mask;
          s_acc_simd[acc_group] += s_data * s_b_ext;

        }
      }
    }

    for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
      for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
        auto s_index_r = s_och_pack * c_ow_pack + s_ow_pack;
        for (auto s_simd = 0; s_simd < c_simd; s_simd++) {
          t_acc_simd s_acc_simd_value = 0;
          t_acc_simd s_acc_adj = 0;
          if ((s_index_r > 0))
            s_acc_adj.range(0, 0) = s_acc_simd[s_simd].range(
              c_pad_acc_bits * (s_index_r)-1, c_pad_acc_bits * (s_index_r)-1);
          s_acc_simd_value.range(c_pad_acc_bits - 1, 0) =
            s_acc_simd[s_simd].range(c_pad_acc_bits * (s_index_r + 1) - 1,
                                     c_pad_acc_bits * (s_index_r));
          s_acc[s_index_r] += s_acc_simd_value + s_acc_adj;
        }
      }
    }

    if (num_ich != 0) {
      for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
        for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
          auto s_w_index = s_och_pack * c_ow_pack + s_ow_pack;
          s_acc[s_w_index] += s_acc_base[s_w_index];
        }
      }
    }

    for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
      for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
        auto s_r_index = s_och_pack * c_ow_pack + s_ow_pack;
        auto s_w_index = (ops + s_och_pack) * c_ow_ops + s_ow_ops + s_ow_pack;
        i_acc_buff[s_w_index] = s_acc[s_r_index];
#ifndef __SYNTHESIS__
#ifdef DEBUG_ACC
        std::cout << "ACC " << i_acc_buff[reuse][s_w_index] << std::endl;
#endif
#endif
      }
    }

    if (num_ich == c_ich - c_in_ops) {
      for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
        for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
          if constexpr (std::is_same<t_add, std::nullptr_t>::value ==
                        false) {
            s_output_struct[s_ow_ops + s_ow_pack][ops + s_och_pack] =
              quant_and_add_stream<t_output,
                                   t_output_clip,
                                   t_output_mask,
                                   t_add,
                                   t_acc,
                                   c_relu,
                                   c_leakyrelu>(
                s_acc[s_och_pack * c_ow_pack + s_ow_pack],
                i_add[s_ow_pack + s_ow_ops][ich_idx_add + s_och_pack]);
          } else {
            s_output_struct[s_ow_ops + s_ow_pack][ops + s_och_pack] =
              quant_stream<t_output,
                           t_output_clip,
                           t_output_mask,
                           t_acc,
                           c_relu,
                           c_leakyrelu>(s_acc[s_och_pack * c_ow_pack + s_ow_pack]);
          }
        }
      }
    }
  } else {

    if constexpr(c_ow_pack > 1) {
      
      t_acc s_acc[c_ow_pack];
      t_acc s_acc_base[c_ow_pack];

      ap_uint<48> s_acc_simd[c_simd];
      
      if constexpr(c_depth == 1) {
        for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
          s_acc_base[s_ow_pack] = 0;
        }
      } else {
        for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
          s_acc_base[s_ow_pack] =
            i_acc_buff[ops * c_ow_ops + s_ow_pack + s_ow_ops];
          s_acc[s_ow_pack] = 0;
        }
      }

      if constexpr (std::is_same<t_bias, std::nullptr_t>::value == false) {
        if constexpr (c_depth == 0) {
          if (num_ich == 0) {
            for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
#ifndef __SYNTHESIS__
              // #ifdef DEBUG_CONV
              //   std::cout << "B" << s_bias << " ";
              // #endif
#endif
              s_acc[s_ow_pack] += i_bias[0][ops];
            }
          }
        }
      }

      for (auto s_simd = 0; s_simd < c_simd; s_simd++) {
        s_acc_simd[s_simd] = 0;
      }

      ICH_OPS_OCH_PACKING_LOOP:
      for (auto ich_idx = 0; ich_idx < c_in_ops; ich_idx++) {
        if constexpr(c_depth == 1) {
          for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
            s_acc[s_ow_pack] = 0;
          }
        }

        /* Depthwise convolutions have a bias for each channel. */
        if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
          if constexpr(c_depth == 1) {
            for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
              s_acc[s_ow_pack] = i_bias[0][ich_idx];
              #ifndef __SYNTHESIS__
                // std::cout << "s_acc[" << s_ow_pack << "] " << s_acc[s_ow_pack] << " + " << i_bias[0][ich_idx] << std::endl;
              #endif
            }
          }
        }

        FH_OCH_PACKING_LOOP:
        for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
          FW_OCH_PACKING_LOOP:
          for (auto s_fw = 0; s_fw < c_fw; s_fw++) {
            t_input_mod s_input_ext[c_ow_pack];

            /* Reading the activations */
            for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
              auto s_index = s_fh * FW + s_fw +
                             (c_ow_ops - (s_ow_pack + s_ow_ops) - 1) * c_str;
              s_input_ext[s_ow_pack] = i_input[s_index][ich_idx];
            }

            /* Weight index in the window filter */
            auto s_index = s_fh * c_fw + s_fw;
            auto acc_group =
              (ich_idx * (c_fh * c_fw) + s_fh * c_fw + s_fw) & c_mask;

            s_acc_simd[acc_group] =
              double_packing_wrap<t_input_mod,
                                  t_weight_st,
                                  ap_uint<48>,
                                  c_pad_bits>(s_input_ext[0],
                                              s_input_ext[1],
                                              i_weight[s_index][ich_idx][ops],
                                              s_acc_simd[acc_group]);
          }
        }

        if constexpr(c_depth == 1) {
          for (auto s_simd = 0; s_simd < c_simd; s_simd++) {
            for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {

              t_acc_simd s_acc_simd_value = 0;
              t_acc_simd s_acc_adj = 0;

              /* Recovering the upper dot product as explained in the white paper
              * from Xilinx. https://docs.xilinx.com/v/u/en-US/wp487-int8-acceleration */
              if (s_ow_pack > 0) {
                s_acc_adj[0] =
                  s_acc_simd[s_simd][(c_pad_acc_bits * s_ow_pack) - 1];
              }

              s_acc_simd_value.range(c_pad_acc_bits - 1, 0) =
                s_acc_simd[s_simd].range(c_pad_acc_bits * (s_ow_pack + 1) - 1,
                                          c_pad_acc_bits * s_ow_pack);
              s_acc[s_ow_pack] += s_acc_simd_value + s_acc_adj;
            }
            s_acc_simd[s_simd] = 0;
          }

          for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
            s_output_struct[s_ow_ops + s_ow_pack][ich_idx] =
              quant_stream<t_output,
                           t_output_clip,
                           t_output_mask,
                           t_acc,
                           c_relu,
                           c_leakyrelu>(s_acc[s_ow_pack]);
          }
        }
      }

      if constexpr(c_depth == 0) {
        for (auto s_simd = 0; s_simd < c_simd; s_simd++) {
          for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {

            t_acc_simd s_acc_simd_value = 0;
            t_acc_simd s_acc_adj = 0;

            /* Recovering the upper dot product as explained in the white paper
             * from Xilinx.
             * https://docs.xilinx.com/v/u/en-US/wp487-int8-acceleration */
            if (s_ow_pack > 0) {
              s_acc_adj[0] =
                s_acc_simd[s_simd][(c_pad_acc_bits * s_ow_pack) - 1];
            }

            s_acc_simd_value.range(c_pad_acc_bits - 1, 0) =
              s_acc_simd[s_simd].range(c_pad_acc_bits * (s_ow_pack + 1) - 1,
                                       c_pad_acc_bits * s_ow_pack);
            s_acc[s_ow_pack] += s_acc_simd_value + s_acc_adj;
          }
        }
      }

      if constexpr(c_depth == 0) {
        if (num_ich != 0) {
          for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
            s_acc[s_ow_pack] += s_acc_base[s_ow_pack];
          }
        }
      }

      // If c_depth is 1 then there is no need to store the previous
      // results
      if constexpr(c_depth == 0) {
        for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
          i_acc_buff[ops * c_ow_ops + s_ow_ops + s_ow_pack] =
            s_acc[s_ow_pack];
        }
      }

#ifndef __SYNTHESIS__
#ifdef DEBUG_ACC
      for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
        std::cout << "ACC " << s_acc[s_ow_ops + s_ow_pack] << std::endl;
      }
#endif /* DEBUG_ACC */
#endif /* __SYNTHESIS__ */
      
      if constexpr (c_depth == 0) {
        if (num_ich == c_ich - c_in_ops) {
          for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
            if constexpr (std::is_same<t_add, std::nullptr_t>::value == false) {
              s_output_struct[s_ow_ops + s_ow_pack][ops] =
                quant_and_add_stream<t_output,
                                     t_output_clip,
                                     t_output_mask,
                                     t_add,
                                     t_acc,
                                     c_relu,
                                     c_leakyrelu>(
                  s_acc[s_ow_pack], i_add[s_ow_pack + s_ow_ops][ich_idx_add]);
            } else {
              s_output_struct[s_ow_ops + s_ow_pack][ops] =
                quant_stream<t_output,
                             t_output_clip,
                             t_output_mask,
                             t_acc,
                             c_relu,
                             c_leakyrelu>(s_acc[s_ow_pack]);
            }
          }
        }
      }
    } else {

      t_acc s_acc = 0;
      t_acc s_acc_base = 0;

      // If c_depth is 1 then there is no need to accumulate the previous
      // results
      if constexpr(c_depth == 1) {
        s_acc_base = 0;
      } else {
        s_acc_base = i_acc_buff[ops * c_ow_ops + s_ow_ops];
      }

      if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
        if constexpr(c_depth == 0) {
          if (num_ich == 0) {
            s_acc += i_bias[0][ops];
            #ifndef __SYNTHESIS__
              #ifdef DEBUG_CONV
                std::cout << "B " << i_bias[0][ops] << std::endl;
              #endif
            #endif
          }
        }
      }

      for (auto ich_idx = 0; ich_idx < c_in_ops; ich_idx++) {

        if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
          if constexpr(c_depth == 1) {
              s_acc = i_bias[0][ich_idx];
            #ifndef __SYNTHESIS__
              #ifdef DEBUG_CONV
                std::cout << "B " << i_bias[0][ich_idx] << std::endl;
              #endif
            #endif
          }
        }

        for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
          for (auto s_fw = 0; s_fw < c_fw; s_fw++) {
            auto s_index_act = s_fh*FW+s_fw+(c_ow_ops-s_ow_ops-1)*c_str;
            auto s_index = s_fh*c_fw+s_fw;
            #ifndef __SYNTHESIS__
              #ifdef DEBUG_CONV
                std::cout << "W" << s_index << " " << i_weight[s_index][ich_idx][ops] << " ";
                std::cout << "A" << s_index << " " << i_input[s_index][ich_idx] << " ";
              #endif
              #ifdef DEBUG_ACT
                std::cout << "A" << s_index << " " << s_data << std::endl;
              #endif
            #endif
            t_input_mod s_data = i_input[s_index_act][ich_idx];
            // std::cout << s_acc << "+" << s_data << "*" << i_weight[s_index][ich_idx][ops] << std::endl;
            s_acc += s_data * i_weight[s_index][ich_idx][ops];
            // std::cout << s_acc << "+" << s_data << "*" << i_weight[s_index][ich_idx][ops] << std::endl;
          }
        }

        if constexpr (c_depth == 1) {
          s_output_struct[s_ow_ops][ich_idx] =
            quant_stream<t_output, t_output_clip, t_output_mask, t_acc, c_relu, c_leakyrelu>(
              s_acc);
        }
      }

      if constexpr(c_depth == 0) {
        if (num_ich != 0) s_acc += s_acc_base;
      }

      #ifndef __SYNTHESIS__
        #ifdef DEBUG_ACC
          std::cout <<  "ACC " << s_acc << std::endl;
        #endif
      #endif

      // If c_depth is 1 then there is no need to store the previous
      // results
      if constexpr(c_depth == 0){
        i_acc_buff[ops * c_ow_ops + s_ow_ops] = s_acc;
      }

      if constexpr (c_depth == 0) {
        if (num_ich == c_ich - c_in_ops) {
          if constexpr (std::is_same<t_add, std::nullptr_t>::value ==
                        false) {
            s_output_struct[s_ow_ops][ops] =
              quant_and_add_stream<t_output,
                                   t_output_clip,
                                   t_output_mask,
                                   t_add,
                                   t_acc,
                                   c_relu,
                                   c_leakyrelu>(
                s_acc, i_add[s_ow_ops][ich_idx_add]);
          } else {
            s_output_struct[s_ow_ops][ops] =
              quant_stream<t_output,
                           t_output_clip,
                           t_output_mask,
                           t_acc,
                           c_relu,
                           c_leakyrelu>(s_acc);
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
template<class t_input_struct,
         class t_input,
         class t_input_data,
         class t_input_st,
         class t_weight,
         class t_weight_st,
         class t_bias,
         class t_add_struct,
         class t_add_vector,
         class t_add,
         class t_forward_struct,
         class t_input_mod,
         class t_input_1x1,
         class t_weight_1x1,
         class t_weight_1x1_st,
         class t_bias_1x1,
         class t_acc_struct,
         class t_acc,
         class t_acc_1x1_struct,
         class t_acc_1x1,
         class t_output_struct,
         class t_output_vector,
         class t_output,
         class t_output_clip,
         class t_output_mask,
         class t_output_struct_1x1,
         class t_output_1x1,
         int c_ich,
         int c_och,
         int c_och_1x1,
         int c_oh,
         int c_ow,
         int c_fh,
         int c_fw,
         int c_index,
         int c_str,
         int c_ops,
         int c_ops_out,
         int c_in_ops,
         int c_add_ops,
         int c_ow_ops,
         int c_ow_ops_out,
         int c_relu,
         int c_leakyrelu,
         int c_reuse,
         int c_ow_pack,
         int c_och_pack,
         int c_in_bits,
         int c_in_ibits,
         int c_w_bits,
         int c_w_ibits,
         int c_simd_bits,
         int c_simd,
         int c_mask,
         int c_in_bits_1x1,
         int c_in_ibits_1x1,
         int c_w_bits_1x1,
         int c_w_ibits_1x1,
         int c_simd_bits_1x1,
         int c_simd_1x1,
         int c_mask_1x1,
         int c_depth>
void
conv_comp(hls::stream<t_input_struct> i_input[1],
          hls::stream<t_weight> i_weights[c_index],
          hls::stream<t_bias> i_bias[1],
          hls::stream<t_weight_1x1> i_weights_1x1[1],
          hls::stream<t_bias_1x1> i_bias_1x1[1],
          hls::stream<t_add_struct> i_add[c_ow_ops],
          hls::stream<t_forward_struct> o_forward[c_ow_ops],
          hls::stream<t_output_struct> o_output[c_ow_ops_out],
          hls::stream<t_output_struct_1x1> o_output_1x1[c_ow_ops_out])
{
  // Generic Convolution Computation

  /* We want to use chain of DSPs and not balance the expressions, we are not
   * interested in short pipeline, more on resources */
//#pragma HLS expression_balance off
  // The output ow_ops must be greater or equal and a multliples of ow_ops
  static_assert(c_ow_ops_out >= c_ow_ops, "c_ow_ops_out >= c_ow_ops");
  static_assert(c_ow_ops_out % c_ow_ops == 0, "c_ow_ops_out % c_ow_ops == 0");
  
  if constexpr (c_depth == 0) {
    static_assert(c_ops_out >= c_ops, "c_ops_out >= c_ops");
    static_assert(c_ops_out % c_ops == 0, "c_ops_out % c_ops == 0");
  } else {
    static_assert(c_ops_out >= c_in_ops, "c_ops_out >= c_in_ops");
    static_assert(c_ops_out % c_in_ops == 0, "c_ops_out % c_in_ops == 0");
  }
  
  if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false) {
    // If c_add_ops is not a multiple of 2 then the packing over och_ops could
    // create a mess, since to index the channel of the skip tensor is used och
    // % c_add_ops. Och is a multiple of 2 otherwise there will be no packing.
    // och % c_add_ops must not overflow.
    // static_assert(c_add_ops % 2 == 0 || c_add_ops == 1, "c_add_ops % 2 != 0");
    static_assert(c_add_ops >= c_ops, "c_add_ops < c_ops");
    static_assert(c_add_ops % c_ops == 0, "c_add_ops % c_ops != 0");
    static_assert(c_depth == 0, "Depthwise convolutions with add are not supported");
  }
  
  const auto c_och_depth = (c_depth == 1) ? 1 : c_och;
  const auto c_o_index = c_oh * c_ow / c_ow_ops_out;
  const auto c_reuse_iter = c_reuse / c_ow_ops;
  const auto c_iter = c_reuse_iter;
  const auto c_och_1x1_depth = (c_depth == 1) ? 1 : c_och_1x1;
  const auto c_num_och_1x1 = c_och_1x1_depth / c_ops;
  const auto c_iter_1x1 = c_reuse_iter * c_num_och_1x1;
  const auto c_ops_1x1 = (c_och_1x1 < c_ops) ? c_och_1x1 : c_ops;
  const auto c_iter_ich = (c_depth == 1) ? c_ops_out : c_in_ops;
  const auto c_iter_och = (c_depth == 1) ? 1 : c_ops_out;
  const auto c_iter_ops_out = (c_depth == 1) ? c_in_ops : c_ops;

  constexpr int FW = (c_fw + (c_ow_ops - 1) * c_str);
  // constexpr int MO = (c_fh * FW) / 2;

  /* Groups of accumulator for a total of och * ow_ops. The one that must be
   * available in parallel are c_ops * c_ow_ops. This data structure is not used
   * by depth convolutions */
  t_acc s_acc_buff[c_reuse_iter][c_och_depth / c_ops][c_ops * c_ow_ops];
  t_acc_1x1 s_acc_buff_1x1[c_reuse_iter][c_och_depth / c_ops][c_ops * c_ow_ops];
  auto acc_group = 0;
#pragma HLS array_partition variable = s_acc_buff type = complete dim = 3
#pragma HLS array_partition variable = s_acc_buff_1x1 type = complete dim = 3

  t_input s_input;
#pragma HLS aggregate variable = s_input
#pragma HLS array_partition variable = s_input type = complete
  
  bool s_last = false;
  t_weight s_weight[c_index];
#pragma HLS array_partition variable = s_weight type = complete
  
  t_weight_1x1 s_weight_1x1[1];
#pragma HLS array_partition variable = s_weight_1x1 type = complete
  
  t_bias s_bias;
#pragma HLS array_partition variable = s_bias type = complete dim = 1
  
  t_bias_1x1 s_bias_1x1;
#pragma HLS array_partition variable = s_bias_1x1 type = complete dim = 1
  
  t_add_vector s_add[c_ow_ops];
#pragma HLS array_partition variable = s_add type = complete dim = 0

/* Output struct variable used to pack the output in och_ops_out * c_ow_ops.
 * Divided in och_ops_out / och_ops packets, each of dimension c_ow_ops * c_ops
 * which are written in parallel.
 */
  t_output s_output_vector[c_ops_out / c_iter_ops_out][c_ow_ops]
                          [c_iter_ops_out];
  t_output_1x1 s_output_vector_1x1[c_ops_out / c_iter_ops_out][c_ow_ops]
                                  [c_iter_ops_out];

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the first pipeline
  // round to the higher log2 c_simd
  const int c_pad_bits = 27 - c_in_bits - 1;
  const int c_int_pad_bits = c_simd_bits + c_in_ibits + c_w_ibits;
  const int c_pad_acc_bits = c_simd_bits + c_in_bits + c_w_bits;
  // mask on c_simd bits
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the second pipeline
  // mask on c_simd bits

  const int c_pad_bits_1x1 = 27 - c_in_bits_1x1 - 1;
  const int c_int_pad_bits_1x1 =
    c_simd_bits_1x1 + c_in_ibits_1x1 + c_w_ibits_1x1;
  const int c_pad_acc_bits_1x1 = c_simd_bits_1x1 + c_in_bits_1x1 + c_w_bits_1x1;

  ////////////////////////////////////////////////////////////////////////////

  // TODO: the number of integer bits of the weights type must be considered
  typedef ap_fixed<c_pad_acc_bits, c_int_pad_bits, AP_RND_ZERO, AP_WRAP>
    t_acc_simd;
  typedef ap_fixed<c_pad_acc_bits_1x1, c_int_pad_bits_1x1, AP_RND_ZERO, AP_WRAP>
    t_acc_simd_1x1;
  auto ich_idx_add = 0;
  auto ich_idx_packets = 0;

  /* Stores the iterator of ops in ops_out */
  auto och_packet_in_ops_out = 0;

  #ifndef __SYNTHESIS__
    if (c_depth == 1)
      std::cout << "depth_conv_op " << c_ich << " " << c_och <<  " " << c_reuse_iter << std::endl;
    else
      std::cout << "conv_op " << c_ich << " " << c_ops << " " << c_in_ops << std::endl;
    if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false)
      std::cout << "#### The convolution has add" << std::endl;
    std::cout << "parallelism " << c_ops << " " << c_in_ops << " " << c_ow_ops << std::endl;
    std::cout << "output_stream " << c_ops << " " << c_ow_ops_out << std::endl;
    std::cout << "packing " << c_och_pack << " " << c_ow_pack << std::endl;
    std::cout << "padding " << c_pad_bits << " " << c_int_pad_bits << " " << c_simd_bits << " " << c_pad_acc_bits << " " << c_simd << std::endl;
    std::cout << "input type " << std::is_same<typename t_input_st::Base::Base, _AP_ROOT_TYPE<c_in_bits, true>>::value << std::endl;
    std::cout << "weight type " << std::is_same<typename t_weight_st::Base::Base, _AP_ROOT_TYPE<c_w_bits, true>>::value << std::endl;
    std::cout << "s_input.size() = " << i_input[0].size() << std::endl;
    std::cout << "s_weight.size() = " << i_weights[0].size() << std::endl;
    std::cout << "s_bias.size() = " << i_bias[0].size() << std::endl;
  #endif

  // Iterating over the portion of tensor of each ow_ops_out slice
  CONV_LOOP:
  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {

    // Iterating over each ow_ops_out slice
    OW_OPS_OUT_LOOP:
    for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out;
         s_ow_ops_out += c_reuse) {

      // Iterating over the tensor input channels with steps of input packets
      ICH_LOOP:
      for (auto s_num_ich = 0; s_num_ich < c_ich;
           s_num_ich += c_iter_ich, acc_group = 0) {

        // Iterating over the tensor output channels with steps of output
        // packets
        OCH_LOOP:
        for (auto s_num_och = 0, ich_idx_packets = 0; s_num_och < c_och_depth;
             s_num_och += c_iter_och) {

          // Iterating over single output packet with steps of ops
          OCH_PACKET_LOOP:
            for (auto s_num_ops_out = 0, och_packet_in_ops_out = 0;
                 s_num_ops_out < c_ops_out;
                 s_num_ops_out += c_iter_ops_out,
                      ich_idx_packets++,
                      acc_group++,
                      och_packet_in_ops_out++) {
#pragma HLS pipeline II = 1 style = stp
              for (auto s_iter = 0; s_iter < c_iter; s_iter++) {

              auto s_reuse = s_iter;

              // Reading ich_ops windows of input data each och/c_ops cycles
              if (((s_num_och == 0) && (s_num_ops_out == 0)) ||
                  (c_depth == 1)) {
                t_input_struct s_input_struct = i_input[0].read();
#pragma HLS array_partition variable = s_input_struct.data type = complete
                s_input = s_input_struct.data;
                /* Sending last only at the bottom right data */
                s_last = s_input_struct.last;
#ifndef __SYNTHESIS__
#ifdef DEBUG_INPUT
                constexpr int c_pixels = (c_fh * FW);
                ap_uint<8 * c_in_ops * c_pixels> tmp;
                for (auto s_pixels = 0; s_pixels < c_pixels; s_pixels++) {
                  for (auto s_in_ops = 0; s_in_ops < c_in_ops; s_in_ops++) {
                    tmp.range(8 * (s_pixels * c_in_ops + s_in_ops + 1) - 1,
                              8 * (s_pixels * c_in_ops + s_in_ops)) =
                      s_input[s_pixels][s_in_ops].range(7, 0);
                  }
                }
                std::cout << "inp " << tmp.to_string(16) << std::endl;
#endif /* DEBUG_INPUT */
#endif /* __SYNTHESIS__ */
              }

              /* Buffering to speed up computations */
              /* TODO: Adjust for generic bit quantizations */
              if (s_reuse == 0) {
                for (auto s_index = 0; s_index < c_index; s_index++) {
                  #ifndef __SYNTHESIS__
                    #ifdef DEBUG_WEIGHTS
                      std::cout << "reading s_weight[" << c_index-1-s_index << "]" << std::endl;
                    #endif
                  #endif
                  s_weight[s_index] = i_weights[s_index].read();
                  #ifndef __SYNTHESIS__
                    #ifdef DEBUG_WEIGHTS
                    for (auto s_log_ich = 0; s_log_ich < c_in_ops; s_log_ich++) {
                      for (auto s_log_ops = 0; s_log_ops < c_ops; s_log_ops++) {
                        std::cout << "s_weight[" << c_index-1-s_index << "][" << s_log_ich << "][" << s_log_ops << "] = " << s_weight[c_index - 1 - s_index][s_log_ich][s_log_ops] << std::endl;
                      }
                    }
                    #endif
                  #endif
                }

                // If it is the first reuse iteration, read the 1x1 weights
                if constexpr(std::is_same<t_weight_1x1, std::nullptr_t>::value == false) {
                  s_weight_1x1[0] = i_weights_1x1[0].read();
                }

                if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
                  if ((s_num_ich == 0) || (c_depth == 1)) s_bias = i_bias[0].read();
                }

                if constexpr(std::is_same<t_bias_1x1, std::nullptr_t>::value == false) {
                  if ((s_num_ich == 0) || (c_depth == 1)) s_bias_1x1 = i_bias_1x1[0].read();
                }
              }
              
              // Done to avoid partitioning of the stream and resource wasting
              // Theoretically with the add the input and output dimensions should
              // be equal, so we could perform the addition in different iterations
              if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false){
                // auto s_add_read = (s_num_och + s_num_ops_out) % c_add_ops;
                if (ich_idx_packets == (c_add_ops / c_ops)) {
                  ich_idx_packets = 0;
                }
                if ((ich_idx_packets == 0) && (s_num_ich == c_ich - c_in_ops)){
                  ich_idx_add = 0;
                  for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  // FIX FOR MOBILENETv2: Taking into account different output and
                  // input channels
                    
                    t_add_struct s_add_struct = i_add[s_ow_ops].read();
                    s_add[s_ow_ops] = s_add_struct.data[0];
#ifndef __SYNTHESIS__
#ifdef DEBUG_ADD
                    ap_uint<t_add::width * c_add_ops> tmp = 0;
                    for (auto s_add_ops = 0; s_add_ops < c_add_ops;
                         s_add_ops++) {
                      tmp.range((t_add::width * (s_add_ops + 1)) - 1,
                                t_add::width * s_add_ops) =
                        s_add[s_ow_ops][s_add_ops].range(t_add::width - 1, 0);
                    }
                    std::cout << "add " << tmp.to_string(16) << " " << s_ow_ops
                              << std::endl;
#endif
#endif
#ifndef __SYNTHESIS__
#ifdef DEBUG_ADD
                    for (auto s_ich_idx_add = 0; s_ich_idx_add < c_add_ops;
                         s_ich_idx_add++) {
                      std::cout
                        << "add[" << s_ow_ops << "][" << s_ich_idx_add << "] "
                        << s_add[s_ow_ops].data[0][s_ich_idx_add] << std::endl;
                    }
#endif
#endif
                  }
                }
              }

              std::array<t_input_data, c_ow_ops> s_input_1x1;
              if constexpr (std::is_same<t_acc_1x1_struct,
                                         std::nullptr_t>::value == false) {
#pragma HLS array_partition variable = s_input_1x1 type = complete
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  auto forward_index =
                    (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                  // s_input_1x1[s_ow_ops] = s_input[MO + MO%c_str -
                  // s_ow_ops*c_str];
                  s_input_1x1[c_ow_ops - s_ow_ops - 1] = s_input[forward_index];
                }
              }

              OCH_OPS_LOOP:
              for (auto s_ops = 0; s_ops < c_ops; s_ops += c_och_pack, ich_idx_add+=c_och_pack) {
                  auto s_och = s_num_och + s_num_ops_out + s_ops;
                OW_OPS_LOOP:
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops+=c_ow_pack) {
                  conv_pipe<
                    t_input,
                    t_weight,
                    t_weight_st,
                    t_bias,
                    t_add_vector,
                    t_add,
                    t_input_mod,
                    t_acc_struct,
                    t_acc,
                    t_acc_simd,
                    t_output,
                    t_output_clip,
                    t_output_mask,
                    c_reuse_iter,
                    c_fh,
                    c_fw,
                    c_index,
                    c_str,
                    c_ops,
                    c_iter_ops_out,
                    c_in_ops,
                    c_add_ops,
                    c_ow_ops,
                    c_ow_pack,
                    c_och_pack,
                    c_relu,
                    c_leakyrelu,
                    c_ich,
                    c_och_depth,
                    c_in_bits,
                    c_simd_bits,
                    c_simd,
                    c_pad_bits,
                    c_int_pad_bits,
                    c_pad_acc_bits,
                    c_mask,
                    c_w_bits,
                    c_depth
                  > (
                    s_input,
                    s_weight,
                    s_bias,
                    s_ops,
                    s_num_ich,
                    s_ow_ops,
                    ich_idx_add,
                    s_add,
                    s_acc_buff[s_reuse][acc_group],
                    s_output_vector[och_packet_in_ops_out]
                  );

                  // TODO: split the loop in two parts controlled by different ops options
                  if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
                    if constexpr(c_ops_1x1 != c_ops){
                      if ((s_och > 0) | (s_ops > c_ops_1x1))  continue;
                    }
                    conv_pipe<
                      std::array<t_input_data, c_ow_ops>,
                      t_weight_1x1,
                      t_weight_1x1_st,
                      t_bias_1x1,
                      std::nullptr_t,
                      std::nullptr_t,
                      t_input_1x1,
                      t_acc_1x1_struct,
                      t_acc_1x1,
                      t_acc_simd_1x1,
                      t_output_1x1,
                      std::nullptr_t,
                      std::nullptr_t,
                      c_reuse_iter,
                      1,
                      1,
                      1,
                      1,
                      c_ops,
                      c_iter_ops_out,
                      c_in_ops,
                      c_add_ops,
                      c_ow_ops,
                      c_ow_pack,
                      c_och_pack,
                      0,
                      0,
                      c_ich,
                      c_och_depth,
                      c_in_bits_1x1,
                      c_simd_bits_1x1,
                      c_simd_1x1,
                      c_pad_bits_1x1,
                      c_int_pad_bits_1x1,
                      c_pad_acc_bits_1x1,
                      c_mask_1x1,
                      c_w_bits_1x1,
                      c_depth
                    > (
                      s_input_1x1,
                      s_weight_1x1,
                      s_bias_1x1,
                      s_ops,
                      s_num_ich,
                      s_ow_ops,
                      ich_idx_add,
                      nullptr,
                      s_acc_buff_1x1[s_reuse][acc_group],
                      s_output_vector_1x1[och_packet_in_ops_out]
                    );
                  }
                }
              }

              if constexpr (std::is_same<t_forward_struct,
                                         std::nullptr_t>::value == false) {
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  if ((s_num_ops_out == (c_ops_out - c_iter_ops_out)) &&
                      (s_num_och == (c_och_depth - c_iter_och))) {
                    t_forward_struct s_forward;
                    // auto forward_index = MO + MO%c_str - s_ow_ops*c_str;

                    /* Compute the center of each window in input */
                    auto forward_index =
                      (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                    s_forward.data[0] = s_input[forward_index];
                    o_forward[s_ow_ops_out + s_ow_ops].write(s_forward);
#ifndef __SYNTHESIS__
#ifdef DEBUG_FORWARD
                    for (auto s_log_ich = 0; s_log_ich < c_in_ops;
                         s_log_ich++) {
                      std::cout << "forward[" << s_ow_ops_out + s_ow_ops << "]["
                                << s_log_ich << "] "
                                << s_forward.data[0][s_log_ich] << std::endl;
                    }
#endif /* DEBUG_FORWARD */
#endif /* __SYNTHESIS__ */
                  }
                }
              }

              // Writing in output only when all the ich channels have been
              // considered and the och_ops_out packets are ready
              if (((s_num_ich == (c_ich - c_in_ops)) || (c_depth == 1)) &&
                  (s_num_ops_out == (c_ops_out - c_iter_ops_out))) {

                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  t_output_struct temp_struct;
                  for (auto s_och_ops_out = 0;
                       s_och_ops_out < c_ops_out / c_iter_ops_out;
                       s_och_ops_out++) {
                    for (auto s_och_ops = 0; s_och_ops < c_iter_ops_out;
                         s_och_ops++) {
                      temp_struct
                        .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                        s_output_vector[s_och_ops_out][s_ow_ops][s_och_ops];
                    }
                  }
                  temp_struct.last = s_last;
                  o_output[s_ow_ops_out + s_ow_ops].write(temp_struct);

// #ifndef __SYNTHESIS__
//                   ap_uint<8 * c_ops_out> temp = 0;
//                   for (auto s_tmp = c_ops_out - 1; s_tmp >= 0; s_tmp--) {
//                     temp <<= 8;
//                     temp.range(7, 0) = temp_struct.data[0][s_tmp].range(7, 0);
//                   }
//                   std::cout << temp.to_string(16) << " " << s_ow_ops + s_ow_ops_out << std::endl;
// #endif
                  if constexpr (std::is_same<t_output_struct_1x1,
                                             std::nullptr_t>::value == false) {
                    t_output_struct_1x1 temp_struct_1x1;
                    for (auto s_och_ops_out = 0;
                         s_och_ops_out < c_ops_out / c_iter_ops_out;
                         s_och_ops_out++) {
                      for (auto s_och_ops = 0; s_och_ops < c_iter_ops_out;
                           s_och_ops++) {
                        temp_struct_1x1
                          .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                          s_output_vector_1x1[s_och_ops_out][s_ow_ops]
                                             [s_och_ops];
                      }
                    }
                    if (s_iter < c_iter_1x1) {
                      o_output_1x1[s_ow_ops_out + s_ow_ops].write(
                        temp_struct_1x1);
// #ifndef __SYNTHESIS__
//                       ap_uint<8 * c_ops_out> temp = 0;
//                       for (auto s_tmp = c_ops_out - 1; s_tmp >= 0; s_tmp--) {
//                         temp <<= 8;
//                         temp.range(7, 0) = temp_struct_1x1.data[0][s_tmp].range(7, 0);
//                       }
//                       std::cout << temp.to_string(16) << " " << s_ow_ops + s_ow_ops_out << " 1x1" << std::endl;
// #endif
                    }
                  }
                }
              }
            }
          }
        }
      }

#ifndef __SYNTHESIS__
        #ifdef DEBUG_RES
          for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
            for (auto s_och = 0; s_och < c_och_depth; s_och++) {
              std::cout <<  "PRE RES " << s_acc_buff[0][s_och*c_ow_ops+s_ow_ops] << std::endl;
              auto s_acc_log = quant_stream<
                t_output, t_output_clip, t_output_mask, t_acc, c_relu, c_leakyrelu
              >(s_acc_buff[0][s_och*c_ow_ops+s_ow_ops]);
              std::cout <<  "RES " << s_acc_log << std::endl;
            }
            if constexpr(std::is_same<t_output_struct_1x1, std::nullptr_t>::value == false) {
              for (auto s_och = 0; s_och < c_och_depth; s_och++) {
                auto s_acc_log = quant_stream<
                  t_output_1x1, std::nullptr_t, std::nullptr_t, t_acc_1x1, 0, 0
                >(s_acc_1x1_buff[0][s_och*c_ow_ops+s_ow_ops]);
                std::cout <<  "RES 1x1 " << s_acc_log << std::endl;
              }
            }
          }
        #endif /* DEBUG_RES */
      #endif /* __SYNTHESIS__ */
    }
  }

#ifndef __SYNTHESIS__
  std::cout << i_bias[0].size() << std::endl;
  // Check if input stream is empty
#ifndef SKIP_ASSERTION
  for (auto s_ow_ops = 0; s_ow_ops < 1; s_ow_ops++) {
    if (i_input[s_ow_ops].size() != 0) {
      std::cout << "ERROR: The input stream is not empty" << std::endl;
      std::cout << "i_input[" << s_ow_ops
                << "].size() = " << i_input[s_ow_ops].size() << std::endl;
      assert(false);
    }
  }
  // Check if input add is empty
  for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
    if constexpr (std::is_same<t_add_struct, std::nullptr_t>::value == false) {
      if (i_add[s_ow_ops].size() != 0) {
        std::cout << "ERROR: The input add stream is not empty" << std::endl;
        std::cout << "i_add[" << s_ow_ops
                  << "].size() = " << i_add[s_ow_ops].size() << std::endl;
        assert(false);
      }
    }
  }
  for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out; s_ow_ops_out++) {
    std::cout << "o_output[" << s_ow_ops_out
              << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
    if (o_output[s_ow_ops_out].size() == 0) {
      std::cout << "ERROR: The output stream is empty" << std::endl;
      std::cout << "o_output[" << s_ow_ops_out
                << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
      assert(false);
    }
  }
#endif /* SKIP_ASSERTION */
  if (c_depth == 1)
    std::cout << "end depth_conv_op " << c_ich << " " << c_depth << std::endl;
  else
    std::cout << "end conv_op " << c_ich << " " << c_depth << std::endl;
#endif /* __SYNTHESIS__ */
}

template<typename t_input_struct, // input activation struct type
         typename t_input,        // input activation matrix type
         typename t_input_data,   // input activation vector type without window
         typename t_input_st,     // input activation standard type
         typename t_weight,       // weight matrix type
         typename t_weight_st,    // weight standard type
         typename t_bias,         // bias vector type
         typename t_bias_st,      // bias standard type
         typename t_add_struct,   // add activation struct type
         typename t_add_vector,   // add activation vector type
         typename t_add_st,       // add activation standard type
         typename t_forward_struct, // forward activation struct type
         typename t_input_mod,      // input activation modified type (input
         // quantization)
         typename t_input_1x1,     // input activation 1x1 type
         typename t_weight_1x1,    // weight 1x1 matrix type
         typename t_weight_1x1_st, // weight 1x1 standard type
         typename t_bias_1x1,      // bias 1x1 vector type
         typename t_bias_1x1_st,   // bias 1x1 standard type
         typename t_acc_struct,
         typename t_acc,
         typename t_acc_1x1_struct,
         typename t_acc_1x1,
         typename t_output_struct,
         typename t_output_vector,
         typename t_output,
         typename t_output_clip,
         typename t_output_mask,
         typename t_output_struct_1x1,
         typename t_output_1x1,
         typename p_stream_t,
         const int c_ich_groups,       // input channels divided by groups
         const int c_ich,
         const int c_och,
         const int c_och_1x1,
         const int c_oh,
         const int c_ow,
         const int c_fh,
         const int c_fw,
         const int c_index,
         const int c_str,
         const int c_och_ops,
         const int c_och_ops_out,
         const int c_ich_ops,
         const int c_add_ops,
         const int c_ow_ops,
         const int c_ow_ops_out,
         const int c_bias_ops,
         const int c_reuse,
         const int c_ow_pack,
         const int c_och_pack,
         const int c_in_bits,
         const int c_in_ibits,
         const int c_w_bits,
         const int c_w_ibits,
         const int c_simd_bits,
         const int c_simd,
         const int c_mask,
         const int c_in_bits_1x1,
         const int c_in_ibits_1x1,
         const int c_w_bits_1x1,
         const int c_w_ibits_1x1,
         const int c_simd_bits_1x1,
         const int c_simd_1x1,
         const int c_mask_1x1,
         const int c_relu,
         const int c_leakyrelu,
         const int c_depth>
void
conv_comp_onchip_OW_OPS_OUT(
  t_bias_st mem_bias[c_och / c_bias_ops][c_bias_ops],
  t_weight_st mem_weights[c_fh * c_fw][c_och * c_ich_groups / (c_och_ops * c_ich_ops)]
                         [c_och_ops * c_ich_ops],
  t_bias_1x1_st mem_bias_1x1[c_och_1x1 / c_bias_ops][c_bias_ops],
  t_weight_1x1_st mem_weights_1x1[1][c_och_1x1 * c_ich_groups / (c_och_ops * c_ich_ops)]
                                 [c_och_ops * c_ich_ops],

  hls::stream<t_input_struct> i_input[1],
  hls::stream<t_add_struct> i_add[c_ow_ops],
  hls::stream<t_forward_struct> o_forward[c_ow_ops],
  hls::stream<t_output_struct> o_output[c_ow_ops_out],
  hls::stream<t_output_struct_1x1> o_output_1x1[c_ow_ops_out])
{
  // Generic Convolution Computation

  /* We want to use chain of DSPs and not balance the expressions, we are not
   * interested in short pipeline, more on resources */
//#pragma HLS expression_balance off

  const auto c_och_depth = (c_depth == 1) ? 1 : c_och;
  const auto c_o_index = c_oh * c_ow / c_ow_ops_out;
  const auto c_reuse_iter = c_reuse / c_ow_ops;
  const auto c_iter = c_reuse_iter;
  const auto c_och_1x1_depth = (c_depth == 1) ? 1 : c_och_1x1;
  const auto c_num_och_1x1 = c_och_1x1_depth / c_och_ops;
  const auto c_iter_1x1 = c_reuse_iter * c_num_och_1x1;
  const auto c_ops_1x1 = (c_och_1x1 < c_och_ops) ? c_och_1x1 : c_och_ops;
  const auto c_iter_ich = (c_depth == 1) ? c_och_ops_out : c_ich_ops;
  const auto c_iter_och = (c_depth == 1) ? 1 : c_och_ops_out;
  const auto c_iter_ops_out = (c_depth == 1) ? c_ich_ops : c_och_ops;
  constexpr int FW = (c_fw + (c_ow_ops - 1) * c_str);

  /* Groups of accumulator for a total of och * ow_ops. The one that must be
   * available in parallel are c_och_ops * c_ow_ops. This data structure is not used
   * by depth convolutions */
  t_acc s_acc_buff[c_reuse_iter][c_och_depth / c_och_ops][c_och_ops * c_ow_ops];
  t_acc_1x1 s_acc_buff_1x1[c_reuse_iter][c_och_depth / c_och_ops][c_och_ops * c_ow_ops];
  auto acc_group = 0;
#pragma HLS array_partition variable = s_acc_buff type = complete dim = 3
#pragma HLS array_partition variable = s_acc_buff_1x1 type = complete dim = 3

  t_input s_input;
#pragma HLS aggregate variable = s_input
#pragma HLS array_partition variable = s_input type = complete
  
  bool s_last = false;
  t_weight s_weight[c_index];
#pragma HLS array_partition variable = s_weight type = complete
  
  t_weight_1x1 s_weight_1x1[1];
#pragma HLS array_partition variable = s_weight_1x1 type = complete
  
  t_bias s_bias;
#pragma HLS array_partition variable = s_bias type = complete dim = 1
  
  t_bias_1x1 s_bias_1x1;
#pragma HLS array_partition variable = s_bias_1x1 type = complete dim = 1
  
  t_add_vector s_add[c_ow_ops];
#pragma HLS array_partition variable = s_add type = complete dim = 0

/* Output struct variable used to pack the output in och_ops_out * c_ow_ops.
 * Divided in och_ops_out / och_ops packets, each of dimension c_ow_ops * c_och_ops
 * which are written in parallel. */
  t_output s_output_vector[c_och_ops_out / c_iter_ops_out][c_ow_ops]
                          [c_iter_ops_out];
  t_output_1x1 s_output_vector_1x1[c_och_ops_out / c_iter_ops_out][c_ow_ops]
                                  [c_iter_ops_out];

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the first pipeline
  // round to the higher log2 c_simd
  const int c_pad_bits = 27 - c_in_bits - 1;
  const int c_int_pad_bits = c_simd_bits + c_in_ibits + c_w_ibits;
  const int c_pad_acc_bits = c_simd_bits + c_in_bits + c_w_bits;
  // mask on c_simd bits
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the second pipeline
  // mask on c_simd bits
  const int c_pad_bits_1x1 = 27 - c_in_bits_1x1 - 1;
  const int c_int_pad_bits_1x1 =
    c_simd_bits_1x1 + c_in_ibits_1x1 + c_w_ibits_1x1;
  const int c_pad_acc_bits_1x1 = c_simd_bits_1x1 + c_in_bits_1x1 + c_w_bits_1x1;
  ////////////////////////////////////////////////////////////////////////////

  // TODO: the number of integer bits of the weights type must be considered
  typedef ap_fixed<c_pad_acc_bits, c_int_pad_bits, AP_RND_ZERO, AP_WRAP>
    t_acc_simd;
  typedef ap_fixed<c_pad_acc_bits_1x1, c_int_pad_bits_1x1, AP_RND_ZERO, AP_WRAP>
    t_acc_simd_1x1;
  auto ich_idx_add = 0;
  auto ich_idx_packets = 0;

  /* Stores the iterator of ops in ops_out */
  auto och_packet_in_ops_out = 0;

  /* Stores the iterator of weights packet in memory */
  auto s_weight_index = 0;
  auto s_bias_index = 0;

  // Iterating over the portion of tensor of each ow_ops_out slice
  CONV_LOOP:
  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {

    // Iterating over each ow_ops_out slice
    OW_OPS_OUT_LOOP:
    for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out;
         s_ow_ops_out += c_reuse, s_weight_index = 0, s_bias_index = 0) {
#pragma HLS pipeline II = 1

      // Iterating over the tensor input channels with steps of input packets
      ICH_LOOP:
      for (auto s_num_ich = 0; s_num_ich < c_ich;
           s_num_ich += c_iter_ich, acc_group = 0) {

        // Iterating over the tensor output channels with steps of output
        // packets
        OCH_LOOP:
        for (auto s_num_och = 0, ich_idx_packets = 0; s_num_och < c_och_depth;
             s_num_och += c_iter_och) {

          // Iterating over single output packet with steps of ops
          OCH_PACKET_LOOP:
            for (auto s_num_ops_out = 0, och_packet_in_ops_out = 0;
                 s_num_ops_out < c_och_ops_out;
                 s_num_ops_out += c_iter_ops_out,
                      ich_idx_packets++,
                      acc_group++,
                      och_packet_in_ops_out++) {

              for (auto s_iter = 0; s_iter < c_iter; s_iter++) {

                auto s_reuse = s_iter;

                // Reading ich_ops windows of input data each och/c_och_ops
                // cycles
                if (((s_num_och == 0) && (s_num_ops_out == 0)) ||
                    (c_depth == 1)) {
                  t_input_struct s_input_struct = i_input[0].read();
#pragma HLS array_partition variable = s_input_struct.data type = complete
                  s_input = s_input_struct.data;
                  /* Sending last only at the bottom right data */
                  s_last = s_input_struct.last;
#ifndef __SYNTHESIS__
#ifdef DEBUG_INPUT
                  constexpr int c_pixels = (c_fh * FW);
                  ap_uint<8 * c_ich_ops * c_pixels> tmp;
                  for (auto s_pixels = 0; s_pixels < c_pixels; s_pixels++) {
                    for (auto s_in_ops = 0; s_in_ops < c_ich_ops; s_in_ops++) {
                      tmp.range(8 * (s_pixels * c_ich_ops + s_in_ops + 1) - 1,
                                8 * (s_pixels * c_ich_ops + s_in_ops)) =
                        s_input[s_pixels][s_in_ops].range(7, 0);
                    }
                  }
                  std::cout << "inp " << tmp.to_string(16) << std::endl;
#endif /* DEBUG_INPUT */
#endif /* __SYNTHESIS__ */
                }

                /* Buffering to speed up computations */
                /* TODO: Adjust for generic bit quantizations */
                if (s_reuse == 0) {
                  for (auto s_index = 0; s_index < c_index; s_index++) {
                    for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops;
                         s_ich_ops++) {
                      for (auto s_ops = 0; s_ops < c_och_ops; s_ops++) {
                        auto s_ops_index = s_ich_ops * c_och_ops + s_ops;
                        s_weight[s_index][s_ich_ops][s_ops] =
                          mem_weights[s_index][s_weight_index][s_ops_index];
                      }
                    }
                  }

                  // If it is the first reuse iteration, read the 1x1 weights
                  if constexpr (std::is_same<t_weight_1x1,
                                             std::nullptr_t>::value == false) {
                    for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops;
                         s_ich_ops++) {
                      for (auto s_ops = 0; s_ops < c_och_ops; s_ops++) {
                        auto s_ops_index = s_ich_ops * c_och_ops + s_ops;
                        s_weight_1x1[0][s_ich_ops][s_ops] =
                          mem_weights_1x1[0][s_weight_index][s_ops_index];
                      }
                    }
                  }

                  if (s_num_ich == 0 || c_depth == 1) {
                    if constexpr (std::is_same<t_bias, std::nullptr_t>::value ==
                                  false) {
                      for (auto s_ops = 0; s_ops < c_bias_ops; s_ops++) {
                        s_bias[0][s_ops] = mem_bias[s_bias_index][s_ops];
                      }
                      if constexpr (std::is_same<t_bias_1x1,
                                                 std::nullptr_t>::value ==
                                    false) {
                        for (auto s_ops = 0; s_ops < c_bias_ops; s_ops++) {
                          s_bias_1x1[0][s_ops] =
                            mem_bias_1x1[s_bias_index][s_ops];
                        }
                      }
                    }
                    s_bias_index++;
                  }

                  s_weight_index++;
                }

                // Done to avoid partitioning of the stream and resource wasting
                // Theoretically with the add the input and output dimensions
                // should be equal, so we could perform the addition in
                // different iterations
                if constexpr (std::is_same<t_add_struct,
                                           std::nullptr_t>::value == false) {
                  // auto s_add_read = (s_num_och + s_num_ops_out) % c_add_ops;
                  if (ich_idx_packets == (c_add_ops / c_och_ops)) {
                    ich_idx_packets = 0;
                  }
                  if ((ich_idx_packets == 0) &&
                      (s_num_ich == c_ich - c_ich_ops)) {
                    ich_idx_add = 0;
                    for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                      // FIX FOR MOBILENETv2: Taking into account different
                      // output and input channels

                      t_add_struct s_add_struct = i_add[s_ow_ops].read();
                      s_add[s_ow_ops] = s_add_struct.data[0];
#ifndef __SYNTHESIS__
#ifdef DEBUG_ADD
                      ap_uint<t_add_st::width * c_add_ops> tmp = 0;
                      for (auto s_add_ops = 0; s_add_ops < c_add_ops;
                           s_add_ops++) {
                        tmp.range((t_add_st::width * (s_add_ops + 1)) - 1,
                                  t_add_st::width * s_add_ops) =
                          s_add[s_ow_ops][s_add_ops].range(t_add_st::width - 1,
                                                           0);
                      }
                      std::cout << "add " << tmp.to_string(16) << " "
                                << s_ow_ops << std::endl;
#endif
#endif
#ifndef __SYNTHESIS__
#ifdef DEBUG_ADD
                      for (auto s_ich_idx_add = 0; s_ich_idx_add < c_add_ops;
                           s_ich_idx_add++) {
                        std::cout << "add[" << s_ow_ops << "][" << s_ich_idx_add
                                  << "] "
                                  << s_add[s_ow_ops].data[0][s_ich_idx_add]
                                  << std::endl;
                      }
#endif
#endif
                  }
                }
              }

              std::array<t_input_data, c_ow_ops> s_input_1x1;
              if constexpr (std::is_same<t_acc_1x1_struct,
                                         std::nullptr_t>::value == false) {
#pragma HLS array_partition variable = s_input_1x1 type = complete
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  auto forward_index =
                    (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                  // s_input_1x1[s_ow_ops] = s_input[MO + MO%c_str -
                  // s_ow_ops*c_str];
                  s_input_1x1[c_ow_ops - s_ow_ops - 1] = s_input[forward_index];
                }
              }

              OCH_OPS_LOOP:
              for (auto s_ops = 0; s_ops < c_och_ops; s_ops += c_och_pack, ich_idx_add+=c_och_pack) {
                  auto s_och = s_num_och + s_num_ops_out + s_ops;
                OW_OPS_LOOP:
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops+=c_ow_pack) {
                  conv_pipe<
                    t_input,
                    t_weight,
                    t_weight_st,
                    t_bias,
                    t_add_vector,
                    t_add_st,
                    t_input_mod,
                    t_acc_struct,
                    t_acc,
                    t_acc_simd,
                    t_output,
                    t_output_clip,
                    t_output_mask,
                    c_reuse_iter,
                    c_fh,
                    c_fw,
                    c_index,
                    c_str,
                    c_och_ops,
                    c_iter_ops_out,
                    c_ich_ops,
                    c_add_ops,
                    c_ow_ops,
                    c_ow_pack,
                    c_och_pack,
                    c_relu,
                    c_leakyrelu,
                    c_ich,
                    c_och_depth,
                    c_in_bits,
                    c_simd_bits,
                    c_simd,
                    c_pad_bits,
                    c_int_pad_bits,
                    c_pad_acc_bits,
                    c_mask,
                    c_w_bits,
                    c_depth
                  > (
                    s_input,
                    s_weight,
                    s_bias,
                    s_ops,
                    s_num_ich,
                    s_ow_ops,
                    ich_idx_add,
                    s_add,
                    s_acc_buff[s_reuse][acc_group],
                    s_output_vector[och_packet_in_ops_out]
                  );

                  // TODO: split the loop in two parts controlled by different ops options
                  if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
                    if constexpr(c_ops_1x1 != c_och_ops){
                      if ((s_och > 0) | (s_ops > c_ops_1x1))  continue;
                    }
                    conv_pipe<
                      std::array<t_input_data, c_ow_ops>,
                      t_weight_1x1,
                      t_weight_1x1_st,
                      t_bias_1x1,
                      std::nullptr_t,
                      std::nullptr_t,
                      t_input_1x1,
                      t_acc_1x1_struct,
                      t_acc_1x1,
                      t_acc_simd_1x1,
                      t_output_1x1,
                      std::nullptr_t,
                      std::nullptr_t,
                      c_reuse_iter,
                      1,
                      1,
                      1,
                      1,
                      c_och_ops,
                      c_iter_ops_out,
                      c_ich_ops,
                      c_add_ops,
                      c_ow_ops,
                      c_ow_pack,
                      c_och_pack,
                      0,
                      0,
                      c_ich,
                      c_och_depth,
                      c_in_bits_1x1,
                      c_simd_bits_1x1,
                      c_simd_1x1,
                      c_pad_bits_1x1,
                      c_int_pad_bits_1x1,
                      c_pad_acc_bits_1x1,
                      c_mask_1x1,
                      c_w_bits_1x1,
                      c_depth
                    > (
                      s_input_1x1,
                      s_weight_1x1,
                      s_bias_1x1,
                      s_ops,
                      s_num_ich,
                      s_ow_ops,
                      ich_idx_add,
                      nullptr,
                      s_acc_buff_1x1[s_reuse][acc_group],
                      s_output_vector_1x1[och_packet_in_ops_out]
                    );
                  }
                }
              }

              if constexpr (std::is_same<t_forward_struct,
                                         std::nullptr_t>::value == false) {
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  if ((s_num_ops_out == (c_och_ops_out - c_iter_ops_out)) &&
                      (s_num_och == (c_och_depth - c_iter_och))) {
                    t_forward_struct s_forward;
                    // auto forward_index = MO + MO%c_str - s_ow_ops*c_str;

                    /* Compute the center of each window in input */
                    auto forward_index =
                      (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                    s_forward.data[0] = s_input[forward_index];
                    o_forward[s_ow_ops_out + s_ow_ops].write(s_forward);
#ifndef __SYNTHESIS__
#ifdef DEBUG_FORWARD
                    for (auto s_log_ich = 0; s_log_ich < c_ich_ops;
                         s_log_ich++) {
                      std::cout << "forward[" << s_ow_ops_out + s_ow_ops << "]["
                                << s_log_ich << "] "
                                << s_forward.data[0][s_log_ich] << std::endl;
                    }
#endif /* DEBUG_FORWARD */
#endif /* __SYNTHESIS__ */
                  }
                }
              }

              // Writing in output only when all the ich channels have been
              // considered and the och_ops_out packets are ready
              if (((s_num_ich == (c_ich - c_ich_ops)) || (c_depth == 1)) &&
                  (s_num_ops_out == (c_och_ops_out - c_iter_ops_out))) {

                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  t_output_struct temp_struct;
                  for (auto s_och_ops_out = 0;
                       s_och_ops_out < c_och_ops_out / c_iter_ops_out;
                       s_och_ops_out++) {
                    for (auto s_och_ops = 0; s_och_ops < c_iter_ops_out;
                         s_och_ops++) {
                      temp_struct
                        .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                        s_output_vector[s_och_ops_out][s_ow_ops][s_och_ops];
                    }
                  }
                  temp_struct.last = s_last;
                  o_output[s_ow_ops_out + s_ow_ops].write(temp_struct);

// #ifndef __SYNTHESIS__
//                   ap_uint<8 * c_och_ops_out> temp = 0;
//                   for (auto s_tmp = c_och_ops_out - 1; s_tmp >= 0; s_tmp--) {
//                     temp <<= 8;
//                     temp.range(7, 0) = temp_struct.data[0][s_tmp].range(7, 0);
//                   }
//                   std::cout << temp.to_string(16) << " " << s_ow_ops + s_ow_ops_out << std::endl;
// #endif
                  if constexpr (std::is_same<t_output_struct_1x1,
                                             std::nullptr_t>::value == false) {
                    t_output_struct_1x1 temp_struct_1x1;
                    for (auto s_och_ops_out = 0;
                         s_och_ops_out < c_och_ops_out / c_iter_ops_out;
                         s_och_ops_out++) {
                      for (auto s_och_ops = 0; s_och_ops < c_iter_ops_out;
                           s_och_ops++) {
                        temp_struct_1x1
                          .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                          s_output_vector_1x1[s_och_ops_out][s_ow_ops]
                                             [s_och_ops];
                      }
                    }
                    if (s_iter < c_iter_1x1) {
                      o_output_1x1[s_ow_ops_out + s_ow_ops].write(
                        temp_struct_1x1);
// #ifndef __SYNTHESIS__
//                       ap_uint<8 * c_och_ops_out> temp = 0;
//                       for (auto s_tmp = c_och_ops_out - 1; s_tmp >= 0; s_tmp--) {
//                         temp <<= 8;
//                         temp.range(7, 0) = temp_struct_1x1.data[0][s_tmp].range(7, 0);
//                       }
//                       std::cout << temp.to_string(16) << " " << s_ow_ops + s_ow_ops_out << " 1x1" << std::endl;
// #endif
                    }
                  }
                }
              }
            }
          }
        }
      }

#ifndef __SYNTHESIS__
        #ifdef DEBUG_RES
          for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
            for (auto s_och = 0; s_och < c_och_depth; s_och++) {
              std::cout <<  "PRE RES " << s_acc_buff[0][s_och*c_ow_ops+s_ow_ops] << std::endl;
              auto s_acc_log = quant_stream<
                t_output, t_output_clip, t_output_mask, t_acc, c_relu, c_leakyrelu,
              >(s_acc_buff[0][s_och*c_ow_ops+s_ow_ops]);
              std::cout <<  "RES " << s_acc_log << std::endl;
            }
            if constexpr(std::is_same<t_output_struct_1x1, std::nullptr_t>::value == false) {
              for (auto s_och = 0; s_och < c_och_depth; s_och++) {
                auto s_acc_log = quant_stream<
                  t_output_1x1, std::nullptr_t, std::nullptr_t, t_acc_1x1, 0
                >(s_acc_1x1_buff[0][s_och*c_ow_ops+s_ow_ops]);
                std::cout <<  "RES 1x1 " << s_acc_log << std::endl;
              }
            }
          }
        #endif /* DEBUG_RES */
      #endif /* __SYNTHESIS__ */
    }
  }

#ifndef __SYNTHESIS__
  // Check if input stream is empty
#ifndef SKIP_ASSERTION
  for (auto s_ow_ops = 0; s_ow_ops < 1; s_ow_ops++) {
    if (i_input[s_ow_ops].size() != 0) {
      std::cout << "ERROR: The input stream is not empty" << std::endl;
      std::cout << "i_input[" << s_ow_ops
                << "].size() = " << i_input[s_ow_ops].size() << std::endl;
      assert(false);
    }
  }
  // Check if input add is empty
  for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
    if constexpr (std::is_same<t_add_struct, std::nullptr_t>::value == false) {
      if (i_add[s_ow_ops].size() != 0) {
        std::cout << "ERROR: The input add stream is not empty" << std::endl;
        std::cout << "i_add[" << s_ow_ops
                  << "].size() = " << i_add[s_ow_ops].size() << std::endl;
        assert(false);
      }
    }
  }
  for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out; s_ow_ops_out++) {
    std::cout << "o_output[" << s_ow_ops_out
              << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
    if (o_output[s_ow_ops_out].size() == 0) {
      std::cout << "ERROR: The output stream is empty" << std::endl;
      std::cout << "o_output[" << s_ow_ops_out
                << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
      assert(false);
    }
  }
#endif /* SKIP_ASSERTION */
  if (c_depth == 1)
    std::cout << "end depth_conv_op " << c_ich << " " << c_depth << std::endl;
  else
    std::cout << "end conv_op " << c_ich << " " << c_depth << std::endl;
#endif /* __SYNTHESIS__ */
  }


template<typename t_input_struct, // input activation struct type
         typename t_input,        // input activation matrix type
         typename t_input_data,   // input activation vector type without window
         typename t_input_st,     // input activation standard type
         typename t_weight,       // weight matrix type
         typename t_weight_st,    // weight standard type
         typename t_bias,         // bias vector type
         typename t_bias_st,      // bias standard type
         typename t_add_struct,   // add activation struct type
         typename t_add_vector,   // add activation vector type
         typename t_add_st,       // add activation standard type
         typename t_forward_struct, // forward activation struct type
         typename t_input_mod,      // input activation modified type (input
         // quantization)
         typename t_input_1x1,     // input activation 1x1 type
         typename t_weight_1x1,    // weight 1x1 matrix type
         typename t_weight_1x1_st, // weight 1x1 standard type
         typename t_bias_1x1,      // bias 1x1 vector type
         typename t_bias_1x1_st,   // bias 1x1 standard type
         typename t_acc_struct,
         typename t_acc,
         typename t_acc_1x1_struct,
         typename t_acc_1x1,
         typename t_output_struct,
         typename t_output_vector,
         typename t_output,
         typename t_output_clip,
         typename t_output_mask,
         typename t_output_struct_1x1,
         typename t_output_1x1,
         typename p_stream_t,
         const int c_ich_groups,       // input channels divided by groups
         const int c_ich,
         const int c_och,
         const int c_och_1x1,
         const int c_oh,
         const int c_ow,
         const int c_fh,
         const int c_fw,
         const int c_index,
         const int c_str,
         const int c_och_ops,
         const int c_och_ops_out,
         const int c_ich_ops,
         const int c_add_ops,
         const int c_ow_ops,
         const int c_ow_ops_out,
         const int c_bias_ops,
         const int c_reuse,
         const int c_ow_pack,
         const int c_och_pack,
         const int c_in_bits,
         const int c_in_ibits,
         const int c_w_bits,
         const int c_w_ibits,
         const int c_simd_bits,
         const int c_simd,
         const int c_mask,
         const int c_in_bits_1x1,
         const int c_in_ibits_1x1,
         const int c_w_bits_1x1,
         const int c_w_ibits_1x1,
         const int c_simd_bits_1x1,
         const int c_simd_1x1,
         const int c_mask_1x1,
         const int c_relu,
         const int c_leakyrelu,
         const int c_depth>
void
conv_comp_onchip_ICH(
  t_bias_st mem_bias[c_och / c_bias_ops][c_bias_ops],
  t_weight_st mem_weights[c_fh * c_fw][c_och * c_ich_groups / (c_och_ops * c_ich_ops)]
                         [c_och_ops * c_ich_ops],
  t_bias_1x1_st mem_bias_1x1[c_och_1x1 / c_bias_ops][c_bias_ops],
  t_weight_1x1_st mem_weights_1x1[1][c_och_1x1 * c_ich_groups / (c_och_ops * c_ich_ops)]
                                 [c_och_ops * c_ich_ops],

  hls::stream<t_input_struct> i_input[1],
  hls::stream<t_add_struct> i_add[c_ow_ops],
  hls::stream<t_forward_struct> o_forward[c_ow_ops],
  hls::stream<t_output_struct> o_output[c_ow_ops_out],
  hls::stream<t_output_struct_1x1> o_output_1x1[c_ow_ops_out])
{
  // Generic Convolution Computation

  /* We want to use chain of DSPs and not balance the expressions, we are not
   * interested in short pipeline, more on resources */
//#pragma HLS expression_balance off

  const auto c_och_depth = (c_depth == 1) ? 1 : c_och;
  const auto c_o_index = c_oh * c_ow / c_ow_ops_out;
  const auto c_reuse_iter = c_reuse / c_ow_ops;
  const auto c_iter = c_reuse_iter;
  const auto c_och_1x1_depth = (c_depth == 1) ? 1 : c_och_1x1;
  const auto c_num_och_1x1 = c_och_1x1_depth / c_och_ops;
  const auto c_iter_1x1 = c_reuse_iter * c_num_och_1x1;
  const auto c_ops_1x1 = (c_och_1x1 < c_och_ops) ? c_och_1x1 : c_och_ops;
  const auto c_iter_ich = (c_depth == 1) ? c_och_ops_out : c_ich_ops;
  const auto c_iter_och = (c_depth == 1) ? 1 : c_och_ops_out;
  const auto c_iter_ops_out = (c_depth == 1) ? c_ich_ops : c_och_ops;
  constexpr int FW = (c_fw + (c_ow_ops - 1) * c_str);

  /* Groups of accumulator for a total of och * ow_ops. The one that must be
   * available in parallel are c_och_ops * c_ow_ops. This data structure is not used
   * by depth convolutions */
  t_acc s_acc_buff[c_reuse_iter][c_och_depth / c_och_ops][c_och_ops * c_ow_ops];
  t_acc_1x1 s_acc_buff_1x1[c_reuse_iter][c_och_depth / c_och_ops][c_och_ops * c_ow_ops];
  auto acc_group = 0;
#pragma HLS array_partition variable = s_acc_buff type = complete dim = 3
#pragma HLS array_partition variable = s_acc_buff_1x1 type = complete dim = 3

  t_input s_input;
#pragma HLS aggregate variable = s_input
#pragma HLS array_partition variable = s_input type = complete
  
  bool s_last = false;
  t_weight s_weight[c_index];
#pragma HLS array_partition variable = s_weight type = complete
  
  t_weight_1x1 s_weight_1x1[1];
#pragma HLS array_partition variable = s_weight_1x1 type = complete
  
  t_bias s_bias;
#pragma HLS array_partition variable = s_bias type = complete dim = 1
  
  t_bias_1x1 s_bias_1x1;
#pragma HLS array_partition variable = s_bias_1x1 type = complete dim = 1
  
  t_add_vector s_add[c_ow_ops];
#pragma HLS array_partition variable = s_add type = complete dim = 0

/* Output struct variable used to pack the output in och_ops_out * c_ow_ops.
 * Divided in och_ops_out / och_ops packets, each of dimension c_ow_ops * c_och_ops
 * which are written in parallel. */
  t_output s_output_vector[c_och_ops_out / c_iter_ops_out][c_ow_ops]
                          [c_iter_ops_out];
  t_output_1x1 s_output_vector_1x1[c_och_ops_out / c_iter_ops_out][c_ow_ops]
                                  [c_iter_ops_out];

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the first pipeline
  // round to the higher log2 c_simd
  const int c_pad_bits = 27 - c_in_bits - 1;
  const int c_int_pad_bits = c_simd_bits + c_in_ibits + c_w_ibits;
  const int c_pad_acc_bits = c_simd_bits + c_in_bits + c_w_bits;
  // mask on c_simd bits
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the second pipeline
  // mask on c_simd bits
  const int c_pad_bits_1x1 = 27 - c_in_bits_1x1 - 1;
  const int c_int_pad_bits_1x1 =
    c_simd_bits_1x1 + c_in_ibits_1x1 + c_w_ibits_1x1;
  const int c_pad_acc_bits_1x1 = c_simd_bits_1x1 + c_in_bits_1x1 + c_w_bits_1x1;
  ////////////////////////////////////////////////////////////////////////////

  // TODO: the number of integer bits of the weights type must be considered
  typedef ap_fixed<c_pad_acc_bits, c_int_pad_bits, AP_RND_ZERO, AP_WRAP>
    t_acc_simd;
  typedef ap_fixed<c_pad_acc_bits_1x1, c_int_pad_bits_1x1, AP_RND_ZERO, AP_WRAP>
    t_acc_simd_1x1;
  auto ich_idx_add = 0;
  auto ich_idx_packets = 0;

  /* Stores the iterator of ops in ops_out */
  auto och_packet_in_ops_out = 0;

  /* Stores the iterator of weights packet in memory */
  auto s_weight_index = 0;
  auto s_bias_index = 0;

  // Iterating over the portion of tensor of each ow_ops_out slice
  CONV_LOOP:
  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {

    // Iterating over each ow_ops_out slice
    OW_OPS_OUT_LOOP:
    for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out;
         s_ow_ops_out += c_reuse, s_weight_index = 0, s_bias_index = 0) {

      // Iterating over the tensor input channels with steps of input packets
      ICH_LOOP:
      for (auto s_num_ich = 0; s_num_ich < c_ich;
           s_num_ich += c_iter_ich, acc_group = 0) {
#pragma HLS pipeline II = 1

        // Iterating over the tensor output channels with steps of output
        // packets
        OCH_LOOP:
        for (auto s_num_och = 0, ich_idx_packets = 0; s_num_och < c_och_depth;
             s_num_och += c_iter_och) {

          // Iterating over single output packet with steps of ops
          OCH_PACKET_LOOP:
            for (auto s_num_ops_out = 0, och_packet_in_ops_out = 0;
                 s_num_ops_out < c_och_ops_out;
                 s_num_ops_out += c_iter_ops_out,
                      ich_idx_packets++,
                      acc_group++,
                      och_packet_in_ops_out++) {

              for (auto s_iter = 0; s_iter < c_iter; s_iter++) {

                auto s_reuse = s_iter;

                // Reading ich_ops windows of input data each och/c_och_ops
                // cycles
                if (((s_num_och == 0) && (s_num_ops_out == 0)) ||
                    (c_depth == 1)) {
                  t_input_struct s_input_struct = i_input[0].read();
#pragma HLS array_partition variable = s_input_struct.data type = complete
                  s_input = s_input_struct.data;
                  /* Sending last only at the bottom right data */
                  s_last = s_input_struct.last;
#ifndef __SYNTHESIS__
#ifdef DEBUG_INPUT
                  constexpr int c_pixels = (c_fh * FW);
                  ap_uint<8 * c_ich_ops * c_pixels> tmp;
                  for (auto s_pixels = 0; s_pixels < c_pixels; s_pixels++) {
                    for (auto s_in_ops = 0; s_in_ops < c_ich_ops; s_in_ops++) {
                      tmp.range(8 * (s_pixels * c_ich_ops + s_in_ops + 1) - 1,
                                8 * (s_pixels * c_ich_ops + s_in_ops)) =
                        s_input[s_pixels][s_in_ops].range(7, 0);
                    }
                  }
                  std::cout << "inp " << tmp.to_string(16) << std::endl;
#endif /* DEBUG_INPUT */
#endif /* __SYNTHESIS__ */
                }

                /* Buffering to speed up computations */
                /* TODO: Adjust for generic bit quantizations */
                if (s_reuse == 0) {
                  for (auto s_index = 0; s_index < c_index; s_index++) {
                    for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops;
                         s_ich_ops++) {
                      for (auto s_ops = 0; s_ops < c_och_ops; s_ops++) {
                        auto s_ops_index = s_ich_ops * c_och_ops + s_ops;
                        s_weight[s_index][s_ich_ops][s_ops] =
                          mem_weights[s_index][s_weight_index][s_ops_index];
                      }
                    }
                  }

                  // If it is the first reuse iteration, read the 1x1 weights
                  if constexpr (std::is_same<t_weight_1x1,
                                             std::nullptr_t>::value == false) {
                    for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops;
                         s_ich_ops++) {
                      for (auto s_ops = 0; s_ops < c_och_ops; s_ops++) {
                        auto s_ops_index = s_ich_ops * c_och_ops + s_ops;
                        s_weight_1x1[0][s_ich_ops][s_ops] =
                          mem_weights_1x1[0][s_weight_index][s_ops_index];
                      }
                    }
                  }

                  if (s_num_ich == 0 || c_depth == 1) {
                    if constexpr (std::is_same<t_bias, std::nullptr_t>::value ==
                                  false) {
                      for (auto s_ops = 0; s_ops < c_bias_ops; s_ops++) {
                        s_bias[0][s_ops] = mem_bias[s_bias_index][s_ops];
                      }
                      if constexpr (std::is_same<t_bias_1x1,
                                                 std::nullptr_t>::value ==
                                    false) {
                        for (auto s_ops = 0; s_ops < c_bias_ops; s_ops++) {
                          s_bias_1x1[0][s_ops] =
                            mem_bias_1x1[s_bias_index][s_ops];
                        }
                      }
                    }
                    s_bias_index++;
                  }

                  s_weight_index++;
                }

                // Done to avoid partitioning of the stream and resource wasting
                // Theoretically with the add the input and output dimensions
                // should be equal, so we could perform the addition in
                // different iterations
                if constexpr (std::is_same<t_add_struct,
                                           std::nullptr_t>::value == false) {
                  // auto s_add_read = (s_num_och + s_num_ops_out) % c_add_ops;
                  if (ich_idx_packets == (c_add_ops / c_och_ops)) {
                    ich_idx_packets = 0;
                  }
                  if ((ich_idx_packets == 0) &&
                      (s_num_ich == c_ich - c_ich_ops)) {
                    ich_idx_add = 0;
                    for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                      // FIX FOR MOBILENETv2: Taking into account different
                      // output and input channels

                      t_add_struct s_add_struct = i_add[s_ow_ops].read();
                      s_add[s_ow_ops] = s_add_struct.data[0];
#ifndef __SYNTHESIS__
#ifdef DEBUG_ADD
                      ap_uint<t_add_st::width * c_add_ops> tmp = 0;
                      for (auto s_add_ops = 0; s_add_ops < c_add_ops;
                           s_add_ops++) {
                        tmp.range((t_add_st::width * (s_add_ops + 1)) - 1,
                                  t_add_st::width * s_add_ops) =
                          s_add[s_ow_ops][s_add_ops].range(t_add_st::width - 1,
                                                           0);
                      }
                      std::cout << "add " << tmp.to_string(16) << " "
                                << s_ow_ops << std::endl;
#endif
#endif
#ifndef __SYNTHESIS__
#ifdef DEBUG_ADD
                      for (auto s_ich_idx_add = 0; s_ich_idx_add < c_add_ops;
                           s_ich_idx_add++) {
                        std::cout << "add[" << s_ow_ops << "][" << s_ich_idx_add
                                  << "] "
                                  << s_add[s_ow_ops].data[0][s_ich_idx_add]
                                  << std::endl;
                      }
#endif
#endif
                  }
                }
              }

              std::array<t_input_data, c_ow_ops> s_input_1x1;
              if constexpr (std::is_same<t_acc_1x1_struct,
                                         std::nullptr_t>::value == false) {
#pragma HLS array_partition variable = s_input_1x1 type = complete
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  auto forward_index =
                    (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                  // s_input_1x1[s_ow_ops] = s_input[MO + MO%c_str -
                  // s_ow_ops*c_str];
                  s_input_1x1[c_ow_ops - s_ow_ops - 1] = s_input[forward_index];
                }
              }

              OCH_OPS_LOOP:
              for (auto s_ops = 0; s_ops < c_och_ops; s_ops += c_och_pack, ich_idx_add+=c_och_pack) {
                  auto s_och = s_num_och + s_num_ops_out + s_ops;
                OW_OPS_LOOP:
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops+=c_ow_pack) {
                  conv_pipe<
                    t_input,
                    t_weight,
                    t_weight_st,
                    t_bias,
                    t_add_vector,
                    t_add_st,
                    t_input_mod,
                    t_acc_struct,
                    t_acc,
                    t_acc_simd,
                    t_output,
                    t_output_clip,
                    t_output_mask,
                    c_reuse_iter,
                    c_fh,
                    c_fw,
                    c_index,
                    c_str,
                    c_och_ops,
                    c_iter_ops_out,
                    c_ich_ops,
                    c_add_ops,
                    c_ow_ops,
                    c_ow_pack,
                    c_och_pack,
                    c_relu,
                    c_leakyrelu,
                    c_ich,
                    c_och_depth,
                    c_in_bits,
                    c_simd_bits,
                    c_simd,
                    c_pad_bits,
                    c_int_pad_bits,
                    c_pad_acc_bits,
                    c_mask,
                    c_w_bits,
                    c_depth
                  > (
                    s_input,
                    s_weight,
                    s_bias,
                    s_ops,
                    s_num_ich,
                    s_ow_ops,
                    ich_idx_add,
                    s_add,
                    s_acc_buff[s_reuse][acc_group],
                    s_output_vector[och_packet_in_ops_out]
                  );

                  // TODO: split the loop in two parts controlled by different ops options
                  if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
                    if constexpr(c_ops_1x1 != c_och_ops){
                      if ((s_och > 0) | (s_ops > c_ops_1x1))  continue;
                    }
                    conv_pipe<
                      std::array<t_input_data, c_ow_ops>,
                      t_weight_1x1,
                      t_weight_1x1_st,
                      t_bias_1x1,
                      std::nullptr_t,
                      std::nullptr_t,
                      t_input_1x1,
                      t_acc_1x1_struct,
                      t_acc_1x1,
                      t_acc_simd_1x1,
                      t_output_1x1,
                      std::nullptr_t,
                      std::nullptr_t,
                      c_reuse_iter,
                      1,
                      1,
                      1,
                      1,
                      c_och_ops,
                      c_iter_ops_out,
                      c_ich_ops,
                      c_add_ops,
                      c_ow_ops,
                      c_ow_pack,
                      c_och_pack,
                      0,
                      0,
                      c_ich,
                      c_och_depth,
                      c_in_bits_1x1,
                      c_simd_bits_1x1,
                      c_simd_1x1,
                      c_pad_bits_1x1,
                      c_int_pad_bits_1x1,
                      c_pad_acc_bits_1x1,
                      c_mask_1x1,
                      c_w_bits_1x1,
                      c_depth
                    > (
                      s_input_1x1,
                      s_weight_1x1,
                      s_bias_1x1,
                      s_ops,
                      s_num_ich,
                      s_ow_ops,
                      ich_idx_add,
                      nullptr,
                      s_acc_buff_1x1[s_reuse][acc_group],
                      s_output_vector_1x1[och_packet_in_ops_out]
                    );
                  }
                }
              }

              if constexpr (std::is_same<t_forward_struct,
                                         std::nullptr_t>::value == false) {
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  if ((s_num_ops_out == (c_och_ops_out - c_iter_ops_out)) &&
                      (s_num_och == (c_och_depth - c_iter_och))) {
                    t_forward_struct s_forward;
                    // auto forward_index = MO + MO%c_str - s_ow_ops*c_str;

                    /* Compute the center of each window in input */
                    auto forward_index =
                      (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                    s_forward.data[0] = s_input[forward_index];
                    o_forward[s_ow_ops_out + s_ow_ops].write(s_forward);
#ifndef __SYNTHESIS__
#ifdef DEBUG_FORWARD
                    for (auto s_log_ich = 0; s_log_ich < c_ich_ops;
                         s_log_ich++) {
                      std::cout << "forward[" << s_ow_ops_out + s_ow_ops << "]["
                                << s_log_ich << "] "
                                << s_forward.data[0][s_log_ich] << std::endl;
                    }
#endif /* DEBUG_FORWARD */
#endif /* __SYNTHESIS__ */
                  }
                }
              }

              // Writing in output only when all the ich channels have been
              // considered and the och_ops_out packets are ready
              if (((s_num_ich == (c_ich - c_ich_ops)) || (c_depth == 1)) &&
                  (s_num_ops_out == (c_och_ops_out - c_iter_ops_out))) {

                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  t_output_struct temp_struct;
                  for (auto s_och_ops_out = 0;
                       s_och_ops_out < c_och_ops_out / c_iter_ops_out;
                       s_och_ops_out++) {
                    for (auto s_och_ops = 0; s_och_ops < c_iter_ops_out;
                         s_och_ops++) {
                      temp_struct
                        .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                        s_output_vector[s_och_ops_out][s_ow_ops][s_och_ops];
                    }
                  }
                  temp_struct.last = s_last;
                  o_output[s_ow_ops_out + s_ow_ops].write(temp_struct);

// #ifndef __SYNTHESIS__
//                   ap_uint<8 * c_och_ops_out> temp = 0;
//                   for (auto s_tmp = c_och_ops_out - 1; s_tmp >= 0; s_tmp--) {
//                     temp <<= 8;
//                     temp.range(7, 0) = temp_struct.data[0][s_tmp].range(7, 0);
//                   }
//                   std::cout << temp.to_string(16) << " " << s_ow_ops + s_ow_ops_out << std::endl;
// #endif
                  if constexpr (std::is_same<t_output_struct_1x1,
                                             std::nullptr_t>::value == false) {
                    t_output_struct_1x1 temp_struct_1x1;
                    for (auto s_och_ops_out = 0;
                         s_och_ops_out < c_och_ops_out / c_iter_ops_out;
                         s_och_ops_out++) {
                      for (auto s_och_ops = 0; s_och_ops < c_iter_ops_out;
                           s_och_ops++) {
                        temp_struct_1x1
                          .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                          s_output_vector_1x1[s_och_ops_out][s_ow_ops]
                                             [s_och_ops];
                      }
                    }
                    if (s_iter < c_iter_1x1) {
                      o_output_1x1[s_ow_ops_out + s_ow_ops].write(
                        temp_struct_1x1);
// #ifndef __SYNTHESIS__
//                       ap_uint<8 * c_och_ops_out> temp = 0;
//                       for (auto s_tmp = c_och_ops_out - 1; s_tmp >= 0; s_tmp--) {
//                         temp <<= 8;
//                         temp.range(7, 0) = temp_struct_1x1.data[0][s_tmp].range(7, 0);
//                       }
//                       std::cout << temp.to_string(16) << " " << s_ow_ops + s_ow_ops_out << " 1x1" << std::endl;
// #endif
                    }
                  }
                }
              }
            }
          }
        }
      }

#ifndef __SYNTHESIS__
        #ifdef DEBUG_RES
          for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
            for (auto s_och = 0; s_och < c_och_depth; s_och++) {
              std::cout <<  "PRE RES " << s_acc_buff[0][s_och*c_ow_ops+s_ow_ops] << std::endl;
              auto s_acc_log = quant_stream<
                t_output, t_output_clip, t_output_mask, t_acc, c_relu, c_leakyrelu
              >(s_acc_buff[0][s_och*c_ow_ops+s_ow_ops]);
              std::cout <<  "RES " << s_acc_log << std::endl;
            }
            if constexpr(std::is_same<t_output_struct_1x1, std::nullptr_t>::value == false) {
              for (auto s_och = 0; s_och < c_och_depth; s_och++) {
                auto s_acc_log = quant_stream<
                  t_output_1x1, std::nullptr_t, std::nullptr_t, t_acc_1x1, 0, 0
                >(s_acc_1x1_buff[0][s_och*c_ow_ops+s_ow_ops]);
                std::cout <<  "RES 1x1 " << s_acc_log << std::endl;
              }
            }
          }
        #endif /* DEBUG_RES */
      #endif /* __SYNTHESIS__ */
    }
  }

#ifndef __SYNTHESIS__
  // Check if input stream is empty
#ifndef SKIP_ASSERTION
  for (auto s_ow_ops = 0; s_ow_ops < 1; s_ow_ops++) {
    if (i_input[s_ow_ops].size() != 0) {
      std::cout << "ERROR: The input stream is not empty" << std::endl;
      std::cout << "i_input[" << s_ow_ops
                << "].size() = " << i_input[s_ow_ops].size() << std::endl;
      assert(false);
    }
  }
  // Check if input add is empty
  for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
    if constexpr (std::is_same<t_add_struct, std::nullptr_t>::value == false) {
      if (i_add[s_ow_ops].size() != 0) {
        std::cout << "ERROR: The input add stream is not empty" << std::endl;
        std::cout << "i_add[" << s_ow_ops
                  << "].size() = " << i_add[s_ow_ops].size() << std::endl;
        assert(false);
      }
    }
  }
  for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out; s_ow_ops_out++) {
    std::cout << "o_output[" << s_ow_ops_out
              << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
    if (o_output[s_ow_ops_out].size() == 0) {
      std::cout << "ERROR: The output stream is empty" << std::endl;
      std::cout << "o_output[" << s_ow_ops_out
                << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
      assert(false);
    }
  }
#endif /* SKIP_ASSERTION */
  if (c_depth == 1)
    std::cout << "end depth_conv_op " << c_ich << " " << c_depth << std::endl;
  else
    std::cout << "end conv_op " << c_ich << " " << c_depth << std::endl;
#endif /* __SYNTHESIS__ */
  }


template<typename t_input_struct, // input activation struct type
         typename t_input,        // input activation matrix type
         typename t_input_data,   // input activation vector type without window
         typename t_input_st,     // input activation standard type
         typename t_weight,       // weight matrix type
         typename t_weight_st,    // weight standard type
         typename t_bias,         // bias vector type
         typename t_bias_st,      // bias standard type
         typename t_add_struct,   // add activation struct type
         typename t_add_vector,   // add activation vector type
         typename t_add_st,       // add activation standard type
         typename t_forward_struct, // forward activation struct type
         typename t_input_mod,      // input activation modified type (input
         // quantization)
         typename t_input_1x1,     // input activation 1x1 type
         typename t_weight_1x1,    // weight 1x1 matrix type
         typename t_weight_1x1_st, // weight 1x1 standard type
         typename t_bias_1x1,      // bias 1x1 vector type
         typename t_bias_1x1_st,   // bias 1x1 standard type
         typename t_acc_struct,
         typename t_acc,
         typename t_acc_1x1_struct,
         typename t_acc_1x1,
         typename t_output_struct,
         typename t_output_vector,
         typename t_output,
         typename t_output_clip,
         typename t_output_mask,
         typename t_output_struct_1x1,
         typename t_output_1x1,
         typename p_stream_t,
         const int c_ich_groups,       // input channels divided by groups
         const int c_ich,
         const int c_och,
         const int c_och_1x1,
         const int c_oh,
         const int c_ow,
         const int c_fh,
         const int c_fw,
         const int c_index,
         const int c_str,
         const int c_och_ops,
         const int c_och_ops_out,
         const int c_ich_ops,
         const int c_add_ops,
         const int c_ow_ops,
         const int c_ow_ops_out,
         const int c_bias_ops,
         const int c_reuse,
         const int c_ow_pack,
         const int c_och_pack,
         const int c_in_bits,
         const int c_in_ibits,
         const int c_w_bits,
         const int c_w_ibits,
         const int c_simd_bits,
         const int c_simd,
         const int c_mask,
         const int c_in_bits_1x1,
         const int c_in_ibits_1x1,
         const int c_w_bits_1x1,
         const int c_w_ibits_1x1,
         const int c_simd_bits_1x1,
         const int c_simd_1x1,
         const int c_mask_1x1,
         const int c_relu,
         const int c_leakyrelu,
         const int c_depth>
void
conv_comp_onchip_OCH(
  t_bias_st mem_bias[c_och / c_bias_ops][c_bias_ops],
  t_weight_st mem_weights[c_fh * c_fw][c_och * c_ich_groups / (c_och_ops * c_ich_ops)]
                         [c_och_ops * c_ich_ops],
  t_bias_1x1_st mem_bias_1x1[c_och_1x1 / c_bias_ops][c_bias_ops],
  t_weight_1x1_st mem_weights_1x1[1][c_och_1x1 * c_ich_groups / (c_och_ops * c_ich_ops)]
                                 [c_och_ops * c_ich_ops],

  hls::stream<t_input_struct> i_input[1],
  hls::stream<t_add_struct> i_add[c_ow_ops],
  hls::stream<t_forward_struct> o_forward[c_ow_ops],
  hls::stream<t_output_struct> o_output[c_ow_ops_out],
  hls::stream<t_output_struct_1x1> o_output_1x1[c_ow_ops_out])
{
  // Generic Convolution Computation

  /* We want to use chain of DSPs and not balance the expressions, we are not
   * interested in short pipeline, more on resources */
//#pragma HLS expression_balance off

  const auto c_och_depth = (c_depth == 1) ? 1 : c_och;
  const auto c_o_index = c_oh * c_ow / c_ow_ops_out;
  const auto c_reuse_iter = c_reuse / c_ow_ops;
  const auto c_iter = c_reuse_iter;
  const auto c_och_1x1_depth = (c_depth == 1) ? 1 : c_och_1x1;
  const auto c_num_och_1x1 = c_och_1x1_depth / c_och_ops;
  const auto c_iter_1x1 = c_reuse_iter * c_num_och_1x1;
  const auto c_ops_1x1 = (c_och_1x1 < c_och_ops) ? c_och_1x1 : c_och_ops;
  const auto c_iter_ich = (c_depth == 1) ? c_och_ops_out : c_ich_ops;
  const auto c_iter_och = (c_depth == 1) ? 1 : c_och_ops_out;
  const auto c_iter_ops_out = (c_depth == 1) ? c_ich_ops : c_och_ops;
  constexpr int FW = (c_fw + (c_ow_ops - 1) * c_str);

  /* Groups of accumulator for a total of och * ow_ops. The one that must be
   * available in parallel are c_och_ops * c_ow_ops. This data structure is not used
   * by depth convolutions */
  t_acc s_acc_buff[c_reuse_iter][c_och_depth / c_och_ops][c_och_ops * c_ow_ops];
  t_acc_1x1 s_acc_buff_1x1[c_reuse_iter][c_och_depth / c_och_ops][c_och_ops * c_ow_ops];
  auto acc_group = 0;
#pragma HLS array_partition variable = s_acc_buff type = complete dim = 3
#pragma HLS array_partition variable = s_acc_buff_1x1 type = complete dim = 3

  t_input s_input;
#pragma HLS aggregate variable = s_input
#pragma HLS array_partition variable = s_input type = complete
  
  bool s_last = false;
  t_weight s_weight[c_index];
#pragma HLS array_partition variable = s_weight type = complete
  
  t_weight_1x1 s_weight_1x1[1];
#pragma HLS array_partition variable = s_weight_1x1 type = complete
  
  t_bias s_bias;
#pragma HLS array_partition variable = s_bias type = complete dim = 1
  
  t_bias_1x1 s_bias_1x1;
#pragma HLS array_partition variable = s_bias_1x1 type = complete dim = 1
  
  t_add_vector s_add[c_ow_ops];
#pragma HLS array_partition variable = s_add type = complete dim = 0

/* Output struct variable used to pack the output in och_ops_out * c_ow_ops.
 * Divided in och_ops_out / och_ops packets, each of dimension c_ow_ops * c_och_ops
 * which are written in parallel. */
  t_output s_output_vector[c_och_ops_out / c_iter_ops_out][c_ow_ops]
                          [c_iter_ops_out];
  t_output_1x1 s_output_vector_1x1[c_och_ops_out / c_iter_ops_out][c_ow_ops]
                                  [c_iter_ops_out];

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the first pipeline
  // round to the higher log2 c_simd
  const int c_pad_bits = 27 - c_in_bits - 1;
  const int c_int_pad_bits = c_simd_bits + c_in_ibits + c_w_ibits;
  const int c_pad_acc_bits = c_simd_bits + c_in_bits + c_w_bits;
  // mask on c_simd bits
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the second pipeline
  // mask on c_simd bits
  const int c_pad_bits_1x1 = 27 - c_in_bits_1x1 - 1;
  const int c_int_pad_bits_1x1 =
    c_simd_bits_1x1 + c_in_ibits_1x1 + c_w_ibits_1x1;
  const int c_pad_acc_bits_1x1 = c_simd_bits_1x1 + c_in_bits_1x1 + c_w_bits_1x1;
  ////////////////////////////////////////////////////////////////////////////

  // TODO: the number of integer bits of the weights type must be considered
  typedef ap_fixed<c_pad_acc_bits, c_int_pad_bits, AP_RND_ZERO, AP_WRAP>
    t_acc_simd;
  typedef ap_fixed<c_pad_acc_bits_1x1, c_int_pad_bits_1x1, AP_RND_ZERO, AP_WRAP>
    t_acc_simd_1x1;
  auto ich_idx_add = 0;
  auto ich_idx_packets = 0;

  /* Stores the iterator of ops in ops_out */
  auto och_packet_in_ops_out = 0;

  /* Stores the iterator of weights packet in memory */
  auto s_weight_index = 0;
  auto s_bias_index = 0;

  // Iterating over the portion of tensor of each ow_ops_out slice
  CONV_LOOP:
  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {

    // Iterating over each ow_ops_out slice
    OW_OPS_OUT_LOOP:
    for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out;
         s_ow_ops_out += c_reuse, s_weight_index = 0, s_bias_index = 0) {

      // Iterating over the tensor input channels with steps of input packets
      ICH_LOOP:
      for (auto s_num_ich = 0; s_num_ich < c_ich;
           s_num_ich += c_iter_ich, acc_group = 0) {

        // Iterating over the tensor output channels with steps of output
        // packets
        OCH_LOOP:
        for (auto s_num_och = 0, ich_idx_packets = 0; s_num_och < c_och_depth;
             s_num_och += c_iter_och) {
#pragma HLS pipeline II = 1

          // Iterating over single output packet with steps of ops
          OCH_PACKET_LOOP:
            for (auto s_num_ops_out = 0, och_packet_in_ops_out = 0;
                 s_num_ops_out < c_och_ops_out;
                 s_num_ops_out += c_iter_ops_out,
                      ich_idx_packets++,
                      acc_group++,
                      och_packet_in_ops_out++) {

              for (auto s_iter = 0; s_iter < c_iter; s_iter++) {

                auto s_reuse = s_iter;

                // Reading ich_ops windows of input data each och/c_och_ops
                // cycles
                if (((s_num_och == 0) && (s_num_ops_out == 0)) ||
                    (c_depth == 1)) {
                  t_input_struct s_input_struct = i_input[0].read();
#pragma HLS array_partition variable = s_input_struct.data type = complete
                  s_input = s_input_struct.data;
                  /* Sending last only at the bottom right data */
                  s_last = s_input_struct.last;
#ifndef __SYNTHESIS__
#ifdef DEBUG_INPUT
                  constexpr int c_pixels = (c_fh * FW);
                  ap_uint<8 * c_ich_ops * c_pixels> tmp;
                  for (auto s_pixels = 0; s_pixels < c_pixels; s_pixels++) {
                    for (auto s_in_ops = 0; s_in_ops < c_ich_ops; s_in_ops++) {
                      tmp.range(8 * (s_pixels * c_ich_ops + s_in_ops + 1) - 1,
                                8 * (s_pixels * c_ich_ops + s_in_ops)) =
                        s_input[s_pixels][s_in_ops].range(7, 0);
                    }
                  }
                  std::cout << "inp " << tmp.to_string(16) << std::endl;
#endif /* DEBUG_INPUT */
#endif /* __SYNTHESIS__ */
                }

                /* Buffering to speed up computations */
                /* TODO: Adjust for generic bit quantizations */
                if (s_reuse == 0) {
                  for (auto s_index = 0; s_index < c_index; s_index++) {
                    for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops;
                         s_ich_ops++) {
                      for (auto s_ops = 0; s_ops < c_och_ops; s_ops++) {
                        auto s_ops_index = s_ich_ops * c_och_ops + s_ops;
                        s_weight[s_index][s_ich_ops][s_ops] =
                          mem_weights[s_index][s_weight_index][s_ops_index];
                      }
                    }
                  }

                  // If it is the first reuse iteration, read the 1x1 weights
                  if constexpr (std::is_same<t_weight_1x1,
                                             std::nullptr_t>::value == false) {
                    for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops;
                         s_ich_ops++) {
                      for (auto s_ops = 0; s_ops < c_och_ops; s_ops++) {
                        auto s_ops_index = s_ich_ops * c_och_ops + s_ops;
                        s_weight_1x1[0][s_ich_ops][s_ops] =
                          mem_weights_1x1[0][s_weight_index][s_ops_index];
                      }
                    }
                  }

                  if (s_num_ich == 0 || c_depth == 1) {
                    if constexpr (std::is_same<t_bias, std::nullptr_t>::value ==
                                  false) {
                      for (auto s_ops = 0; s_ops < c_bias_ops; s_ops++) {
                        s_bias[0][s_ops] = mem_bias[s_bias_index][s_ops];
                      }
                      if constexpr (std::is_same<t_bias_1x1,
                                                 std::nullptr_t>::value ==
                                    false) {
                        for (auto s_ops = 0; s_ops < c_bias_ops; s_ops++) {
                          s_bias_1x1[0][s_ops] =
                            mem_bias_1x1[s_bias_index][s_ops];
                        }
                      }
                    }
                    s_bias_index++;
                  }

                  s_weight_index++;
                }

                // Done to avoid partitioning of the stream and resource wasting
                // Theoretically with the add the input and output dimensions
                // should be equal, so we could perform the addition in
                // different iterations
                if constexpr (std::is_same<t_add_struct,
                                           std::nullptr_t>::value == false) {
                  // auto s_add_read = (s_num_och + s_num_ops_out) % c_add_ops;
                  if (ich_idx_packets == (c_add_ops / c_och_ops)) {
                    ich_idx_packets = 0;
                  }
                  if ((ich_idx_packets == 0) &&
                      (s_num_ich == c_ich - c_ich_ops)) {
                    ich_idx_add = 0;
                    for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                      // FIX FOR MOBILENETv2: Taking into account different
                      // output and input channels

                      t_add_struct s_add_struct = i_add[s_ow_ops].read();
                      s_add[s_ow_ops] = s_add_struct.data[0];
#ifndef __SYNTHESIS__
#ifdef DEBUG_ADD
                      ap_uint<t_add_st::width * c_add_ops> tmp = 0;
                      for (auto s_add_ops = 0; s_add_ops < c_add_ops;
                           s_add_ops++) {
                        tmp.range((t_add_st::width * (s_add_ops + 1)) - 1,
                                  t_add_st::width * s_add_ops) =
                          s_add[s_ow_ops][s_add_ops].range(t_add_st::width - 1,
                                                           0);
                      }
                      std::cout << "add " << tmp.to_string(16) << " "
                                << s_ow_ops << std::endl;
#endif
#endif
#ifndef __SYNTHESIS__
#ifdef DEBUG_ADD
                      for (auto s_ich_idx_add = 0; s_ich_idx_add < c_add_ops;
                           s_ich_idx_add++) {
                        std::cout << "add[" << s_ow_ops << "][" << s_ich_idx_add
                                  << "] "
                                  << s_add[s_ow_ops].data[0][s_ich_idx_add]
                                  << std::endl;
                      }
#endif
#endif
                  }
                }
              }

              std::array<t_input_data, c_ow_ops> s_input_1x1;
              if constexpr (std::is_same<t_acc_1x1_struct,
                                         std::nullptr_t>::value == false) {
#pragma HLS array_partition variable = s_input_1x1 type = complete
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  auto forward_index =
                    (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                  // s_input_1x1[s_ow_ops] = s_input[MO + MO%c_str -
                  // s_ow_ops*c_str];
                  s_input_1x1[c_ow_ops - s_ow_ops - 1] = s_input[forward_index];
                }
              }

              OCH_OPS_LOOP:
              for (auto s_ops = 0; s_ops < c_och_ops; s_ops += c_och_pack, ich_idx_add+=c_och_pack) {
                  auto s_och = s_num_och + s_num_ops_out + s_ops;
                OW_OPS_LOOP:
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops+=c_ow_pack) {
                  conv_pipe<
                    t_input,
                    t_weight,
                    t_weight_st,
                    t_bias,
                    t_add_vector,
                    t_add_st,
                    t_input_mod,
                    t_acc_struct,
                    t_acc,
                    t_acc_simd,
                    t_output,
                    t_output_clip,
                    t_output_mask,
                    c_reuse_iter,
                    c_fh,
                    c_fw,
                    c_index,
                    c_str,
                    c_och_ops,
                    c_iter_ops_out,
                    c_ich_ops,
                    c_add_ops,
                    c_ow_ops,
                    c_ow_pack,
                    c_och_pack,
                    c_relu,
                    c_leakyrelu,
                    c_ich,
                    c_och_depth,
                    c_in_bits,
                    c_simd_bits,
                    c_simd,
                    c_pad_bits,
                    c_int_pad_bits,
                    c_pad_acc_bits,
                    c_mask,
                    c_w_bits,
                    c_depth
                  > (
                    s_input,
                    s_weight,
                    s_bias,
                    s_ops,
                    s_num_ich,
                    s_ow_ops,
                    ich_idx_add,
                    s_add,
                    s_acc_buff[s_reuse][acc_group],
                    s_output_vector[och_packet_in_ops_out]
                  );

                  // TODO: split the loop in two parts controlled by different ops options
                  if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
                    if constexpr(c_ops_1x1 != c_och_ops){
                      if ((s_och > 0) | (s_ops > c_ops_1x1))  continue;
                    }
                    conv_pipe<
                      std::array<t_input_data, c_ow_ops>,
                      t_weight_1x1,
                      t_weight_1x1_st,
                      t_bias_1x1,
                      std::nullptr_t,
                      std::nullptr_t,
                      t_input_1x1,
                      t_acc_1x1_struct,
                      t_acc_1x1,
                      t_acc_simd_1x1,
                      t_output_1x1,
                      std::nullptr_t,
                      std::nullptr_t,
                      c_reuse_iter,
                      1,
                      1,
                      1,
                      1,
                      c_och_ops,
                      c_iter_ops_out,
                      c_ich_ops,
                      c_add_ops,
                      c_ow_ops,
                      c_ow_pack,
                      c_och_pack,
                      0,
                      0,
                      c_ich,
                      c_och_depth,
                      c_in_bits_1x1,
                      c_simd_bits_1x1,
                      c_simd_1x1,
                      c_pad_bits_1x1,
                      c_int_pad_bits_1x1,
                      c_pad_acc_bits_1x1,
                      c_mask_1x1,
                      c_w_bits_1x1,
                      c_depth
                    > (
                      s_input_1x1,
                      s_weight_1x1,
                      s_bias_1x1,
                      s_ops,
                      s_num_ich,
                      s_ow_ops,
                      ich_idx_add,
                      nullptr,
                      s_acc_buff_1x1[s_reuse][acc_group],
                      s_output_vector_1x1[och_packet_in_ops_out]
                    );
                  }
                }
              }

              if constexpr (std::is_same<t_forward_struct,
                                         std::nullptr_t>::value == false) {
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  if ((s_num_ops_out == (c_och_ops_out - c_iter_ops_out)) &&
                      (s_num_och == (c_och_depth - c_iter_och))) {
                    t_forward_struct s_forward;
                    // auto forward_index = MO + MO%c_str - s_ow_ops*c_str;

                    /* Compute the center of each window in input */
                    auto forward_index =
                      (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                    s_forward.data[0] = s_input[forward_index];
                    o_forward[s_ow_ops_out + s_ow_ops].write(s_forward);
#ifndef __SYNTHESIS__
#ifdef DEBUG_FORWARD
                    for (auto s_log_ich = 0; s_log_ich < c_ich_ops;
                         s_log_ich++) {
                      std::cout << "forward[" << s_ow_ops_out + s_ow_ops << "]["
                                << s_log_ich << "] "
                                << s_forward.data[0][s_log_ich] << std::endl;
                    }
#endif /* DEBUG_FORWARD */
#endif /* __SYNTHESIS__ */
                  }
                }
              }

              // Writing in output only when all the ich channels have been
              // considered and the och_ops_out packets are ready
              if (((s_num_ich == (c_ich - c_ich_ops)) || (c_depth == 1)) &&
                  (s_num_ops_out == (c_och_ops_out - c_iter_ops_out))) {

                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  t_output_struct temp_struct;
                  for (auto s_och_ops_out = 0;
                       s_och_ops_out < c_och_ops_out / c_iter_ops_out;
                       s_och_ops_out++) {
                    for (auto s_och_ops = 0; s_och_ops < c_iter_ops_out;
                         s_och_ops++) {
                      temp_struct
                        .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                        s_output_vector[s_och_ops_out][s_ow_ops][s_och_ops];
                    }
                  }
                  temp_struct.last = s_last;
                  o_output[s_ow_ops_out + s_ow_ops].write(temp_struct);

// #ifndef __SYNTHESIS__
//                   ap_uint<8 * c_och_ops_out> temp = 0;
//                   for (auto s_tmp = c_och_ops_out - 1; s_tmp >= 0; s_tmp--) {
//                     temp <<= 8;
//                     temp.range(7, 0) = temp_struct.data[0][s_tmp].range(7, 0);
//                   }
//                   std::cout << temp.to_string(16) << " " << s_ow_ops + s_ow_ops_out << std::endl;
// #endif
                  if constexpr (std::is_same<t_output_struct_1x1,
                                             std::nullptr_t>::value == false) {
                    t_output_struct_1x1 temp_struct_1x1;
                    for (auto s_och_ops_out = 0;
                         s_och_ops_out < c_och_ops_out / c_iter_ops_out;
                         s_och_ops_out++) {
                      for (auto s_och_ops = 0; s_och_ops < c_iter_ops_out;
                           s_och_ops++) {
                        temp_struct_1x1
                          .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                          s_output_vector_1x1[s_och_ops_out][s_ow_ops]
                                             [s_och_ops];
                      }
                    }
                    if (s_iter < c_iter_1x1) {
                      o_output_1x1[s_ow_ops_out + s_ow_ops].write(
                        temp_struct_1x1);
// #ifndef __SYNTHESIS__
//                       ap_uint<8 * c_och_ops_out> temp = 0;
//                       for (auto s_tmp = c_och_ops_out - 1; s_tmp >= 0; s_tmp--) {
//                         temp <<= 8;
//                         temp.range(7, 0) = temp_struct_1x1.data[0][s_tmp].range(7, 0);
//                       }
//                       std::cout << temp.to_string(16) << " " << s_ow_ops + s_ow_ops_out << " 1x1" << std::endl;
// #endif
                    }
                  }
                }
              }
            }
          }
        }
      }

#ifndef __SYNTHESIS__
        #ifdef DEBUG_RES
          for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
            for (auto s_och = 0; s_och < c_och_depth; s_och++) {
              std::cout <<  "PRE RES " << s_acc_buff[0][s_och*c_ow_ops+s_ow_ops] << std::endl;
              auto s_acc_log = quant_stream<
                t_output, t_output_clip, t_output_mask, t_acc, c_relu, c_leakyrelu
              >(s_acc_buff[0][s_och*c_ow_ops+s_ow_ops]);
              std::cout <<  "RES " << s_acc_log << std::endl;
            }
            if constexpr(std::is_same<t_output_struct_1x1, std::nullptr_t>::value == false) {
              for (auto s_och = 0; s_och < c_och_depth; s_och++) {
                auto s_acc_log = quant_stream<
                  t_output_1x1, std::nullptr_t, std::nullptr_t, t_acc_1x1, 0, 0
                >(s_acc_1x1_buff[0][s_och*c_ow_ops+s_ow_ops]);
                std::cout <<  "RES 1x1 " << s_acc_log << std::endl;
              }
            }
          }
        #endif /* DEBUG_RES */
      #endif /* __SYNTHESIS__ */
    }
  }

#ifndef __SYNTHESIS__
  // Check if input stream is empty
#ifndef SKIP_ASSERTION
  for (auto s_ow_ops = 0; s_ow_ops < 1; s_ow_ops++) {
    if (i_input[s_ow_ops].size() != 0) {
      std::cout << "ERROR: The input stream is not empty" << std::endl;
      std::cout << "i_input[" << s_ow_ops
                << "].size() = " << i_input[s_ow_ops].size() << std::endl;
      assert(false);
    }
  }
  // Check if input add is empty
  for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
    if constexpr (std::is_same<t_add_struct, std::nullptr_t>::value == false) {
      if (i_add[s_ow_ops].size() != 0) {
        std::cout << "ERROR: The input add stream is not empty" << std::endl;
        std::cout << "i_add[" << s_ow_ops
                  << "].size() = " << i_add[s_ow_ops].size() << std::endl;
        assert(false);
      }
    }
  }
  for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out; s_ow_ops_out++) {
    std::cout << "o_output[" << s_ow_ops_out
              << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
    if (o_output[s_ow_ops_out].size() == 0) {
      std::cout << "ERROR: The output stream is empty" << std::endl;
      std::cout << "o_output[" << s_ow_ops_out
                << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
      assert(false);
    }
  }
#endif /* SKIP_ASSERTION */
  if (c_depth == 1)
    std::cout << "end depth_conv_op " << c_ich << " " << c_depth << std::endl;
  else
    std::cout << "end conv_op " << c_ich << " " << c_depth << std::endl;
#endif /* __SYNTHESIS__ */
  }


template<typename t_input_struct, // input activation struct type
         typename t_input,        // input activation matrix type
         typename t_input_data,   // input activation vector type without window
         typename t_input_st,     // input activation standard type
         typename t_weight,       // weight matrix type
         typename t_weight_st,    // weight standard type
         typename t_bias,         // bias vector type
         typename t_bias_st,      // bias standard type
         typename t_add_struct,   // add activation struct type
         typename t_add_vector,   // add activation vector type
         typename t_add_st,       // add activation standard type
         typename t_forward_struct, // forward activation struct type
         typename t_input_mod,      // input activation modified type (input
         // quantization)
         typename t_input_1x1,     // input activation 1x1 type
         typename t_weight_1x1,    // weight 1x1 matrix type
         typename t_weight_1x1_st, // weight 1x1 standard type
         typename t_bias_1x1,      // bias 1x1 vector type
         typename t_bias_1x1_st,   // bias 1x1 standard type
         typename t_acc_struct,
         typename t_acc,
         typename t_acc_1x1_struct,
         typename t_acc_1x1,
         typename t_output_struct,
         typename t_output_vector,
         typename t_output,
         typename t_output_clip,
         typename t_output_mask,
         typename t_output_struct_1x1,
         typename t_output_1x1,
         typename p_stream_t,
         const int c_ich_groups,       // input channels divided by groups
         const int c_ich,
         const int c_och,
         const int c_och_1x1,
         const int c_oh,
         const int c_ow,
         const int c_fh,
         const int c_fw,
         const int c_index,
         const int c_str,
         const int c_och_ops,
         const int c_och_ops_out,
         const int c_ich_ops,
         const int c_add_ops,
         const int c_ow_ops,
         const int c_ow_ops_out,
         const int c_bias_ops,
         const int c_reuse,
         const int c_ow_pack,
         const int c_och_pack,
         const int c_in_bits,
         const int c_in_ibits,
         const int c_w_bits,
         const int c_w_ibits,
         const int c_simd_bits,
         const int c_simd,
         const int c_mask,
         const int c_in_bits_1x1,
         const int c_in_ibits_1x1,
         const int c_w_bits_1x1,
         const int c_w_ibits_1x1,
         const int c_simd_bits_1x1,
         const int c_simd_1x1,
         const int c_mask_1x1,
         const int c_relu,
         const int c_leakyrelu,
         const int c_depth>
void
conv_comp_onchip(
  t_bias_st mem_bias[c_och / c_bias_ops][c_bias_ops],
  t_weight_st mem_weights[c_fh * c_fw][c_och * c_ich_groups / (c_och_ops * c_ich_ops)]
                         [c_och_ops * c_ich_ops],
  t_bias_1x1_st mem_bias_1x1[c_och_1x1 / c_bias_ops][c_bias_ops],
  t_weight_1x1_st mem_weights_1x1[1][c_och_1x1 * c_ich_groups / (c_och_ops * c_ich_ops)]
                                 [c_och_ops * c_ich_ops],

  hls::stream<t_input_struct> i_input[1],
  hls::stream<t_add_struct> i_add[c_ow_ops],
  hls::stream<t_forward_struct> o_forward[c_ow_ops],
  hls::stream<t_output_struct> o_output[c_ow_ops_out],
  hls::stream<t_output_struct_1x1> o_output_1x1[c_ow_ops_out])
{
  // Generic Convolution Computation

  /* We want to use chain of DSPs and not balance the expressions, we are not
   * interested in short pipeline, more on resources */
//#pragma HLS expression_balance off

  const auto c_och_depth = (c_depth == 1) ? 1 : c_och;
  const auto c_o_index = c_oh * c_ow / c_ow_ops_out;
  const auto c_reuse_iter = c_reuse / c_ow_ops;
  const auto c_iter = c_reuse_iter;
  const auto c_och_1x1_depth = (c_depth == 1) ? 1 : c_och_1x1;
  const auto c_num_och_1x1 = c_och_1x1_depth / c_och_ops;
  const auto c_iter_1x1 = c_reuse_iter * c_num_och_1x1;
  const auto c_ops_1x1 = (c_och_1x1 < c_och_ops) ? c_och_1x1 : c_och_ops;
  const auto c_iter_ich = (c_depth == 1) ? c_och_ops_out : c_ich_ops;
  const auto c_iter_och = (c_depth == 1) ? 1 : c_och_ops_out;
  const auto c_iter_ops_out = (c_depth == 1) ? c_ich_ops : c_och_ops;
  constexpr int FW = (c_fw + (c_ow_ops - 1) * c_str);

  /* Groups of accumulator for a total of och * ow_ops. The one that must be
   * available in parallel are c_och_ops * c_ow_ops. This data structure is not used
   * by depth convolutions */
  t_acc s_acc_buff[c_reuse_iter][c_och_depth / c_och_ops][c_och_ops * c_ow_ops];
  t_acc_1x1 s_acc_buff_1x1[c_reuse_iter][c_och_depth / c_och_ops][c_och_ops * c_ow_ops];
  auto acc_group = 0;
#pragma HLS array_partition variable = s_acc_buff type = complete dim = 3
#pragma HLS array_partition variable = s_acc_buff_1x1 type = complete dim = 3

  t_input s_input;
#pragma HLS aggregate variable = s_input
#pragma HLS array_partition variable = s_input type = complete
  
  bool s_last = false;
  t_weight s_weight[c_index];
#pragma HLS array_partition variable = s_weight type = complete
  
  t_weight_1x1 s_weight_1x1[1];
#pragma HLS array_partition variable = s_weight_1x1 type = complete
  
  t_bias s_bias;
#pragma HLS array_partition variable = s_bias type = complete dim = 1
  
  t_bias_1x1 s_bias_1x1;
#pragma HLS array_partition variable = s_bias_1x1 type = complete dim = 1
  
  t_add_vector s_add[c_ow_ops];
#pragma HLS array_partition variable = s_add type = complete dim = 0

/* Output struct variable used to pack the output in och_ops_out * c_ow_ops.
 * Divided in och_ops_out / och_ops packets, each of dimension c_ow_ops * c_och_ops
 * which are written in parallel. */
  t_output s_output_vector[c_och_ops_out / c_iter_ops_out][c_ow_ops]
                          [c_iter_ops_out];
  t_output_1x1 s_output_vector_1x1[c_och_ops_out / c_iter_ops_out][c_ow_ops]
                                  [c_iter_ops_out];

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the first pipeline
  // round to the higher log2 c_simd
  const int c_pad_bits = 27 - c_in_bits - 1;
  const int c_int_pad_bits = c_simd_bits + c_in_ibits + c_w_ibits;
  const int c_pad_acc_bits = c_simd_bits + c_in_bits + c_w_bits;
  // mask on c_simd bits
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the second pipeline
  // mask on c_simd bits
  const int c_pad_bits_1x1 = 27 - c_in_bits_1x1 - 1;
  const int c_int_pad_bits_1x1 =
    c_simd_bits_1x1 + c_in_ibits_1x1 + c_w_ibits_1x1;
  const int c_pad_acc_bits_1x1 = c_simd_bits_1x1 + c_in_bits_1x1 + c_w_bits_1x1;
  ////////////////////////////////////////////////////////////////////////////

  // TODO: the number of integer bits of the weights type must be considered
  typedef ap_fixed<c_pad_acc_bits, c_int_pad_bits, AP_RND_ZERO, AP_WRAP>
    t_acc_simd;
  typedef ap_fixed<c_pad_acc_bits_1x1, c_int_pad_bits_1x1, AP_RND_ZERO, AP_WRAP>
    t_acc_simd_1x1;
  auto ich_idx_add = 0;
  auto ich_idx_packets = 0;

  /* Stores the iterator of ops in ops_out */
  auto och_packet_in_ops_out = 0;

  /* Stores the iterator of weights packet in memory */
  auto s_weight_index = 0;
  auto s_bias_index = 0;

  // Iterating over the portion of tensor of each ow_ops_out slice
  CONV_LOOP:
  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {

    // Iterating over each ow_ops_out slice
    OW_OPS_OUT_LOOP:
    for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out;
         s_ow_ops_out += c_reuse, s_weight_index = 0, s_bias_index = 0) {

      // Iterating over the tensor input channels with steps of input packets
      ICH_LOOP:
      for (auto s_num_ich = 0; s_num_ich < c_ich;
           s_num_ich += c_iter_ich, acc_group = 0) {

        // Iterating over the tensor output channels with steps of output
        // packets
        OCH_LOOP:
        for (auto s_num_och = 0, ich_idx_packets = 0; s_num_och < c_och_depth;
             s_num_och += c_iter_och) {

          // Iterating over single output packet with steps of ops
          OCH_PACKET_LOOP:
            for (auto s_num_ops_out = 0, och_packet_in_ops_out = 0;
                 s_num_ops_out < c_och_ops_out;
                 s_num_ops_out += c_iter_ops_out,
                      ich_idx_packets++,
                      acc_group++,
                      och_packet_in_ops_out++) {
#pragma HLS pipeline II = 1

              for (auto s_iter = 0; s_iter < c_iter; s_iter++) {

                auto s_reuse = s_iter;

                // Reading ich_ops windows of input data each och/c_och_ops
                // cycles
                if (((s_num_och == 0) && (s_num_ops_out == 0)) ||
                    (c_depth == 1)) {
                  t_input_struct s_input_struct = i_input[0].read();
#pragma HLS array_partition variable = s_input_struct.data type = complete
                  s_input = s_input_struct.data;
                  /* Sending last only at the bottom right data */
                  s_last = s_input_struct.last;
#ifndef __SYNTHESIS__
#ifdef DEBUG_INPUT
                  constexpr int c_pixels = (c_fh * FW);
                  ap_uint<8 * c_ich_ops * c_pixels> tmp;
                  for (auto s_pixels = 0; s_pixels < c_pixels; s_pixels++) {
                    for (auto s_in_ops = 0; s_in_ops < c_ich_ops; s_in_ops++) {
                      tmp.range(8 * (s_pixels * c_ich_ops + s_in_ops + 1) - 1,
                                8 * (s_pixels * c_ich_ops + s_in_ops)) =
                        s_input[s_pixels][s_in_ops].range(7, 0);
                    }
                  }
                  std::cout << "inp " << tmp.to_string(16) << std::endl;
#endif /* DEBUG_INPUT */
#endif /* __SYNTHESIS__ */
                }

                /* Buffering to speed up computations */
                /* TODO: Adjust for generic bit quantizations */
                if (s_reuse == 0) {
                  for (auto s_index = 0; s_index < c_index; s_index++) {
                    for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops;
                         s_ich_ops++) {
                      for (auto s_ops = 0; s_ops < c_och_ops; s_ops++) {
                        auto s_ops_index = s_ich_ops * c_och_ops + s_ops;
                        s_weight[s_index][s_ich_ops][s_ops] =
                          mem_weights[s_index][s_weight_index][s_ops_index];
                      }
                    }
                  }

                  // If it is the first reuse iteration, read the 1x1 weights
                  if constexpr (std::is_same<t_weight_1x1,
                                             std::nullptr_t>::value == false) {
                    for (auto s_ich_ops = 0; s_ich_ops < c_ich_ops;
                         s_ich_ops++) {
                      for (auto s_ops = 0; s_ops < c_och_ops; s_ops++) {
                        auto s_ops_index = s_ich_ops * c_och_ops + s_ops;
                        s_weight_1x1[0][s_ich_ops][s_ops] =
                          mem_weights_1x1[0][s_weight_index][s_ops_index];
                      }
                    }
                  }

                  if (s_num_ich == 0 || c_depth == 1) {
                    if constexpr (std::is_same<t_bias, std::nullptr_t>::value ==
                                  false) {
                      for (auto s_ops = 0; s_ops < c_bias_ops; s_ops++) {
                        s_bias[0][s_ops] = mem_bias[s_bias_index][s_ops];
                      }
                      if constexpr (std::is_same<t_bias_1x1,
                                                 std::nullptr_t>::value ==
                                    false) {
                        for (auto s_ops = 0; s_ops < c_bias_ops; s_ops++) {
                          s_bias_1x1[0][s_ops] =
                            mem_bias_1x1[s_bias_index][s_ops];
                        }
                      }
                    }
                    s_bias_index++;
                  }

                  s_weight_index++;
                }

                // Done to avoid partitioning of the stream and resource wasting
                // Theoretically with the add the input and output dimensions
                // should be equal, so we could perform the addition in
                // different iterations
                if constexpr (std::is_same<t_add_struct,
                                           std::nullptr_t>::value == false) {
                  // auto s_add_read = (s_num_och + s_num_ops_out) % c_add_ops;
                  if (ich_idx_packets == (c_add_ops / c_och_ops)) {
                    ich_idx_packets = 0;
                  }
                  if ((ich_idx_packets == 0) &&
                      (s_num_ich == c_ich - c_ich_ops)) {
                    ich_idx_add = 0;
                    for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                      // FIX FOR MOBILENETv2: Taking into account different
                      // output and input channels

                      t_add_struct s_add_struct = i_add[s_ow_ops].read();
                      s_add[s_ow_ops] = s_add_struct.data[0];
#ifndef __SYNTHESIS__
#ifdef DEBUG_ADD
                      ap_uint<t_add_st::width * c_add_ops> tmp = 0;
                      for (auto s_add_ops = 0; s_add_ops < c_add_ops;
                           s_add_ops++) {
                        tmp.range((t_add_st::width * (s_add_ops + 1)) - 1,
                                  t_add_st::width * s_add_ops) =
                          s_add[s_ow_ops][s_add_ops].range(t_add_st::width - 1,
                                                           0);
                      }
                      std::cout << "add " << tmp.to_string(16) << " "
                                << s_ow_ops << std::endl;
#endif
#endif
#ifndef __SYNTHESIS__
#ifdef DEBUG_ADD
                      for (auto s_ich_idx_add = 0; s_ich_idx_add < c_add_ops;
                           s_ich_idx_add++) {
                        std::cout << "add[" << s_ow_ops << "][" << s_ich_idx_add
                                  << "] "
                                  << s_add[s_ow_ops].data[0][s_ich_idx_add]
                                  << std::endl;
                      }
#endif
#endif
                  }
                }
              }

              std::array<t_input_data, c_ow_ops> s_input_1x1;
              if constexpr (std::is_same<t_acc_1x1_struct,
                                         std::nullptr_t>::value == false) {
#pragma HLS array_partition variable = s_input_1x1 type = complete
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  auto forward_index =
                    (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                  // s_input_1x1[s_ow_ops] = s_input[MO + MO%c_str -
                  // s_ow_ops*c_str];
                  s_input_1x1[c_ow_ops - s_ow_ops - 1] = s_input[forward_index];
                }
              }

              OCH_OPS_LOOP:
              for (auto s_ops = 0; s_ops < c_och_ops; s_ops += c_och_pack, ich_idx_add+=c_och_pack) {
                  auto s_och = s_num_och + s_num_ops_out + s_ops;
                OW_OPS_LOOP:
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops+=c_ow_pack) {
                  conv_pipe<
                    t_input,
                    t_weight,
                    t_weight_st,
                    t_bias,
                    t_add_vector,
                    t_add_st,
                    t_input_mod,
                    t_acc_struct,
                    t_acc,
                    t_acc_simd,
                    t_output,
                    t_output_clip,
                    t_output_mask,
                    c_reuse_iter,
                    c_fh,
                    c_fw,
                    c_index,
                    c_str,
                    c_och_ops,
                    c_iter_ops_out,
                    c_ich_ops,
                    c_add_ops,
                    c_ow_ops,
                    c_ow_pack,
                    c_och_pack,
                    c_relu,
                    c_leakyrelu,
                    c_ich,
                    c_och_depth,
                    c_in_bits,
                    c_simd_bits,
                    c_simd,
                    c_pad_bits,
                    c_int_pad_bits,
                    c_pad_acc_bits,
                    c_mask,
                    c_w_bits,
                    c_depth
                  > (
                    s_input,
                    s_weight,
                    s_bias,
                    s_ops,
                    s_num_ich,
                    s_ow_ops,
                    ich_idx_add,
                    s_add,
                    s_acc_buff[s_reuse][acc_group],
                    s_output_vector[och_packet_in_ops_out]
                  );

                  // TODO: split the loop in two parts controlled by different ops options
                  if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
                    if constexpr(c_ops_1x1 != c_och_ops){
                      if ((s_och > 0) | (s_ops > c_ops_1x1))  continue;
                    }
                    conv_pipe<
                      std::array<t_input_data, c_ow_ops>,
                      t_weight_1x1,
                      t_weight_1x1_st,
                      t_bias_1x1,
                      std::nullptr_t,
                      std::nullptr_t,
                      t_input_1x1,
                      t_acc_1x1_struct,
                      t_acc_1x1,
                      t_acc_simd_1x1,
                      t_output_1x1,
                      std::nullptr_t,
                      std::nullptr_t,
                      c_reuse_iter,
                      1,
                      1,
                      1,
                      1,
                      c_och_ops,
                      c_iter_ops_out,
                      c_ich_ops,
                      c_add_ops,
                      c_ow_ops,
                      c_ow_pack,
                      c_och_pack,
                      0,
                      0,
                      c_ich,
                      c_och_depth,
                      c_in_bits_1x1,
                      c_simd_bits_1x1,
                      c_simd_1x1,
                      c_pad_bits_1x1,
                      c_int_pad_bits_1x1,
                      c_pad_acc_bits_1x1,
                      c_mask_1x1,
                      c_w_bits_1x1,
                      c_depth
                    > (
                      s_input_1x1,
                      s_weight_1x1,
                      s_bias_1x1,
                      s_ops,
                      s_num_ich,
                      s_ow_ops,
                      ich_idx_add,
                      nullptr,
                      s_acc_buff_1x1[s_reuse][acc_group],
                      s_output_vector_1x1[och_packet_in_ops_out]
                    );
                  }
                }
              }

              if constexpr (std::is_same<t_forward_struct,
                                         std::nullptr_t>::value == false) {
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  if ((s_num_ops_out == (c_och_ops_out - c_iter_ops_out)) &&
                      (s_num_och == (c_och_depth - c_iter_och))) {
                    t_forward_struct s_forward;
                    // auto forward_index = MO + MO%c_str - s_ow_ops*c_str;

                    /* Compute the center of each window in input */
                    auto forward_index =
                      (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                    s_forward.data[0] = s_input[forward_index];
                    o_forward[s_ow_ops_out + s_ow_ops].write(s_forward);
#ifndef __SYNTHESIS__
#ifdef DEBUG_FORWARD
                    for (auto s_log_ich = 0; s_log_ich < c_ich_ops;
                         s_log_ich++) {
                      std::cout << "forward[" << s_ow_ops_out + s_ow_ops << "]["
                                << s_log_ich << "] "
                                << s_forward.data[0][s_log_ich] << std::endl;
                    }
#endif /* DEBUG_FORWARD */
#endif /* __SYNTHESIS__ */
                  }
                }
              }

              // Writing in output only when all the ich channels have been
              // considered and the och_ops_out packets are ready
              if (((s_num_ich == (c_ich - c_ich_ops)) || (c_depth == 1)) &&
                  (s_num_ops_out == (c_och_ops_out - c_iter_ops_out))) {

                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  t_output_struct temp_struct;
                  for (auto s_och_ops_out = 0;
                       s_och_ops_out < c_och_ops_out / c_iter_ops_out;
                       s_och_ops_out++) {
                    for (auto s_och_ops = 0; s_och_ops < c_iter_ops_out;
                         s_och_ops++) {
                      temp_struct
                        .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                        s_output_vector[s_och_ops_out][s_ow_ops][s_och_ops];
                    }
                  }
                  temp_struct.last = s_last;
                  o_output[s_ow_ops_out + s_ow_ops].write(temp_struct);

// #ifndef __SYNTHESIS__
//                   ap_uint<8 * c_och_ops_out> temp = 0;
//                   for (auto s_tmp = c_och_ops_out - 1; s_tmp >= 0; s_tmp--) {
//                     temp <<= 8;
//                     temp.range(7, 0) = temp_struct.data[0][s_tmp].range(7, 0);
//                   }
//                   std::cout << temp.to_string(16) << " " << s_ow_ops + s_ow_ops_out << std::endl;
// #endif
                  if constexpr (std::is_same<t_output_struct_1x1,
                                             std::nullptr_t>::value == false) {
                    t_output_struct_1x1 temp_struct_1x1;
                    for (auto s_och_ops_out = 0;
                         s_och_ops_out < c_och_ops_out / c_iter_ops_out;
                         s_och_ops_out++) {
                      for (auto s_och_ops = 0; s_och_ops < c_iter_ops_out;
                           s_och_ops++) {
                        temp_struct_1x1
                          .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                          s_output_vector_1x1[s_och_ops_out][s_ow_ops]
                                             [s_och_ops];
                      }
                    }
                    if (s_iter < c_iter_1x1) {
                      o_output_1x1[s_ow_ops_out + s_ow_ops].write(
                        temp_struct_1x1);
// #ifndef __SYNTHESIS__
//                       ap_uint<8 * c_och_ops_out> temp = 0;
//                       for (auto s_tmp = c_och_ops_out - 1; s_tmp >= 0; s_tmp--) {
//                         temp <<= 8;
//                         temp.range(7, 0) = temp_struct_1x1.data[0][s_tmp].range(7, 0);
//                       }
//                       std::cout << temp.to_string(16) << " " << s_ow_ops + s_ow_ops_out << " 1x1" << std::endl;
// #endif
                    }
                  }
                }
              }
            }
          }
        }
      }

#ifndef __SYNTHESIS__
        #ifdef DEBUG_RES
          for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
            for (auto s_och = 0; s_och < c_och_depth; s_och++) {
              std::cout <<  "PRE RES " << s_acc_buff[0][s_och*c_ow_ops+s_ow_ops] << std::endl;
              auto s_acc_log = quant_stream<
                t_output, t_output_clip, t_output_mask, t_acc, c_relu, c_leakyrelu
              >(s_acc_buff[0][s_och*c_ow_ops+s_ow_ops]);
              std::cout <<  "RES " << s_acc_log << std::endl;
            }
            if constexpr(std::is_same<t_output_struct_1x1, std::nullptr_t>::value == false) {
              for (auto s_och = 0; s_och < c_och_depth; s_och++) {
                auto s_acc_log = quant_stream<
                  t_output_1x1, std::nullptr_t, std::nullptr_t, t_acc_1x1, 0, 0
                >(s_acc_1x1_buff[0][s_och*c_ow_ops+s_ow_ops]);
                std::cout <<  "RES 1x1 " << s_acc_log << std::endl;
              }
            }
          }
        #endif /* DEBUG_RES */
      #endif /* __SYNTHESIS__ */
    }
  }

#ifndef __SYNTHESIS__
  // Check if input stream is empty
#ifndef SKIP_ASSERTION
  for (auto s_ow_ops = 0; s_ow_ops < 1; s_ow_ops++) {
    if (i_input[s_ow_ops].size() != 0) {
      std::cout << "ERROR: The input stream is not empty" << std::endl;
      std::cout << "i_input[" << s_ow_ops
                << "].size() = " << i_input[s_ow_ops].size() << std::endl;
      assert(false);
    }
  }
  // Check if input add is empty
  for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
    if constexpr (std::is_same<t_add_struct, std::nullptr_t>::value == false) {
      if (i_add[s_ow_ops].size() != 0) {
        std::cout << "ERROR: The input add stream is not empty" << std::endl;
        std::cout << "i_add[" << s_ow_ops
                  << "].size() = " << i_add[s_ow_ops].size() << std::endl;
        assert(false);
      }
    }
  }
  for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out; s_ow_ops_out++) {
    std::cout << "o_output[" << s_ow_ops_out
              << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
    if (o_output[s_ow_ops_out].size() == 0) {
      std::cout << "ERROR: The output stream is empty" << std::endl;
      std::cout << "o_output[" << s_ow_ops_out
                << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
      assert(false);
    }
  }
#endif /* SKIP_ASSERTION */
  if (c_depth == 1)
    std::cout << "end depth_conv_op " << c_ich << " " << c_depth << std::endl;
  else
    std::cout << "end conv_op " << c_ich << " " << c_depth << std::endl;
#endif /* __SYNTHESIS__ */
  }

template<typename t_input_struct, // input activation struct type
         typename t_input,        // input activation matrix type
         typename t_input_data,   // input activation vector type without window
         typename t_input_st,     // input activation standard type
         typename t_weight,       // weight matrix type
         typename t_weight_st,    // weight standard type
         typename t_bias,         // bias vector type
         typename t_bias_st,      // bias standard type
         typename t_add_struct,   // add activation struct type
         typename t_add_vector,   // add activation vector type
         typename t_add_st,       // add activation standard type
         typename t_forward_struct, // forward activation struct type
         typename t_input_mod,      // input activation modified type (input
         // quantization)
         typename t_input_1x1,     // input activation 1x1 type
         typename t_weight_1x1,    // weight 1x1 matrix type
         typename t_weight_1x1_st, // weight 1x1 standard type
         typename t_bias_1x1,      // bias 1x1 vector type
         typename t_bias_1x1_st,   // bias 1x1 standard type
         typename t_acc_struct,     // accumulator struct type
         typename t_acc,            // accumulator type
         typename t_acc_1x1_struct, // 1x1 accumulator struct type
         typename t_acc_1x1,        // 1x1 accumulator type
         typename t_output_struct,  // output activation struct type
         typename t_output_vector,  // output activation vector type
         typename t_output_st,      // output activation standard type
         typename t_output_clip,    // output activation clip type cast
         typename t_output_mask,    // output activation mask type cast
         typename t_output_struct_1x1,
         typename t_output_1x1,
         typename p_stream_t,
         const int c_ich_groups,
         const int c_ich,
         const int c_och,
         const int c_och_1x1,
         const int c_oh,
         const int c_ow,
         const int c_fh,
         const int c_fw,
         const int c_index,
         const int c_str,
         const int c_och_ops,
         const int c_och_ops_out,
         const int c_ich_ops,
         const int c_add_ops,
         const int c_ow_ops,
         const int c_ow_ops_out,
         const int c_bias_ops,
         const int c_reuse,
         const int c_ow_pack,
         const int c_och_pack,
         const int c_in_bits,
         const int c_in_ibits,
         const int c_w_bits,
         const int c_w_ibits,
         const int c_simd_bits,
         const int c_simd,
         const int c_mask,
         const int c_in_bits_1x1,
         const int c_in_ibits_1x1,
         const int c_w_bits_1x1,
         const int c_w_ibits_1x1,
         const int c_simd_bits_1x1,
         const int c_simd_1x1,
         const int c_mask_1x1,
         const int c_data_to_shift,
         const int c_stream_bits,
         const int c_read_weight,
         const int c_read_bias,
         const int c_read_weight_1x1,
         const int c_read_bias_1x1,
         const int c_relu,
         const int c_leakyrelu,
         const int c_depth>
void
conv_comp_wrap(
  hls::stream<p_stream_t> p_in[1],
  bool& s_init,
  hls::stream<p_stream_t> p_out[1],
  t_bias_st mem_bias[c_och / c_bias_ops][c_bias_ops],
  t_weight_st mem_weights[c_fh * c_fw][c_och * c_ich_groups / (c_och_ops * c_ich_ops)]
                         [c_och_ops * c_ich_ops],
  t_bias_1x1_st mem_bias_1x1[c_och_1x1 / c_bias_ops][c_bias_ops],
  t_weight_1x1_st mem_weights_1x1[1][c_och_1x1 * c_ich_groups / (c_och_ops * c_ich_ops)]
                                 [c_och_ops * c_ich_ops],
  hls::stream<t_input_struct> i_input[1],
  hls::stream<t_add_struct> i_add[c_ow_ops],
  hls::stream<t_forward_struct> o_forward[c_ow_ops],
  hls::stream<t_output_struct> o_output[c_ow_ops_out],
  hls::stream<t_output_struct_1x1> o_output_1x1[c_ow_ops_out])
{
  /* Generic convolution with bias and weights inside */
  constexpr unsigned c_fsz = c_fh * c_fw;
  constexpr unsigned c_ch_weight = c_ich_groups * c_och / (c_och_ops * c_ich_ops);
  constexpr unsigned c_ch_weight_1x1 = c_ich_groups * c_och_1x1 / (c_och_ops * c_ich_ops);
  
  /* The output ow_ops must be greater or equal and a multliples of ow_ops */
  static_assert(c_ow_ops_out >= c_ow_ops, "c_ow_ops_out >= c_ow_ops");
  static_assert(c_ow_ops_out % c_ow_ops == 0, "c_ow_ops_out % c_ow_ops == 0");
  
  if constexpr (c_depth == 0) {
    static_assert(c_och_ops_out >= c_och_ops, "c_och_ops_out >= c_och_ops");
    static_assert(c_och_ops_out % c_och_ops == 0, "c_och_ops_out % c_och_ops == 0");
  } else {
    static_assert(c_och_ops_out >= c_ich_ops, "c_och_ops_out >= c_ich_ops");
    static_assert(c_och_ops_out % c_ich_ops == 0, "c_och_ops_out % c_ich_ops == 0");
  }
  
  if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false) {
    /* If c_add_ops is not a multiple of 2 then the packing over och_ops could
    * create a mess, since to index the channel of the skip tensor is used och
    * % c_add_ops. Och is a multiple of 2 otherwise there will be no packing.
    * och % c_add_ops must not overflow. */
    // static_assert(c_add_ops % 2 == 0 || c_add_ops == 1, "c_add_ops % 2 != 0"); 
    static_assert(c_add_ops >= c_och_ops, "c_add_ops < c_och_ops");
    static_assert(c_add_ops % c_och_ops == 0, "c_add_ops % c_och_ops != 0");
    static_assert(c_depth == 0, "Depthwise convolutions with add are not supported");
  }
  
  const auto c_och_depth = (c_depth == 1) ? 1 : c_och;
  const auto c_o_index = c_oh * c_ow / c_ow_ops_out;
  const auto c_reuse_iter = c_reuse / c_ow_ops;
  const auto c_iter = c_reuse_iter;
  const auto c_och_1x1_depth = (c_depth == 1) ? 1 : c_och_1x1;
  const auto c_num_och_1x1 = c_och_1x1_depth / c_och_ops;
  const auto c_iter_1x1 = c_reuse_iter * c_num_och_1x1;
  const auto c_ops_1x1 = (c_och_1x1 < c_och_ops) ? c_och_1x1 : c_och_ops;
  const auto c_iter_ich = (c_depth == 1) ? c_och_ops_out : c_ich_ops;
  const auto c_iter_och = (c_depth == 1) ? 1 : c_och_ops_out;
  const auto c_iter_ops_out = (c_depth == 1) ? c_ich_ops : c_och_ops;
  constexpr int FW = (c_fw + (c_ow_ops - 1) * c_str);

#ifndef __SYNTHESIS__
  std::cout << "INFO: Call to conv_comp" << std::endl;
  std::cout << "\t\tc_ich = " << c_ich << std::endl;
  std::cout << "\t\tc_och = " << c_och << std::endl;
  std::cout << "\t\tc_oh = " << c_oh << std::endl;
  std::cout << "\t\tc_ow = " << c_ow << std::endl;
  std::cout << "\t\tc_fh = " << c_fh << std::endl;
  std::cout << "\t\tc_fw = " << c_fw << std::endl;
  std::cout << "\t\tc_index = " << c_index << std::endl;
  std::cout << "\t\tc_str = " << c_str << std::endl;
  std::cout << "\t\tc_och_ops = " << c_och_ops << std::endl;
  std::cout << "\t\tc_och_ops_out = " << c_och_ops_out << std::endl;
  std::cout << "\t\tc_ich_ops = " << c_ich_ops << std::endl;
  std::cout << "\t\tc_add_ops = " << c_add_ops << std::endl;
  std::cout << "\t\tc_ow_ops = " << c_ow_ops << std::endl;
  std::cout << "\t\tc_ow_ops_out = " << c_ow_ops_out << std::endl;
  std::cout << "\t\tc_bias_ops = " << c_bias_ops << std::endl;
  std::cout << "\t\tc_relu = " << c_relu << std::endl;
  std::cout << "\t\tc_reuse = " << c_reuse << std::endl;
  std::cout << "\t\tc_ow_pack = " << c_ow_pack << std::endl;
  std::cout << "\t\tc_och_pack = " << c_och_pack << std::endl;
  std::cout << "\t\tc_in_bits = " << c_in_bits << std::endl;
  std::cout << "\t\tc_in_ibits = " << c_in_ibits << std::endl;
  std::cout << "\t\tc_w_bits = " << c_w_bits << std::endl;
  std::cout << "\t\tc_w_ibits = " << c_w_ibits << std::endl;
  std::cout << "\t\tc_simd_bits = " << c_simd_bits << std::endl;
  std::cout << "\t\tc_simd = " << c_simd << std::endl;
  std::cout << "\t\tc_mask = " << c_mask << std::endl;
  std::cout << "\t\tc_in_bits_1x1 = " << c_in_bits_1x1 << std::endl;
  std::cout << "\t\tc_in_ibits_1x1 = " << c_in_ibits_1x1 << std::endl;
  std::cout << "\t\tc_w_bits_1x1 = " << c_w_bits_1x1 << std::endl;
  std::cout << "\t\tc_w_ibits_1x1 = " << c_w_ibits_1x1 << std::endl;
  std::cout << "\t\tc_simd_bits_1x1 = " << c_simd_bits_1x1 << std::endl;
  std::cout << "\t\tc_simd_1x1 = " << c_simd_1x1 << std::endl;
  std::cout << "\t\tc_mask_1x1 = " << c_mask_1x1 << std::endl;
  std::cout << "\t\tc_depth = " << c_depth << std::endl;
  std::cout << "\t\tc_data_to_shift = " << c_data_to_shift << std::endl;
#endif

  if (!s_init) {
    p_stream_t packed_data;

    /* Storing weights */
  STORE_WEIGHT_LOOP:
    for (auto s_ch = 0; s_ch < c_ch_weight; s_ch++) {
#pragma HLS pipeline off
      for (auto s_index = 0; s_index < c_fsz; s_index++) {
#pragma HLS pipeline off
        for (auto s_ops = 0; s_ops < c_och_ops * c_ich_ops; s_ops++) {
#pragma HLS pipeline off
          ap_uint<c_read_weight * c_stream_bits> unpacked_data = 0;
          for (auto s_read = 0; s_read < c_read_weight; s_read++) {
#pragma HLS pipeline off

            packed_data = p_in[0].read();
            unpacked_data <<= c_stream_bits;
            unpacked_data.range(c_stream_bits - 1, 0) =
              packed_data.range(c_stream_bits - 1, 0);
          }
          
          /* Step needed to support also 4 bit weights inside a 8 bit word*/
          t_weight_st tmp;
          tmp.range(t_weight_st::width - 1, 0) =
            unpacked_data.range(t_weight_st::width - 1, 0);
          mem_weights[s_index][s_ch][s_ops] = tmp;
        }
      }
    }

    if constexpr (std::is_same<t_bias_st, std::nullptr_t>::value == false) {

      /* Storing biases */
      const size_t c_loops_bias = c_och / c_bias_ops;
    STORE_BIAS_LOOP:
      for (auto s_ch = 0; s_ch < c_loops_bias; s_ch++) {
#pragma HLS pipeline off
        for (auto s_ops = 0; s_ops < c_bias_ops; s_ops++) {
#pragma HLS pipeline off
          ap_uint<c_read_bias * c_stream_bits> unpacked_data = 0;
          for (auto s_read = 0; s_read < c_read_bias; s_read++) {
#pragma HLS pipeline off

            packed_data = p_in[0].read();
            unpacked_data <<= c_stream_bits;
            unpacked_data.range(c_stream_bits - 1, 0) =
              packed_data.range(c_stream_bits - 1, 0);
          }
          t_bias_st tmp;
          tmp.range(t_bias_st::width - 1, 0) =
            unpacked_data.range(t_bias_st::width - 1, 0);
          mem_bias[s_ch][s_ops] = tmp;
        }
      }
    }

    if constexpr (std::is_same<t_weight_1x1_st, std::nullptr_t>::value ==
                  false) {

      /* Storing weights 1x1 */
      for (auto s_ch = 0; s_ch < c_ch_weight_1x1; s_ch++) {
        for (auto s_ops = 0; s_ops < c_och_ops * c_ich_ops; s_ops++) {
#pragma HLS pipeline off
          ap_uint<c_read_weight_1x1 * c_stream_bits> unpacked_data = 0;
          for (auto s_read = 0; s_read < c_read_weight_1x1; s_read++) {
#pragma HLS pipeline off

            packed_data = p_in[0].read();
            unpacked_data <<= c_stream_bits;
            unpacked_data.range(c_stream_bits - 1, 0) =
              packed_data.range(c_stream_bits - 1, 0);
          }
          t_weight_1x1_st tmp;
          tmp.range(t_weight_1x1_st::width - 1, 0) =
            unpacked_data.range(t_weight_1x1_st::width - 1, 0);
          mem_weights_1x1[0][s_ch][s_ops] = tmp;
        }
      }
    }

    if constexpr (std::is_same<t_bias_1x1_st, std::nullptr_t>::value == false) {

      /* Storing biases 1x1 */
      const size_t c_loops_bias_1x1 = c_och_1x1 / c_bias_ops;
      for (auto s_ch = 0; s_ch < c_loops_bias_1x1; s_ch++) {
        for (auto s_ops = 0; s_ops < c_bias_ops; s_ops++) {
#pragma HLS pipeline off
          ap_uint<c_read_bias_1x1 * c_stream_bits> unpacked_data = 0;
          for (auto s_read = 0; s_read < c_read_bias_1x1; s_read++) {
#pragma HLS pipeline off

            packed_data = p_in[0].read();
            unpacked_data <<= c_stream_bits;
            unpacked_data.range(c_stream_bits - 1, 0) =
              packed_data.range(c_stream_bits - 1, 0);
          }
          t_bias_1x1_st tmp;
          tmp.range(t_bias_1x1_st::width - 1, 0) =
            unpacked_data.range(t_bias_1x1_st::width - 1, 0);
          mem_bias_1x1[s_ch][s_ops] = tmp;
        }
      }
    }

    /* Shift remaining parameters to following layers */
  SHIFT_LOOP:
    for (auto p_left = 0; p_left < c_data_to_shift; p_left++) {
#pragma HLS pipeline II = 1 style = stp
      p_out[0].write(p_in[0].read());
    }

#ifndef __SYNTHESIS__
#ifndef SKIP_ASSERTIONS
    std::cout << "\tINFO: Finished saving parameters." << std::endl;
    /* Check that all the input streams are empty */
    if (p_in[0].size() > 0) {
      std::cout << "\tERROR: Not empty input stream. p_in[0].size() = "
                << p_in[0].size() << std::endl;
    }
    assert(p_in[0].size() == 0);

    /* Check that all the output streams are not empty */
    if (p_out[0].size() != c_data_to_shift) {
      std::cout << "\tERROR: Not full output stream. p_out.size() = "
                << p_out[0].size() << std::endl;
    }
    assert(p_out[0].size() == c_data_to_shift);
#endif /* SKIP_ASSERTIONS */
#endif
  }

  s_init = true;

  if constexpr (c_och_ops_out == c_iter_ops_out && c_och_depth == c_iter_och &&
                c_ich == c_iter_ich) {
    conv_comp_onchip_OW_OPS_OUT<t_input_struct,
                       t_input,
                       t_input_data,
                       t_input_st,
                       t_weight,
                       t_weight_st,
                       t_bias,
                       t_bias_st,
                       t_add_struct,
                       t_add_vector,
                       t_add_st,
                       t_forward_struct,
                       t_input_mod,
                       t_input_1x1,
                       t_weight_1x1,
                       t_weight_1x1_st,
                       t_bias_1x1,
                       t_bias_1x1_st,
                       t_acc_struct,
                       t_acc,
                       t_acc_1x1_struct,
                       t_acc_1x1,
                       t_output_struct,
                       t_output_vector,
                       t_output_st,
                       t_output_clip,
                       t_output_mask,
                       t_output_struct_1x1,
                       t_output_1x1,
                       p_stream_t,
                       c_ich_groups,
                       c_ich,
                       c_och,
                       c_och_1x1,
                       c_oh,
                       c_ow,
                       c_fh,
                       c_fw,
                       c_index,
                       c_str,
                       c_och_ops,
                       c_och_ops_out,
                       c_ich_ops,
                       c_add_ops,
                       c_ow_ops,
                       c_ow_ops_out,
                       c_bias_ops,
                       c_reuse,
                       c_ow_pack,
                       c_och_pack,
                       c_in_bits,
                       c_in_ibits,
                       c_w_bits,
                       c_w_ibits,
                       c_simd_bits,
                       c_simd,
                       c_mask,
                       c_in_bits_1x1,
                       c_in_ibits_1x1,
                       c_w_bits_1x1,
                       c_w_ibits_1x1,
                       c_simd_bits_1x1,
                       c_simd_1x1,
                       c_mask_1x1,
                       c_relu,
                       c_leakyrelu,
                       c_depth>(mem_bias,
                                mem_weights,
                                mem_bias_1x1,
                                mem_weights_1x1,
                                i_input,
                                i_add,
                                o_forward,
                                o_output,
                                o_output_1x1);
  } else if constexpr (c_och_ops_out == c_iter_ops_out && c_och_depth == c_iter_och) {
    conv_comp_onchip_ICH<t_input_struct,
                       t_input,
                       t_input_data,
                       t_input_st,
                       t_weight,
                       t_weight_st,
                       t_bias,
                       t_bias_st,
                       t_add_struct,
                       t_add_vector,
                       t_add_st,
                       t_forward_struct,
                       t_input_mod,
                       t_input_1x1,
                       t_weight_1x1,
                       t_weight_1x1_st,
                       t_bias_1x1,
                       t_bias_1x1_st,
                       t_acc_struct,
                       t_acc,
                       t_acc_1x1_struct,
                       t_acc_1x1,
                       t_output_struct,
                       t_output_vector,
                       t_output_st,
                       t_output_clip,
                       t_output_mask,
                       t_output_struct_1x1,
                       t_output_1x1,
                       p_stream_t,
                       c_ich_groups,
                       c_ich,
                       c_och,
                       c_och_1x1,
                       c_oh,
                       c_ow,
                       c_fh,
                       c_fw,
                       c_index,
                       c_str,
                       c_och_ops,
                       c_och_ops_out,
                       c_ich_ops,
                       c_add_ops,
                       c_ow_ops,
                       c_ow_ops_out,
                       c_bias_ops,
                       c_reuse,
                       c_ow_pack,
                       c_och_pack,
                       c_in_bits,
                       c_in_ibits,
                       c_w_bits,
                       c_w_ibits,
                       c_simd_bits,
                       c_simd,
                       c_mask,
                       c_in_bits_1x1,
                       c_in_ibits_1x1,
                       c_w_bits_1x1,
                       c_w_ibits_1x1,
                       c_simd_bits_1x1,
                       c_simd_1x1,
                       c_mask_1x1,
                       c_relu,
                       c_leakyrelu,
                       c_depth>(mem_bias,
                                mem_weights,
                                mem_bias_1x1,
                                mem_weights_1x1,
                                i_input,
                                i_add,
                                o_forward,
                                o_output,
                                o_output_1x1);

   } else if constexpr (c_och_ops_out == c_iter_ops_out) {
      conv_comp_onchip_OCH<t_input_struct,
              t_input,
              t_input_data,
              t_input_st,
              t_weight,
              t_weight_st,
              t_bias,
              t_bias_st,
              t_add_struct,
              t_add_vector,
              t_add_st,
              t_forward_struct,
              t_input_mod,
              t_input_1x1,
              t_weight_1x1,
              t_weight_1x1_st,
              t_bias_1x1,
              t_bias_1x1_st,
              t_acc_struct,
              t_acc,
              t_acc_1x1_struct,
              t_acc_1x1,
              t_output_struct,
              t_output_vector,
              t_output_st,
              t_output_clip,
              t_output_mask,
              t_output_struct_1x1,
              t_output_1x1,
              p_stream_t,
              c_ich_groups,
              c_ich,
              c_och,
              c_och_1x1,
              c_oh,
              c_ow,
              c_fh,
              c_fw,
              c_index,
              c_str,
              c_och_ops,
              c_och_ops_out,
              c_ich_ops,
              c_add_ops,
              c_ow_ops,
              c_ow_ops_out,
              c_bias_ops,
              c_reuse,
              c_ow_pack,
              c_och_pack,
              c_in_bits,
              c_in_ibits,
              c_w_bits,
              c_w_ibits,
              c_simd_bits,
              c_simd,
              c_mask,
              c_in_bits_1x1,
              c_in_ibits_1x1,
              c_w_bits_1x1,
              c_w_ibits_1x1,
              c_simd_bits_1x1,
              c_simd_1x1,
              c_mask_1x1,
              c_relu,
              c_leakyrelu,
              c_depth>(mem_bias,
                       mem_weights,
                       mem_bias_1x1,
                       mem_weights_1x1,
                       i_input,
                       i_add,
                       o_forward,
                       o_output,
                       o_output_1x1);
    } else {
      conv_comp_onchip<t_input_struct,
                       t_input,
                       t_input_data,
                       t_input_st,
                       t_weight,
                       t_weight_st,
                       t_bias,
                       t_bias_st,
                       t_add_struct,
                       t_add_vector,
                       t_add_st,
                       t_forward_struct,
                       t_input_mod,
                       t_input_1x1,
                       t_weight_1x1,
                       t_weight_1x1_st,
                       t_bias_1x1,
                       t_bias_1x1_st,
                       t_acc_struct,
                       t_acc,
                       t_acc_1x1_struct,
                       t_acc_1x1,
                       t_output_struct,
                       t_output_vector,
                       t_output_st,
                       t_output_clip,
                       t_output_mask,
                       t_output_struct_1x1,
                       t_output_1x1,
                       p_stream_t,
                       c_ich_groups,
                       c_ich,
                       c_och,
                       c_och_1x1,
                       c_oh,
                       c_ow,
                       c_fh,
                       c_fw,
                       c_index,
                       c_str,
                       c_och_ops,
                       c_och_ops_out,
                       c_ich_ops,
                       c_add_ops,
                       c_ow_ops,
                       c_ow_ops_out,
                       c_bias_ops,
                       c_reuse,
                       c_ow_pack,
                       c_och_pack,
                       c_in_bits,
                       c_in_ibits,
                       c_w_bits,
                       c_w_ibits,
                       c_simd_bits,
                       c_simd,
                       c_mask,
                       c_in_bits_1x1,
                       c_in_ibits_1x1,
                       c_w_bits_1x1,
                       c_w_ibits_1x1,
                       c_simd_bits_1x1,
                       c_simd_1x1,
                       c_mask_1x1,
                       c_relu,
                       c_leakyrelu,
                       c_depth>(mem_bias,
                                mem_weights,
                                mem_bias_1x1,
                                mem_weights_1x1,
                                i_input,
                                i_add,
                                o_forward,
                                o_output,
                                o_output_1x1);
    }
  }
} // namespace nn2fpga
#pragma GCC diagnostic pop

#endif // NN2FPGA_PACKED_CONV_H_