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

template<class t_output,
         class t_output_clip,
         class t_output_mask,
         class t_acc,
         int c_relu>
t_output
quant_stream(t_acc i_acc)
{
#pragma HLS inline

  t_acc s_acc = i_acc;

  if (c_relu == 1) {
    s_acc = relu_op<t_acc>(s_acc);
  }

  if constexpr (std::is_same<t_output_clip, std::nullptr_t>::value == false) {
    s_acc = t_output_clip(s_acc);
  }

  if constexpr (std::is_same<t_output_mask, std::nullptr_t>::value == false) {
    s_acc = t_output_mask(s_acc);
  }

  return t_output(s_acc);
}

template<class t_output,
         class t_output_clip,
         class t_output_mask,
         class t_acc,
         int c_relu>
t_output
quant_and_add_stream(t_acc i_acc, t_output_clip i_add)
{
#pragma HLS inline

  t_acc s_acc = i_acc;

  if constexpr (c_relu == 1) {
    s_acc = relu_op<t_acc>(s_acc);
  }

  if constexpr (std::is_same<t_output_clip, std::nullptr_t>::value == false) {
    s_acc = t_output_clip(s_acc);
  }
  
  s_acc += i_add;

  if constexpr (std::is_same<t_output_mask, std::nullptr_t>::value == false) {
    s_acc = t_output_mask(s_acc);
  }

  return t_output(s_acc);
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
         class t_input_st,
         class t_weight,
         class t_weight_st,
         class t_bias,
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
          const uint32_t s_num_ops_out,
          const uint32_t ich_idx_add,
          const t_add i_add[c_ow_ops],
          t_acc i_acc_buff[c_ops * c_ow_ops],
          t_output s_output_struct[c_ow_ops][c_iter_ops_out])
{
#pragma HLS inline

  const int FW = (c_fw + (c_ow_ops - 1) * c_str);

  if (c_och_pack > 1) {
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

            if constexpr(std::is_same<t_input_mod, std::nullptr_t>::value == false) {
              s_b_ext.range(s_w_index + c_bits - 1, s_w_index) = t_input_mod(i_input[s_index][ich_idx]).range(c_bits-1, 0);
              if constexpr(std::is_same<typename t_input_mod::Base::Base, _AP_ROOT_TYPE<t_input_mod::Base::width, true>>::value) {
                for (auto pos = s_w_index+c_bits; pos < 18; pos++) {
                  s_b_ext.range(pos,pos) = s_b_ext.range(s_w_index + c_bits - 1, s_w_index + c_bits - 1);
                }
              }
            }
            else {
              s_b_ext.range(s_w_index + c_bits - 1, s_w_index) = i_input[s_index][ich_idx].range(c_bits-1, 0);
              if constexpr(std::is_same<typename t_input_st::Base::Base, _AP_ROOT_TYPE<t_input_st::Base::width, true>>::value) {
                for (auto pos = s_w_index+c_bits; pos < 18; pos++) {
                  s_b_ext.range(pos,pos) = s_b_ext.range(s_w_index + c_bits - 1, s_w_index + c_bits - 1);
                }
              }
            }

            #ifndef __SYNTHESIS__
              #ifdef DEBUG_CONV
                std::cout << "A" << s_index << " " << i_input[s_index][ich_idx] << std::endl;
              #endif
            #endif

          }

          auto s_index = s_fh * c_fw + s_fw;
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
          auto s_w_index = s_och_pack*c_ow_pack+s_ow_pack;
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
                                   t_acc,
                                   c_relu>(
                s_acc[s_och_pack * c_ow_pack + s_ow_pack],
                i_add[s_ow_pack + s_ow_ops][ich_idx_add + s_och_pack]);
          } else {
            s_output_struct[s_ow_ops + s_ow_pack][ops + s_och_pack] =
              quant_stream<t_output,
                           t_output_clip,
                           t_output_mask,
                           t_acc,
                           c_relu>(s_acc[s_och_pack * c_ow_pack + s_ow_pack]);
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
            t_input_st s_input_ext[c_ow_pack];

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
              double_packing_wrap<t_input_st,
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
                           c_relu>(s_acc[s_ow_pack]);
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
                                     t_acc,
                                     c_relu>(
                  s_acc[s_ow_pack], i_add[s_ow_pack + s_ow_ops][ich_idx_add]);
            } else {
              s_output_struct[s_ow_ops + s_ow_pack][ops] =
                quant_stream<t_output,
                             t_output_clip,
                             t_output_mask,
                             t_acc,
                             c_relu>(s_acc[s_ow_pack]);
            }
          }
        }
      }
    } else {

      t_acc s_acc = 0;
      t_acc s_acc_base = 0;
      s_acc = 0;

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
            auto s_data = i_input[s_index_act][ich_idx];
            if constexpr(std::is_same<t_input_mod, std::nullptr_t>::value == false)
              s_data = t_input_mod(s_data);
            s_acc += s_data * i_weight[s_index][ich_idx][ops];
          }
        }

        if constexpr (c_depth == 1) {
          s_output_struct[s_ow_ops][ich_idx] =
            quant_stream<t_output, t_output_clip, t_output_mask, t_acc, c_relu>(
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
                                   t_acc,
                                   c_relu>(
                s_acc, i_add[s_ow_ops][ich_idx_add]);
          } else {
            s_output_struct[s_ow_ops][ops] =
              quant_stream<t_output,
                           t_output_clip,
                           t_output_mask,
                           t_acc,
                           c_relu>(s_acc);
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
  /* #pragma HLS inline */
  // Generic Convolution Computation

#pragma HLS expression_balance off
  // The output ow_ops must be greater or equal and a mutliples of ow_ops
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
  constexpr int MO = (c_fh * FW) / 2;

  /* Groups of accumulator for a total of och * ow_ops. The one that must be
   * available in parallel are c_ops * c_ow_ops. This data structure is not used
   * by depth convolutions */
  t_acc s_acc_buff[c_reuse_iter][c_och_depth / c_ops][c_ops * c_ow_ops];
  t_acc_1x1 s_acc_buff_1x1[c_reuse_iter][c_och_depth / c_ops][c_ops * c_ow_ops];
  auto acc_group = 0;
#pragma HLS array_partition variable = s_acc_buff type = complete dim = 3
#pragma HLS array_partition variable = s_acc_buff_1x1 type = complete dim = 3

  t_input s_input;
#pragma HLS array_partition variable = s_input type = complete
#pragma HLS aggregate variable = s_input
  
  bool s_last = false;
  t_weight s_weight[c_index];
#pragma HLS array_partition variable = s_weight type = complete
  
  t_weight_1x1 s_weight_1x1[1];
#pragma HLS array_partition variable = s_weight_1x1 type = complete
  
  t_bias s_bias;
#pragma HLS array_partition variable = s_bias type = complete dim = 1
  
  t_bias_1x1 s_bias_1x1;
#pragma HLS array_partition variable = s_bias_1x1 type = complete dim = 1
  
  t_add s_add[c_ow_ops];
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
        for (auto s_num_och = 0, ich_idx_packet = 0; s_num_och < c_och_depth;
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
              if (((s_num_och == 0) && (s_num_ops_out == 0)) || (c_depth == 1)) {
                t_input_struct s_input_struct = i_input[0].read();
                #pragma HLS array_partition variable=s_input_struct.data type=complete
                s_input = s_input_struct.data;
                /* Sending last only at the bottom right data */
                s_last = s_input_struct.last;
                #ifndef __SYNTHESIS__
                  #ifdef DEBUG_INPUT
                    for (auto s_index = 0; s_index < c_fh*FW; s_index++) {
                      for (auto s_log_ich = 0; s_log_ich < c_in_ops; s_log_ich++) {
                        std::cout << "input[" << s_index << "][" << s_log_ich << "]" << " " << s_input[s_index][s_log_ich] << std::endl;
                      }
                    }
                  #endif
                #endif
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
                    #ifndef __SYNTHESIS__
                      #ifdef DEBUG_ADD
                        std::cout << "reading s_add[" << s_ow_ops << "]" << std::endl;
                      #endif
                    #endif
                    t_add_struct s_add_struct = i_add[s_ow_ops].read();
                    s_add[s_ow_ops] = s_add_struct.data[0];   
                    #ifndef __SYNTHESIS__
                      #ifdef DEBUG_ADD
                      for (auto s_ich_idx_add = 0; s_ich_idx_add < c_add_ops; s_ich_idx_add++) {
                        std::cout << "add[" << s_ow_ops << "][" << s_ich_idx_add << "] " << s_add[s_ow_ops].data[0][s_ich_idx_add] << std::endl;
                      }
                      #endif
                    #endif
                  }
                }
              }

              std::array<t_input_data, c_ow_ops> s_input_1x1;
              if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
                #pragma HLS array_partition variable = s_input_1x1 type = complete
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  auto forward_index =
                    (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                  // s_input_1x1[s_ow_ops] = s_input[MO + MO%c_str - s_ow_ops*c_str];
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
                    t_input_st,
                    t_weight,
                    t_weight_st,
                    t_bias,
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
                    s_num_ops_out,
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
                      t_input_st,
                      t_weight_1x1,
                      t_weight_1x1_st,
                      t_bias_1x1,
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
                      s_num_ops_out,
                      ich_idx_add,
                      nullptr,
                      s_acc_buff_1x1,
                      s_output_vector_1x1[och_packet_in_ops_out]
                    );
                  }
                }
              }

              if constexpr(std::is_same<t_forward_struct, std::nullptr_t>::value == false) {
                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  if ((s_num_ops_out == (c_ops_out - c_iter_ops_out)) && (s_num_och == (c_och_depth - c_iter_och))) {
                    t_forward_struct s_forward;
                    // auto forward_index = MO + MO%c_str - s_ow_ops*c_str;
                    auto forward_index =
                      (c_fh / 2 + 1) * FW - c_fw / 2 - s_ow_ops * c_str - 1;
                    s_forward.data[0] = s_input[forward_index];
                    s_forward.last = false;
                    o_forward[s_ow_ops_out+s_ow_ops].write(s_forward);
                    #ifndef __SYNTHESIS__
                      #ifdef DEBUG_FORWARD
                        for (auto s_log_ich = 0; s_log_ich < c_in_ops; s_log_ich++) {
                          std::cout << "forward[" << s_ow_ops_out+s_ow_ops << "][" << s_log_ich << "] " << s_forward.data[0][s_log_ich] << std::endl;
                        }
                      #endif
                    #endif
                  }
                }
              }

              // Writing in output only when all the ich channels have been
              // considered and the och_ops_out packets are ready
              if (((s_num_ich == (c_ich - c_in_ops)) || (c_depth == 1)) &&
                  (s_num_ops_out == (c_ops_out - c_iter_ops_out))) {

                for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                  t_output_struct temp_struct;
                  t_output_struct_1x1 temp_struct_1x1;
                  for (auto s_och_ops_out = 0;
                       s_och_ops_out < c_ops_out / c_iter_ops_out;
                       s_och_ops_out++) {
                    for (auto s_och_ops = 0; s_och_ops < c_iter_ops_out;
                         s_och_ops++) {
                      temp_struct
                        .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                        s_output_vector[s_och_ops_out][s_ow_ops][s_och_ops];

                      // TEST THIS STUFF
                      if constexpr (std::is_same<t_output_struct_1x1,
                                                 std::nullptr_t>::value ==
                                    false) {
                        temp_struct_1x1
                          .data[0][s_och_ops_out * c_iter_ops_out + s_och_ops] =
                          s_output_vector_1x1[s_och_ops_out][s_ow_ops]
                                             [s_och_ops];
                      }
                    }
                  }
                  temp_struct.last = s_last;
                  o_output[s_ow_ops_out + s_ow_ops].write(
                    temp_struct);

                  // TEST THIS STUFF
                  if constexpr (std::is_same<t_output_struct_1x1,
                                             std::nullptr_t>::value == false) {
                    if (s_iter < c_iter_1x1) {
                      temp_struct_1x1.last = s_last;
                      o_output_1x1[s_ow_ops_out + s_ow_ops].write(
                        temp_struct_1x1);
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
                t_output, t_output_clip, t_output_mask, t_acc, c_relu
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
    for (auto s_ow_ops = 0; s_ow_ops < 1; s_ow_ops++) {
      if (i_input[s_ow_ops].size() != 0) {
        std::cout << "ERROR: The input stream is not empty" << std::endl;
        std::cout << "i_input[" << s_ow_ops << "].size() = " << i_input[s_ow_ops].size() << std::endl;
        assert (false);
      }
    }
    // Check if input add is empty
    for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
      if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false) {
        if (i_add[s_ow_ops].size() != 0) {
          std::cout << "ERROR: The input add stream is not empty" << std::endl;
          std::cout << "i_add[" << s_ow_ops << "].size() = " << i_add[s_ow_ops].size() << std::endl;
          assert (false);
        }
      }
    }
    for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out; s_ow_ops_out++) {
      std::cout << "o_output[" << s_ow_ops_out << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
      if (o_output[s_ow_ops_out].size() == 0) {
        std::cout << "ERROR: The output stream is empty" << std::endl;
        std::cout << "o_output[" << s_ow_ops_out << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
        assert (false);
      }
    }
    if (c_depth == 1)
      std::cout << "end depth_conv_op " << c_ich << " " << c_depth << std::endl;
    else
      std::cout << "end conv_op " << c_ich << " " << c_depth << std::endl;
  #endif /* __SYNTHESIS__ */
}

//////////////////////////////////////////////////////////////////////////////

}  // namespace nn2fpga

#endif // NN2FPGA_PACKED_CONV_H_
