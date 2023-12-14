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
template <class t_input, class t_input_st, class t_weight, class t_weight_st, class t_bias,
          class t_add_struct, class t_input_mod, class t_acc_struct, class t_acc, 
          class t_acc_simd, class t_output_struct, class t_output, class t_output_clip, 
          class t_output_mask, int c_reuse, int c_fh, int c_fw, int c_index, int c_str,
          int c_ops, int c_in_ops, int c_ow_ops, int c_ow_pack, int c_och_pack, 
          int c_relu, int c_ich, int c_och, int c_bits, int c_simd_bits,
          int c_simd, int c_pad_bits, int c_int_pad_bits, int c_pad_acc_bits, int c_mask, 
          int c_w_bits, int c_depth>
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
    uint32_t s_ow_ops,
    bool last,
    t_add_struct i_add[c_ow_ops],
    t_acc i_acc_buff[c_reuse][c_och*c_ow_ops],
    t_output_struct s_output_struct[c_ow_ops]) {
#pragma HLS inline

  const int FW = (c_fw+(c_ow_ops-1)*c_str);

  if constexpr(c_och_pack > 1) {

    t_acc s_acc[c_ow_pack*c_och_pack];
    t_acc s_acc_base[c_ow_pack*c_och_pack];

    ap_uint<48> s_acc_simd[c_simd];

    for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
      for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
        auto s_w_index = s_och_pack*c_ow_pack+s_ow_pack;
        s_acc[s_w_index] = 0;
        if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
          if (ich == 0) {
            auto s_bias = i_bias[0][ops+s_och_pack];
            #ifndef __SYNTHESIS__
              #ifdef DEBUG_CONV
                std::cout << "B" << " " << s_bias << std::endl;
              #endif
            #endif
            s_acc[s_w_index] = s_bias;
          }
        }
      }
    }

    if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false) {
      for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
        if (ich == (och+s_och_pack)) {
        // TODO: Add support for multiple inputs (vector type)
          for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
            s_acc[s_och_pack*c_ow_pack+s_ow_pack] += i_add[s_ow_pack+s_ow_ops].data[0][ich_idx_add];
          }
        }
      }
    }

    // If c_depth is 1 then there is no need to accumulate the previous
    // results
    for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
      for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
        auto s_w_index = s_och_pack*c_ow_pack+s_ow_pack;
        auto s_r_index = (och+s_och_pack)*c_ow_ops+s_ow_pack+s_ow_ops;
        if constexpr(c_depth == 1) {
          s_acc_base[s_w_index] = 0;
        } else {
          s_acc_base[s_w_index] = i_acc_buff[reuse][s_r_index];
        }
      }
    }

    for (auto s_simd = 0; s_simd < c_simd; s_simd++) {
      s_acc_simd[s_simd] = 0;
    }

    for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
      for (auto s_fw = 0; s_fw < c_fw; s_fw++) {
        ap_int<27> s_data = 0;
        ap_int<18> s_b_ext = 0;

        ap_int<27> s_a_d_ext[c_ow_pack];

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

          auto s_index = s_fh*FW+s_fw+(c_ow_ops-(s_ow_pack+s_ow_ops)-1)*c_str;

          auto s_w_index = s_ow_pack*(c_pad_acc_bits);

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

        auto s_index = s_fh*c_fw+s_fw;
        s_acc_simd[s_index & c_mask] += s_data * s_b_ext;

      }
    }

    for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
      for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
        auto s_index_r = s_och_pack*c_ow_pack+s_ow_pack;
        for (auto s_simd = 0; s_simd < c_simd; s_simd++) {
          t_acc_simd s_acc_simd_value = 0;
          t_acc_simd s_acc_adj = 0;
          if ((s_index_r > 0))
            s_acc_adj.range(0,0) = s_acc_simd[s_simd].range(c_pad_acc_bits*(s_index_r)-1, c_pad_acc_bits*(s_index_r)-1);
          s_acc_simd_value.range(c_pad_acc_bits-1, 0) = s_acc_simd[s_simd].range(c_pad_acc_bits*(s_index_r+1)-1, c_pad_acc_bits*(s_index_r));
          s_acc[s_index_r] += s_acc_simd_value + s_acc_adj;
        }
        #ifndef __SYNTHESIS__
          #ifdef DEBUG_CONV
            if (och == 0)
              std::cout << "RES " << s_acc[s_index_r] << std::endl;
          #endif
        #endif
      }
    }

    if (ich != 0) {
      for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
        for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
          auto s_w_index = s_och_pack*c_ow_pack+s_ow_pack;
          s_acc[s_w_index] += s_acc_base[s_w_index];
        }
      }
    }

    #ifndef __SYNTHESIS__
      for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
        for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
          #ifdef DEBUG_ACC
            std::cout <<  "ACC " << s_acc[s_och_pack*c_ow_pack+s_ow_pack] << std::endl;
          #endif
        }
      }
    #endif

    // If c_depth is 1 then there is no need to store the previous
    // results
    if constexpr(c_depth == 0) {
      for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
        for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
          auto s_r_index = s_och_pack*c_ow_pack+s_ow_pack;
          i_acc_buff[reuse][(och+s_och_pack)*c_ow_ops+s_ow_ops+s_ow_pack] = s_acc[s_r_index];
        }
      }
    }

    if (ich == c_ich-1) {
      for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
        for (auto s_och_pack = 0; s_och_pack < c_och_pack; s_och_pack++) {
          s_output_struct[s_ow_ops+s_ow_pack].data[0][ops+s_och_pack] = quant_stream<
            t_output, t_output_clip, t_output_mask, t_acc, c_relu
          >(s_acc[s_och_pack*c_ow_pack+s_ow_pack]);
          s_output_struct[s_ow_ops+s_ow_pack].last = last;
        }
      }
    }
  } else {

    if constexpr(c_ow_pack > 1) {
      
      t_acc s_acc[c_ow_pack];
      t_acc s_acc_base[c_ow_pack];

      ap_uint<48> s_acc_simd[c_simd];

      if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
        if (ich == 0) {
          auto s_bias = i_bias[0][ops];
          for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
            #ifndef __SYNTHESIS__
              // #ifdef DEBUG_CONV
              //   std::cout << "B" << s_bias << " ";
              // #endif
            #endif
            s_acc[s_ow_pack] = s_bias;
          }
        } else {
          for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
            s_acc[s_ow_pack] = 0;
          }
        }
      } else {
        for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
          s_acc[s_ow_pack] = 0;
        }
      }

      if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false) {
        if (ich == och) {
          // TODO: Add support for multiple inputs (vector type)
          for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
            s_acc[s_ow_pack] += i_add[s_ow_pack+s_ow_ops].data[0][ich_idx_add];
          }
        }
      }

      // If c_depth is 1 then there is no need to accumulate the previous
      // results
      if constexpr(c_depth == 1) {
        for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
          s_acc_base[s_ow_pack] = 0;
        }
      } else {
        for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
          s_acc_base[s_ow_pack] = i_acc_buff[reuse][och*c_ow_ops+s_ow_pack+s_ow_ops];
        }
      }

      for (auto s_simd = 0; s_simd < c_simd; s_simd++) {
        s_acc_simd[s_simd] = 0;
      }

      for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
        for (auto s_fw = 0; s_fw < c_fw; s_fw++) {
          ap_int<27> s_data = 0;
          ap_int<18> s_weight = 0;

          ap_int<27> s_input_ext[c_ow_pack];

          for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {

            auto s_index = s_fh*FW+s_fw+(c_ow_ops-(s_ow_pack+s_ow_ops)-1)*c_str;

            s_input_ext[s_ow_pack] = 0;

            if constexpr(std::is_same<t_input_mod, std::nullptr_t>::value == false) {
              s_input_ext[s_ow_pack].range(c_pad_bits*s_ow_pack+c_bits-1, c_pad_bits*s_ow_pack) = t_input_mod(i_input[s_index][ich_idx]).range(c_bits-1, 0);
              if constexpr(std::is_same<typename t_input_mod::Base::Base, _AP_ROOT_TYPE<t_input_mod::Base::width, true>>::value) {
                for (auto pos = c_pad_bits*s_ow_pack+c_bits; pos < 27; pos++) {
                  s_input_ext[s_ow_pack].range(pos,pos) = s_input_ext[s_ow_pack].range(c_pad_bits*s_ow_pack+c_bits-1, c_pad_bits*s_ow_pack+c_bits-1);
                }
              }
            }
            else {
              s_input_ext[s_ow_pack].range(c_pad_bits*s_ow_pack+c_bits-1, c_pad_bits*s_ow_pack) = i_input[s_index][ich_idx].range(c_bits-1, 0);
              if constexpr(std::is_same<typename t_input_st::Base::Base, _AP_ROOT_TYPE<t_input_st::Base::width, true>>::value) {
                for (auto pos = c_pad_bits*s_ow_pack+c_bits; pos < 27; pos++) {
                  s_input_ext[s_ow_pack].range(pos,pos) = s_input_ext[s_ow_pack].range(c_pad_bits*s_ow_pack+c_bits-1, c_pad_bits*s_ow_pack+c_bits-1);
                }
              }
            }

            #ifdef DEBUG_ACT
              std::cout << "A" << s_index << " " << i_input[s_index][ich_idx] << std::endl;
            #endif

            #ifndef SIMD_DSP
              s_data += s_input_ext[s_ow_pack];
            #endif

          }
          auto s_index = s_fh*c_fw+s_fw;

          s_weight.range(c_w_bits - 1, 0) = i_weight[s_index][ich_idx][ops].range(c_w_bits - 1, 0);
          // Check if the type is signed and then perform extension
          if constexpr(std::is_same<typename t_weight_st::Base::Base, _AP_ROOT_TYPE<t_weight_st::Base::width, true>>::value) {
            for (auto pos = c_w_bits; pos < 18; pos++) {
              s_weight.range(pos,pos) = s_weight.range(c_w_bits - 1, c_w_bits - 1);
            }
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
        for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {

          t_acc_simd s_acc_simd_value = 0;
          t_acc_simd s_acc_adj = 0;

          if (s_ow_pack > 0)
            s_acc_adj.range(0,0) = s_acc_simd[s_simd].range(c_pad_acc_bits*(s_ow_pack)-1, c_pad_acc_bits*(s_ow_pack)-1);
          s_acc_simd_value.range(c_pad_acc_bits-1, 0) = s_acc_simd[s_simd].range(c_pad_acc_bits*(s_ow_pack+1)-1, c_pad_acc_bits*s_ow_pack);
          s_acc[s_ow_pack] += s_acc_simd_value + s_acc_adj;
        }
      }

      if (ich != 0) {
        for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
          s_acc[s_ow_pack] += s_acc_base[s_ow_pack];
        }
      }

      // If c_depth is 1 then there is no need to store the previous
      // results
      if constexpr(c_depth == 0) {
        for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
          i_acc_buff[reuse][och*c_ow_ops+s_ow_ops+s_ow_pack] = s_acc[s_ow_pack];
        }
      }

      for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
        #ifndef __SYNTHESIS__
          #ifdef DEBUG_ACC
            std::cout <<  "ACC " << s_acc[s_ow_ops+s_ow_pack] << std::endl;
          #endif
        #endif
      }

      if (ich == c_ich-1) {
        for (auto s_ow_pack = 0; s_ow_pack < c_ow_pack; s_ow_pack++) {
          s_output_struct[s_ow_ops+s_ow_pack].data[0][ops] = quant_stream<
            t_output, t_output_clip, t_output_mask, t_acc, c_relu
          >(s_acc[s_ow_pack]);
          s_output_struct[s_ow_ops+s_ow_pack].last = last;
        }
      }

      // return s_acc_struct;
    }


    if constexpr(c_ow_pack == 1) {

      t_acc s_acc = 0;
      t_acc s_acc_base = 0;

      auto s_index_ops = (c_depth == 1) ? ich_idx : (ops);
      if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false) {

        // FIX: Seems that when there is the skip connection the binding of the
        // DSPs is not working, moving things before
        t_acc s_acc_add = 0;
        if ((ich == och) | (c_depth == 1)) {
          s_acc_add = i_add[s_ow_ops].data[0][ich_idx_add];
          #ifndef __SYNTHESIS__
            // if (c_depth == 1)
            // #ifdef DEBUG_CONV
            //   std::cout << "ADD " << s_acc_add << " ";
            // #endif
          #endif
        }

        if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
          if ((ich == 0) | (c_depth == 1)) {
            s_acc = i_bias[0][s_index_ops] + s_acc_add;
            #ifndef __SYNTHESIS__
              // #ifdef DEBUG_CONV
              //   std::cout << "B " << i_bias[0][s_index_ops] << " ";
              // #endif
            #endif
          }
          else
            s_acc = s_acc_add;
        } else {
          s_acc = s_acc_add;
        }

      } else {
        if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
          if ((ich == 0) | (c_depth == 1)) {
            s_acc = i_bias[0][s_index_ops];
            #ifndef __SYNTHESIS__
              #ifdef DEBUG_CONV
                std::cout << "B " << i_bias[0][s_index_ops] << std::endl;
              #endif
            #endif
          }
          else
            s_acc = 0;
        } else {
          s_acc = 0;
        }
      }

      // If c_depth is 1 then there is no need to accumulate the previous
      // results
      if constexpr(c_depth == 1) {
        s_acc_base = 0;
      } else {
        s_acc_base = i_acc_buff[reuse][och*c_ow_ops+s_ow_ops];
      }

      for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
        for (auto s_fw = 0; s_fw < c_fw; s_fw++) {
          auto s_index_act = s_fh*FW+s_fw+(c_ow_ops-s_ow_ops-1)*c_str;
          auto s_index = s_fh*c_fw+s_fw;
          auto s_data = i_input[s_index_act][ich_idx];
          if constexpr(std::is_same<t_input_mod, std::nullptr_t>::value == false)
            s_data = t_input_mod(s_data);
          s_acc += s_data * i_weight[s_index][ich_idx][ops];
          #ifndef __SYNTHESIS__
            // #ifdef DEBUG_CONV
            //   std::cout << "W" << s_index << " " << i_weight[s_index][ich_idx][ops] << " ";
            //   std::cout << "A" << s_index << " " << i_input[s_index][ich_idx] << " " << s_data << " ";
            // #endif
            #ifdef DEBUG_ACT
              std::cout << "A" << s_index << " " << s_data << std::endl;
            #endif
            #ifdef DEBUG_WEIGHTS
              std::cout << i_weight[c_index - 1 - s_index][ich_idx][ops] << std::endl;
            #endif
          #endif
        }
      }

      if (ich != 0) s_acc += s_acc_base;

      #ifndef __SYNTHESIS__
        #ifdef DEBUG_ACC
          std::cout <<  "ACC " << s_acc << std::endl;
        #endif
      #endif

      // If c_depth is 1 then there is no need to store the previous
      // results
      if constexpr(c_depth == 0)
        i_acc_buff[reuse][och*c_ow_ops+s_ow_ops] = s_acc;

      if ((ich == c_ich-1) | (c_depth == 1)) {
        s_output_struct[s_ow_ops].data[0][s_index_ops] = quant_stream<
          t_output, t_output_clip, t_output_mask, t_acc, c_relu
        >(s_acc);
        s_output_struct[s_ow_ops].last = last;
      }
      // return s_acc_struct;

    }

  }

}

////////////////////////////////////////////////////////////////////////////////
template <class t_input_struct, class t_input, class t_input_data, class t_input_st, class t_weight, 
          class t_weight_st, class t_bias,
          class t_add_struct, class t_add, class t_forward_struct, class t_input_mod, class t_input_1x1,
          class t_weight_1x1, class t_weight_1x1_st, class t_bias_1x1, class t_acc_struct, class t_acc,
          class t_acc_1x1_struct, class t_acc_1x1, 
          class t_output_struct, class t_output, class t_output_clip, class t_output_mask, 
          class t_output_struct_1x1, class t_output_1x1,
          int c_ich, int c_och, int c_och_1x1, int c_oh, int c_ow, int c_fh, int c_fw, int c_index, 
          int c_str, int c_ops, int c_in_ops, int c_add_ops, int c_ow_ops, int c_ow_ops_out,
          int c_relu, int c_reuse, int c_ow_pack, int c_och_pack, int c_in_bits, int c_in_ibits, 
          int c_w_bits, int c_w_ibits, int c_simd_bits, int c_simd, int c_mask,
          int c_in_bits_1x1, int c_in_ibits_1x1, int c_w_bits_1x1, int c_w_ibits_1x1,
          int c_simd_bits_1x1, int c_simd_1x1, int c_mask_1x1, int c_depth>
void conv_comp(hls::stream<t_input_struct> i_input[1],
               hls::stream<t_weight> i_weights[c_index],
               hls::stream<t_bias> i_bias[1],
               hls::stream<t_weight_1x1> i_weights_1x1[1],
               hls::stream<t_bias_1x1> i_bias_1x1[1],
               hls::stream<t_add_struct> i_add[c_ow_ops],
               hls::stream<t_forward_struct> o_forward[c_ow_ops],
               hls::stream<t_output_struct> o_output[c_ow_ops_out],
               hls::stream<t_output_struct_1x1> o_output_1x1[c_ow_ops_out]) {
  /* #pragma HLS inline */
  // Generic Convolution Computation

  // The output ow_ops must be greater or equal and a mutliples of ow_ops
  static_assert(c_ow_ops_out >= c_ow_ops, "c_ow_ops_out >= c_ow_ops");
  static_assert(c_ow_ops_out % c_ow_ops == 0, "c_ow_ops_out % c_ow_ops == 0");

  const auto c_och_depth = (c_depth == 1) ? 1 : c_och;
  const auto c_o_index = c_oh * c_ow / c_ow_ops_out;
  const auto c_reuse_iter = c_reuse / c_ow_ops;
  const auto c_num_och = c_och_depth / c_ops;
  const auto c_iter = c_reuse_iter * c_num_och;
  const auto c_och_1x1_depth = (c_depth == 1) ? 1 : c_och_1x1;
  const auto c_num_och_1x1 = c_och_1x1_depth / c_ops;
  const auto c_iter_1x1 = c_reuse_iter * c_num_och_1x1;
  const auto c_ops_1x1 = (c_och_1x1 < c_ops) ? c_och_1x1 : c_ops;

  // constexpr int FW = (c_fw);
  constexpr int FW = (c_fw+(c_ow_ops-1)*c_str);
  constexpr int MO = (c_fh*FW)/2;

  t_acc s_acc_buff[c_reuse_iter][c_och_depth*c_ow_ops];
#pragma HLS array_partition variable = s_acc_buff type = cyclic factor = c_ops*c_ow_ops dim = 2
  t_acc_1x1 s_acc_1x1_buff[c_reuse_iter][c_och_depth*c_ow_ops];
#pragma HLS array_partition variable = s_acc_1x1_buff type = cyclic factor = c_ops*c_ow_ops dim = 2
  t_input s_input;
// #pragma HLS array_partition variable = s_input type = complete dim = 0
#pragma HLS array_partition variable = s_input type = complete
// #pragma HLS array_partition variable = s_input type = complete dim=1
// #pragma HLS array_partition variable = s_input type = complete dim=2
  bool s_last = false;
  t_weight s_weight[c_index];
#pragma HLS array_partition variable = s_weight type = complete
  t_weight_1x1 s_weight_1x1[1];
#pragma HLS array_partition variable = s_weight_1x1 type = complete
  t_bias s_bias;
#pragma HLS array_partition variable = s_bias type = complete dim = 1
  t_bias_1x1 s_bias_1x1;
#pragma HLS array_partition variable = s_bias_1x1 type = complete dim = 1
  t_add_struct s_add[c_ow_ops];
#pragma HLS array_partition variable = s_add type = complete
// #pragma HLS aggregate variable = s_add
//   if (constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false)) {
// #pragma HLS array_partition variable = s_add[0].data type = complete dim=0
//   }
  t_output_struct s_output_struct[c_ow_ops];
#pragma HLS array_partition variable = s_output_struct type = complete
  t_output_struct_1x1 s_output_1x1_struct[c_ow_ops];
#pragma HLS array_partition variable = s_output_1x1_struct type = complete

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the first pipeline
  // round to the higher log2 c_simd
  const int c_pad_bits = 27-c_in_bits-1;
  const int c_int_pad_bits = c_simd_bits+c_in_ibits+c_w_ibits;
  const int c_pad_acc_bits = c_simd_bits+c_in_bits+c_w_bits;
  // mask on c_simd bits
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Constants for the second pipeline
  // mask on c_simd bits

  const int c_pad_bits_1x1 = 27-c_in_bits_1x1-1;
  const int c_int_pad_bits_1x1 = c_simd_bits_1x1+c_in_ibits_1x1+c_w_ibits_1x1;
  const int c_pad_acc_bits_1x1 = c_simd_bits_1x1+c_in_bits_1x1+c_w_bits_1x1;

  ////////////////////////////////////////////////////////////////////////////

  // TODO: the numbre of integer bits of the weights type must be considered
  typedef ap_fixed<c_pad_acc_bits, c_int_pad_bits, AP_RND_ZERO, AP_WRAP> t_acc_simd;
  typedef ap_fixed<c_pad_acc_bits_1x1, c_int_pad_bits_1x1, AP_RND_ZERO, AP_WRAP> t_acc_simd_1x1;

  auto s_ich_idx_add = 0;

  #ifndef __SYNTHESIS__
    if (c_depth == 1)
      std::cout << "depth_conv_op " << c_ich << " " << c_ops <<  " " << c_in_ops << std::endl;
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

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out; s_ow_ops_out+=c_reuse) {
      for (auto s_num_ich = 0; s_num_ich < c_ich; s_num_ich+=c_in_ops) {
        for (auto s_iter = 0; s_iter < c_iter; s_iter++) {
  #pragma HLS pipeline style = stp II=1
          auto s_reuse = s_iter % c_reuse_iter;
          auto s_num_och = s_iter / c_reuse_iter;

          if (s_num_och == 0) {
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
              s_weight[s_index] = i_weights[s_index].read();
            }

            // If it is the first reuse iteration, read the 1x1 weights
            if constexpr(std::is_same<t_weight_1x1, std::nullptr_t>::value == false) {
              if (s_iter < c_iter_1x1) s_weight_1x1[0] = i_weights_1x1[0].read();
            }

            if constexpr(std::is_same<t_bias, std::nullptr_t>::value == false) {
              if ((s_num_ich == 0) | (c_depth == 1)) s_bias = i_bias[0].read();
            }

            if constexpr(std::is_same<t_bias_1x1, std::nullptr_t>::value == false) {
              if (((s_num_ich == 0) | (c_depth == 1)) && (s_iter < c_iter_1x1)) s_bias_1x1 = i_bias_1x1[0].read();
            }
          }

          for (auto s_ich_idx = 0; s_ich_idx < c_in_ops; s_ich_idx++) {
            auto s_ich = s_num_ich + s_ich_idx;
            if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false)
              s_ich_idx_add = s_ich % c_add_ops;

            // Done to avoid partitioning of the stream and resource wasting
            // Theoretically with the add the input and output dimensions should
            // be equal, so we could perform the addition in different iterations
            if constexpr(std::is_same<t_add_struct, std::nullptr_t>::value == false){
              for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
                // FIX FOR MOBILENETv2: Taking into account different output and
                // input channels
                if ((s_iter == 0) && (s_ich_idx_add == 0)){
                  if constexpr(c_ich > c_och)
                    if (s_ich >= c_och)
                      continue;
                  s_add[s_ow_ops] = i_add[s_ow_ops].read();   
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
          #pragma HLS array_partition variable = s_input_1x1 type = complete
            for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
              // s_input_1x1[s_ow_ops] = s_input[MO + MO%c_str - s_ow_ops*c_str];
              auto forward_index = (c_fh/2 + 1)*FW - c_fw/2 - s_ow_ops*c_str;
              s_input_1x1[c_ow_ops - s_ow_ops - 1] = s_input[forward_index];
            }

            COMPUTE:
            for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops+=c_ow_pack) {
              for (auto s_ops = 0; s_ops < c_ops; s_ops+=c_och_pack) {
                auto s_och = s_num_och * c_ops + s_ops;

                conv_pipe<
                  t_input,
                  t_input_st,
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
                  c_fh,
                  c_fw,
                  c_index,
                  c_str,
                  c_ops,
                  c_in_ops,
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
                  s_och,
                  s_ich,
                  s_ich_idx,
                  s_ich_idx_add,
                  s_reuse,
                  s_ow_ops,
                  s_last,
                  s_add,
                  s_acc_buff,
                  s_output_struct
                );

                // TODO: split the loop in two parts controlled by different ops options

                if constexpr(std::is_same<t_acc_1x1_struct, std::nullptr_t>::value == false) {
                  if (s_iter < c_iter_1x1) {
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
                      t_output_struct_1x1,
                      t_output_1x1,
                      std::nullptr_t,
                      std::nullptr_t,
                      c_reuse_iter,
                      1,
                      1,
                      1,
                      1,
                      c_ops,
                      c_in_ops,
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
                      s_och,
                      s_ich,
                      s_ich_idx,
                      s_ich_idx_add,
                      s_reuse,
                      s_ow_ops,
                      s_last,
                      nullptr,
                      s_acc_1x1_buff,
                      s_output_1x1_struct
                    );
                  }
                }
              }
            }
          }
          if constexpr(std::is_same<t_forward_struct, std::nullptr_t>::value == false) {
            for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
              if (s_num_och == (c_num_och - 1)) {
                t_forward_struct s_forward;
                // auto forward_index = MO + MO%c_str - s_ow_ops*c_str;
                auto forward_index = (c_fh/2 + 1)*FW - c_fw/2 - s_ow_ops*c_str;
                s_forward.data[0] = s_input[forward_index];
                s_forward.last = false;
                o_forward[s_ow_ops_out+s_ow_ops].write(s_forward);
              }
            }
          }
          if ((s_num_ich == (c_ich-c_in_ops)) | (c_depth == 1)) {
            for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
              o_output[s_ow_ops_out+s_ow_ops].write(s_output_struct[s_ow_ops]);
              if constexpr(std::is_same<t_output_struct_1x1, std::nullptr_t>::value == false) {
                if (s_iter < c_iter_1x1) o_output_1x1[s_ow_ops_out+s_ow_ops].write(s_output_1x1_struct[s_ow_ops]);
              }
            }
          }
        }
      }
      #ifndef __SYNTHESIS__
        #ifdef DEBUG_RES
          for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
            for (auto s_och = 0; s_och < c_och_depth; s_och++) {
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
        #endif
      #endif
    }
  }
  #ifndef __SYNTHESIS__
    for (auto s_ow_ops_out = 0; s_ow_ops_out < c_ow_ops_out; s_ow_ops_out++) {
      std::cout << "o_output[" << s_ow_ops_out << "].size() = " << o_output[s_ow_ops_out].size() << std::endl;
    }
    if (c_depth == 1)
      std::cout << "end depth_conv_op " << c_ich << " " << c_depth << std::endl;
    else
      std::cout << "end conv_op " << c_ich << " " << c_depth << std::endl;
  #endif
}

//////////////////////////////////////////////////////////////////////////////

}  // namespace nn2fpga

#endif // NN2FPGA_PACKED_CONV_H_
