#ifndef NN2FPGA_POOL_STREAM_H_
#define NN2FPGA_POOL_STREAM_H_

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "nn2fpga/debug.h"
#include "nn2fpga/line_buffer_utils.h"

namespace nn2fpga {

template<class t_input_struct,
         class t_input,
         class t_output_struct,
         class t_output,
         class t_acc,
         class t_div,
         int c_ich,
         int c_och,
         int c_ih,
         int c_iw,
         int c_oh,
         int c_ow,
         int c_fh,
         int c_fw,
         int c_str,
         int c_pad,
         int c_pool,
         int c_ow_ops,
         int c_ops,
         int c_in_ops>
void
pool_op(hls::stream<t_input_struct> i_data[c_ow_ops],
        hls::stream<t_output_struct> o_data[1])
{

  static_assert(c_ops <= c_in_ops, "c_ops <= c_in_ops");
  static_assert(c_in_ops % c_ops == 0, "c_in_ops \% c_ops != 0");

  const int c_index = c_fh * c_fw;
  const int c_f_map = (c_ih * c_iw);
  const uint8_t c_average_scale = (uint8_t)(log2(c_fh * c_fw));
  const int c_quant = 0;

  // If the pool is adaptive then is more convenient to store all
  // och partial accumulations
  const int c_adaptive = (c_index) == (c_f_map);
  const int c_acc_och = (c_adaptive) ? c_och : 1;
  const int c_o_index = (c_adaptive) ? (c_oh * c_ow * c_index) / c_ow_ops : (c_oh * c_ow);
  const int c_fh_iter = (c_adaptive) ? 1 : c_fh;
  const int c_fw_iter = (c_adaptive) ? 1 : c_fw;
  const int c_ow_ops_iter = (c_adaptive) ? 1 : c_ow_ops;
  const int c_str_iter = (c_adaptive) ? 1 : c_str;

  bool s_last;
  t_acc s_acc_buff[c_acc_och];

  #ifndef __SYNTHESIS__
    std::cout << "pool_op " << c_ich << std::endl;
    std::cout << "c_ops = " << c_ops << std::endl;
    std::cout << "c_in_ops = " << c_in_ops << std::endl;
    std::cout << "c_adaptive = " << c_adaptive << std::endl;
    std::cout << "c_acc_och = " << c_acc_och << std::endl;
    std::cout << "c_o_index = " << c_o_index << std::endl;
    std::cout << "c_fh_iter = " << c_fh_iter << std::endl;
    std::cout << "c_fw_iter = " << c_fw_iter << std::endl;
    std::cout << "c_ow_ops_iter = " << c_ow_ops_iter << std::endl;
    std::cout << "c_str_iter = " << c_str_iter << std::endl;
    for (auto i = 0; i < c_ow_ops; i++) {
      std::cout << "i_data[" << i << "].size() = " << i_data[i].size() << std::endl;
    }
  #endif
  t_output_struct s_output_struct;

  t_input_struct s_input_struct;
  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_och = 0; s_och < c_och; s_och += c_in_ops) {
      for (auto s_in_ops = 0; s_in_ops < c_in_ops; s_in_ops += c_ops) {
#pragma HLS pipeline style = stp II=1
        for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++) {
          for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
            for (auto s_fh = 0; s_fh < c_fh_iter; s_fh++) {
              for (auto s_fw = 0; s_fw < c_fw_iter; s_fw++) {
                int s_index ;
                int s_acc_index;
                bool s_init;
                bool s_pool_write;
                #ifndef __SYNTHESIS__
                  #ifdef DEBUG_POOL
                    std::cout << "s_o_index = " << s_o_index << std::endl;
                    std::cout << "s_och = " << s_och << std::endl;
                    std::cout << "s_in_ops = " << s_in_ops << std::endl;
                    std::cout << "s_ow_ops = " << s_ow_ops << std::endl;
                    std::cout << "s_ops = " << s_ops << std::endl;
                    std::cout << "s_fh = " << s_fh << std::endl;
                    std::cout << "s_fw = " << s_fw << std::endl;
                  #endif
                #endif
                if constexpr(c_adaptive){
                  s_index = 0;
                  s_acc_index = s_och + s_in_ops + s_ops;
                  s_init = (s_o_index == 0) && (s_ow_ops == 0);
                } else {
                  s_index = s_fh*c_fw_iter+s_fw+(c_ow_ops_iter-s_ow_ops-1)*c_str_iter;
                  s_acc_index = 0;
                  s_init = (s_index == 0);
                }

                if (s_init) s_acc_buff[s_acc_index] = c_quant;

                if (((s_in_ops) == 0) && (s_ops == 0)) {
                  if constexpr(c_adaptive)
                    s_input_struct = i_data[s_ow_ops].read();
                  else{
                    if (s_index == 0)
                      s_input_struct = i_data[0].read();
                  }
                  s_last = s_input_struct.last;
                }
                // std::cout << "s_input_struct.data[" << s_o_index*c_ow_ops+s_ow_ops << "][s_ops] = " << s_input_struct.data[0][s_ops] << std::endl;

                if constexpr (c_pool == 0)  // Average Pool
                  s_acc_buff[s_acc_index] += s_input_struct.data[s_index][s_in_ops+s_ops];
                if constexpr (c_pool == 1) {  // Max Pool
                  if (s_input_struct.data[s_index][s_in_ops+s_ops] > s_acc_buff[s_acc_index]) s_acc_buff[s_acc_index] = s_input_struct.data[s_index][s_in_ops+s_ops];
                }

                if constexpr(c_adaptive){
                  s_pool_write = (s_o_index == (c_o_index - c_ow_ops)) && (s_ow_ops == (c_ow_ops-1));
                } else {
                  s_pool_write = (s_index == (c_index - 1));
                }

                if (s_pool_write) {
                  t_div s_acc = s_acc_buff[s_acc_index];
                  // std::cout << std::setprecision(8) << "[" << s_acc_index << "] " << s_acc << " / " << divisor << std::endl;
                  if constexpr (c_pool == 0) {  // Average Pool
                    // s_acc = s_acc >> c_average_scale;
                    t_div s_divisor = c_index;
                    s_acc = s_acc / s_divisor;
                  }
                  s_output_struct.data[0][s_ops] = t_output(s_acc);
                  if (s_ops == (c_ops - 1)) {
                    s_output_struct.last = s_last;
                    o_data[0].write(s_output_struct);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  #ifndef __SYNTHESIS__
    for (auto i = 0; i < c_ow_ops; i++) {
      if (i_data[i].size() > 0)
        std::cout << "ERROR: pool_op: i_data[" << i << "].size() " << i_data[i].size() << " > 0\n";
        assert (i_data[i].size() == 0);
      }
      if (o_data[0].size() == 0) {
        std::cout << "ERROR: pool_op: o_data[0].size() " << o_data[0].size() << " == 0\n";
        assert (o_data[0].size() > 0);
      }
      std::cout << "end pool_op " << c_ich << std::endl;
  #endif
}
}

#endif // NN2FPGA_POOL_STREAM_H_
