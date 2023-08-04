#ifndef NN2FPGA_POOL_STREAM_H_
#define NN2FPGA_POOL_STREAM_H_

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "nn2fpga/debug.h"
#include "nn2fpga/line_buffer_utils.h"

namespace nn2fpga {

template <class t_input_struct, class t_input, class t_output_struct,
          class t_output, class t_acc, int c_ich, int c_och, int c_ih, int c_iw,
          int c_oh, int c_ow, int c_fh, int c_fw, int c_str, int c_pad,
          int c_pool, int c_ws, int c_ws_out, int c_ops, int c_in_ops>
void pool_op(hls::stream<t_input_struct> i_data[c_ws],
             hls::stream<t_output_struct> o_data[1]) {
  const int c_index = c_fh * c_fw;
  const int c_o_index = (c_oh * c_ow * c_index) /c_ws;
  const uint8_t c_average_scale = (uint8_t)(log2(c_fh * c_fw));
  const int c_quant = 0;

  bool s_last;
  t_acc s_acc_buff[c_och];

  hls::stream<t_acc> s_acc_stream;
#pragma HLS stream variable = s_acc_stream depth = 2 type = fifo
  t_output_struct s_output_struct;

  t_input_struct s_input_struct;
  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_och = 0; s_och < c_och; s_och+=c_ops) {
  #pragma HLS pipeline style = stp
      for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
        for (auto s_ops = 0; s_ops < c_in_ops; s_ops++) {
          if ((s_o_index == 0) && (s_ws == 0)) s_acc_buff[s_och+s_ops] = c_quant;

          if (s_ops == 0) {
            s_input_struct = i_data[s_ws].read();
            s_last = s_input_struct.last;
          }
          // std::cout << "s_input_struct.data[" << s_o_index*c_ws+s_ws << "][s_ops] = " << s_input_struct.data[0][s_ops] << std::endl;

          if (c_pool == 0)  // Average Pool
            s_acc_buff[s_och+s_ops] += s_input_struct.data[0][s_ops];
          if (c_pool == 1) {  // Max Poool
            if (s_input_struct.data[0][s_ops] > s_acc_buff[s_och+s_ops]) s_acc_buff[s_och+s_ops] = s_input_struct.data[0][s_ops];
          }
          bool s_pool_write = (s_o_index == (c_o_index - c_ws)) && (s_ws == (c_ws-1));
          if (s_pool_write) {
            t_acc s_acc = s_acc_buff[s_och+s_ops];
            if (c_pool == 0)  // Average Pool
              s_acc = s_acc >> c_average_scale;
            s_output_struct.data[0][s_ops] = t_output(s_acc);
            if (s_ops == (c_in_ops - 1)) {
              s_output_struct.last = s_last;
              o_data[0].write(s_output_struct);
            }
          }
        }
      }
    }
  }
}

template <class t_input_struct, class t_input, class t_output_struct,
          class t_output, class t_acc, int c_ich, int c_och, int c_ih, int c_iw,
          int c_oh, int c_ow, int c_fh, int c_fw, int c_str, int c_pad,
          int c_pool, int c_ws, int c_ws_out, int c_ops>
void pool_op(hls::stream<t_input_struct> i_data[c_fh*(c_fw+(c_ws-1)*c_str)],
             hls::stream<t_output_struct> o_data[c_ws]) {
  const int c_index = c_fh * c_fw;
  const int c_o_index = c_oh * c_ow;
  const uint8_t c_average_scale = (uint8_t)(log2(c_fh * c_fw));
  const int c_quant = 0;

  bool s_last;
  t_acc s_acc_buff;

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
    for (auto s_ws = 0; s_ws < c_ws; s_ws++) {
      #pragma HLS unroll
      for (auto s_och = 0; s_och < c_och; s_och++) {
        #pragma HLS pipeline style = stp
        for (auto s_index = 0; s_index < c_index; s_index++) {

          t_input_struct s_input_struct = i_data[s_index].read();
          t_input s_input = s_input_struct.data;
          s_last = s_input_struct.last;

          if constexpr(c_pool == 0) {  // Average Pool
            if (s_index == 0) 
              s_acc_buff = c_quant;
            s_acc_buff += s_input;
          }
          if constexpr(c_pool == 1) {  // Max Poool
            if ((s_index == 0) | (s_input > s_acc_buff))
              s_acc_buff = s_input;
          }
          if (s_index == (c_index - 1)) {
            t_output_struct s_output_struct;
            t_acc s_acc = s_acc_buff;
            if constexpr(c_pool == 0)  // Average Pool
              s_acc = s_acc >> c_average_scale;
            s_output_struct.data = t_output(s_acc);
            s_output_struct.last = s_last;
            o_data[s_ws].write(s_output_struct);
          }
        }
      }
    }
  }
}

}

#endif // NN2FPGA_POOL_STREAM_H_
