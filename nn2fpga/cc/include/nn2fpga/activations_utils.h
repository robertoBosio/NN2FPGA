#ifndef NN2FPGA_ACTIVATIONS_UTILS_H_
#define NN2FPGA_ACTIVATIONS_UTILS_H_

#include "ap_int.h"
#include "hls_stream.h"

namespace nn2fpga {

template <typename t_data, int OCH, int c_ops, int c_reuse>
void store_NCHW(hls::stream<t_data> &din, t_data o_data[c_reuse][OCH]) {
  constexpr int c_och_ops = OCH / c_ops;
  for (auto s_och = 0; s_och < c_och_ops; s_och++) {
    for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) {
      for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
#pragma HLS pipeline style = stp
        t_data s_data = din.read();
        o_data[s_reuse][s_och * c_ops + s_ops] = s_data;
      }
    }
  }
}

template <typename t_data, int OCH, int c_ops, int c_reuse>
void stream_NHWC(t_data din[c_reuse][OCH], hls::stream<t_data> &o_data) {
  for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) {
    for (auto s_och = 0; s_och < OCH; s_och++) {
#pragma HLS pipeline style = stp
      t_data s_data = din[s_reuse][s_och];
      o_data.write(s_data);
    }
  }
}

template <typename t_data, int ICH, int OCH, int OH, int OW, int c_index,
          int c_str, int c_ops, int c_reuse>
void rearrange_op(hls::stream<t_data> &din, hls::stream<t_data> &o_data) {
  /* #pragma HLS inline */

  /* Fix c_ops different than 1 case */
  constexpr int c_o_index = OH * OW / c_reuse;

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS dataflow

    t_data s_reuse_buffer[c_reuse][OCH];
#pragma HLS array_partition variable = s_reuse_buffer type = complete dim = 1
#pragma HLS stream variable = s_reuse_buffer type = shared

    store_NCHW<t_data, OCH, c_ops, c_reuse>(din, s_reuse_buffer);

    stream_NHWC<t_data, OCH, c_ops, c_reuse>(s_reuse_buffer, o_data);
  }
}

template <typename t_data, int ICH, int c_index, int c_reuse>
void store_NHWC(hls::stream<t_data> din[c_index],
                t_data o_data[c_reuse][ICH][c_index]) {
  for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) {
    for (auto s_ich = 0; s_ich < ICH; s_ich++) {
      for (auto s_index = 0; s_index < c_index; s_index++) {
#pragma HLS pipeline style = stp
        t_data s_data = din[s_index].read();
        o_data[s_reuse][s_ich][s_index] = s_data;
      }
    }
  }
}

template <typename t_data, int ICH, int c_index, int c_reuse>
void stream_NCHW(t_data din[c_reuse][ICH][c_index],
                 hls::stream<t_data> o_data[c_index]) {
  for (auto s_ich = 0; s_ich < ICH; s_ich++) {
    for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) {
      for (auto s_index = 0; s_index < c_index; s_index++) {
#pragma HLS pipeline style = stp
        t_data s_data = din[s_reuse][s_ich][s_index];
        o_data[s_index].write(s_data);
      }
    }
  }
}

template <typename t_data, int ICH, int OCH, int OH, int OW, int c_index,
          int c_str, int c_ops, int c_reuse>
void arrange_op(hls::stream<t_data> din[c_index],
                hls::stream<t_data> o_data[c_index]) {
  /* #pragma HLS inline */

  /* Fix c_ops different than 1 case */
  constexpr int c_o_index = OH * OW / c_reuse;

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS dataflow

    t_data s_reuse_buffer[c_reuse][ICH][c_index];
#pragma HLS stream variable = s_reuse_buffer type = shared
#pragma HLS array_partition variable = s_reuse_buffer type = complete dim = 3

    store_NHWC<t_data, ICH, c_index, c_reuse>(din, s_reuse_buffer);

    stream_NCHW<t_data, ICH, c_index, c_reuse>(s_reuse_buffer, o_data);
  }
}

template <typename t_input> 
t_input relu_op(t_input i_data) {
#pragma HLS inline
	if (i_data > 0)
		return i_data;
	else
		return 0;
}

template <typename t_input, int c_feature_map, int c_concat>
void concat_op(
  hls::stream<t_input> din[c_concat],
  const int c_ich[c_concat],
  hls::stream<t_input> &o_data
) {  

  for (auto s_feature_map = 0; s_feature_map < c_feature_map; s_feature_map++) {  
    for (auto s_concat = 0; s_concat < c_concat; s_concat++) {  
      for (auto s_ich = 0; s_ich < c_ich[s_concat]; s_ich++) {  
    #pragma HLS pipeline style = stp
        t_input s_data = din[s_concat].read();
        o_data.write(s_data);
      }
    }
  }
}

template <typename t_input, int c_ich, int c_ih, int c_iw, int c_upsample>
void upsample_op(
  hls::stream<t_input> &din,
  hls::stream<t_input> &o_data
) {

  auto upsample_buff = new t_input[c_ich][1][c_iw];

  for (auto s_ih = 0; s_ih < c_ih; s_ih++) {  
    for (auto s_upsample_h = 0; s_upsample_h < c_upsample; s_upsample_h++) {  
      for (auto s_iw = 0; s_iw < c_iw; s_iw++) {  
        for (auto s_upsample_w = 0; s_upsample_w < c_upsample; s_upsample_w++) {  
          for (auto s_ich = 0; s_ich < c_ich; s_ich++) {  
          #pragma HLS pipeline style = stp
            t_input s_data;
            if (s_upsample_w == 0 && s_upsample_h == 0) {
              s_data = din.read();
              o_data.write(s_data);
              upsample_buff[s_ich][0][s_iw] = s_data;
            } else {
              s_data = upsample_buff[s_ich][0][s_iw];
              o_data.write(s_data);
            }
          }
        }
      }
    }
  }
}


}  // namespace nn2fpga

#endif  // NN2FPGA_ACTIVATIONS_UTILS_H_
