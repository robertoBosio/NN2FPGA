#ifndef __UTILS__
#define __UTILS__

#include <etc/autopilot_ssdm_op.h>

#include "Debug.hpp"
#include "LineBuffer.hpp"
#include "ap_int.h"
#include "hls_math.h"

//////////////////////////// QUANT FUNCTIONS //////////////////////////////////

template <class t_input, int c_scale>
t_input QuantAct(t_input i_data) {
  const int c_scale_inv = -1 * c_scale;
  if (c_scale <= 0)
    return (i_data << c_scale_inv);
  else {
    t_input round = (i_data >> (c_scale - 1)) & 0x1;
    /* t_input round = 0; */
    return ((i_data >> c_scale) + round);
  }
}

template <class t_input, int c_scale, int c_mask>
t_input QuantAct(t_input i_data) {
#pragma HLS inline
	const int c_scale_inv = -1*c_scale;
	t_input s_data = i_data;

	const t_input c_msb    = sizeof(t_input)*8-1;
	const t_input c_mask_tmp = (1 << (c_mask+c_scale)) - 1;
	const t_input c_mask_pad = c_msb - c_mask_tmp; 

	if (c_scale <= 0)
		return (s_data << c_scale_inv);
	else {
    /* if (c_mask > 0) */
    /*   s_data = (t_input)(s_data & c_mask_pad); */
		t_input round = (s_data >> (c_scale-1)) & 0x1;
		/* return (t_input)(((s_data >> c_scale) + round) & c_mask); */
		return ((s_data >> c_scale) + round);
	}
}

template <class t_input, int c_scale, class t_output>
t_output QuantAct(t_input i_data) {
  const t_output c_msb = sizeof(t_output) * 8 - 1;

  const t_output c_max_0 = ~(t_output)(0);
  const t_output c_max_1 = c_max_0 ^ (-1 << c_msb);
  const t_output c_max = (c_max_0 < 0) ? c_max_1 : c_max_0;

  const t_output c_min_0 = ~(t_output)(0);
  const t_output c_min_1 = (-1 << c_msb);
  const t_output c_min = (c_min_0 < 0) ? c_min_1 : 0;

  t_input s_data = QuantAct<t_input, c_scale, c_max_0>(i_data);

  if (s_data > c_max) {
    return c_max;
  }
  if (s_data < c_min) {
    return c_min;
  }
  return (t_output)(s_data);
}

template <class t_input, int c_scale, int c_clip, class t_output>
t_output QuantAct(t_input i_data) {
  t_input s_data = QuantAct<t_input, c_scale>(i_data);
  const t_output c_msb = sizeof(t_output) * 8 - 1;

  const t_output c_max = c_clip;

  const t_output c_min_0 = ~(t_output)(0);
  const t_output c_min_1 = (-1 << c_msb);
  const t_output c_min = (c_min_0 < 0) ? c_min_1 : 0;

  if (s_data > c_max) {
    return c_max;
  }
  if (s_data < c_min) {
    return c_min;
  }
  return (t_output)(s_data);
}

template <
	class t_input,
	int c_scale,
	int c_clip,
	int c_mask,
	class t_output
> t_output QuantAct (
	t_input i_data
) {

	t_input s_data = QuantAct<t_input,c_scale,c_mask>(i_data);

	const t_output c_msb = sizeof(t_output)*8-1;
	const t_output c_max = c_clip;

	const t_output c_min_0 = ~(t_output)(0);
	/* const t_output c_min_1 = (-1 << c_msb); */
	const t_output c_min_1 = (-1*c_clip)-1;
	const t_output c_min   = (c_min_0 < 0) ? c_min_1 : 0;

  if (s_data > c_max) {
    return c_max;
  }
  if (s_data < c_min) {
    return c_min;
  }
  return (t_output)(s_data);
}

//////////////////////////// FROM POINTER TO STREAM ///////////////////////////
// For input activations
/* template < */
/* 	class t_input, */
/* 	class t_output, */
/* 	int c_ich, */
/* 	int c_iw, */
/* 	int c_ih */
/* > void ProduceStream( */
/* 	t_input *i_data, */
/* 	hls::stream<t_output> &s_i_data */
/* ) { */

/* 	const int c_index = c_ich*c_ih*c_iw; */

/* 	PRODSTR: for (int s_index = 0; s_index < c_index; s_index++) { */
/* 		s_i_data.write((t_output)(i_data[s_index])); */
/* 	} */

/* } */

template <class t_input, class t_output, int c_ich, int c_iw, int c_ih>
void ProduceStream(t_input i_data[c_ich * c_ih * c_iw],
                   hls::stream<ap_uint<1>> &i_last,
                   hls::stream<t_output> &o_data) {
  const int c_index = c_ich * c_ih * c_iw;

  /* while(1) { */
PRODSTR:
  for (int s_index = 0; s_index < c_index; s_index++) {
    o_data.write((t_output)(i_data[s_index]));
  }
  ap_uint<1> s_last = i_last.read();
  /* if (s_last) */
  /* 	break; */
  /* } */
}

template <class t_input, class t_input_part, class t_output_struct,
          class t_output, int c_ich, int c_iw, int c_ih, int c_bits,
          int c_scale>
void ProduceStream(hls::stream<t_input> &i_data,
                   hls::stream<t_output_struct> &o_data) {
  const int c_par = c_bits / 8;
  const int c_index = (c_ich * c_ih * c_iw);

  t_input tmp_r;
  ap_uint<c_bits> tmp_r_par;
PRODSTR:
  for (int s_index = 0; s_index < c_index; s_index++) {
#pragma HLS pipeline style = frp
    uint8_t s_par = s_index % c_par;

    if (s_par == 0) {
      tmp_r = i_data.read();
      tmp_r_par = tmp_r.data;
    }

    t_output_struct tmp_w;
    t_input_part tmp_p = (t_input_part)(tmp_r_par & 0xff);
    tmp_w.data = QuantAct<t_input_part, c_scale, t_output>(tmp_p);

    if (s_par < (c_par - 1))
      tmp_w.last = false;
    else
      tmp_w.last = tmp_r.last;
    o_data.write(tmp_w);
    tmp_r_par = tmp_r_par >> 8;
  }
}

template <class t_input, class t_output, int c_ich, int c_iw, int c_ih,
          int c_bits>
void ProduceStream(hls::stream<t_input> &i_data,
                   hls::stream<ap_uint<1>> &o_last,
                   hls::stream<t_output> &o_data) {
  const int c_par = c_bits / 8;
  const int c_index = (c_ich * c_ih * c_iw) / c_par;

  t_input tmp_r;
PRODSTR:
  for (int s_index = 0; s_index < c_index; s_index++) {
    tmp_r = i_data.read();
    ap_uint<64> tmp_r_par = tmp_r.data;

    for (uint8_t s_par = 0; s_par < c_par; s_par++) {
#pragma HLS pipeline off
      /* t_output tmp_w = (t_output)(tmp_r.data(8*(s_par+1)-1,8*s_par)); */
      t_output tmp_w = (t_output)(tmp_r_par & 0xff);
      o_data.write(tmp_w);
      tmp_r_par = tmp_r_par >> 8;
    }
  }

  o_last.write(tmp_r.last);
}

///////////////////////////// FROM STREAM TO POINTER //////////////////////////

// For output activations
template <class t_input, class t_output, int c_och, int c_ow, int c_oh>
void ConsumeStream(hls::stream<t_input> &i_data,
                   t_output o_data[c_och * c_ow * c_oh]) {
  t_input s_read;
  const int c_index = c_och * c_oh * c_ow;

  for (int s_index = 0; s_index < c_index; s_index++) {
    s_read = i_data.read();
    o_data[s_index] = (t_output)(s_read);
  }
}

template <class t_input, class t_output, int c_och, int c_ow, int c_oh>
void ConsumeStream(hls::stream<t_input> &i_data,
                   hls::stream<t_output> &o_data) {
  const int c_index = c_och * c_oh * c_ow;

  for (int s_index = 0; s_index < c_index; s_index++) {
    t_input s_read = i_data.read();
    t_output tmp;
    tmp.data = s_read.data;
    tmp.last = s_read.last;
    tmp.keep = -1;
    o_data.write(tmp);
  }
}

template <class t_input, class t_output, int c_och, int c_ow, int c_oh>
void ConsumeStream(hls::stream<t_input> &i_data,
                   hls::stream<ap_uint<1>> &i_last,
                   hls::stream<t_output> &o_data) {
  const int c_index = c_och * c_oh * c_ow;

  for (int s_index = 0; s_index < c_index - 1; s_index++) {
    t_input s_read = i_data.read();
    t_output tmp;
    tmp.data = s_read;
    tmp.last = false;
    tmp.keep = -1;
    o_data.write(tmp);
  }

  t_output tmp;
  t_input s_read = i_data.read();
  tmp.data = s_read;
  tmp.last = i_last.read();
  tmp.keep = -1;
  o_data.write(tmp);
}

template <class t_input, class t_output, int c_och, int c_ow, int c_oh,
          int c_bits>
void ConsumeStream(hls::stream<t_input> &i_data,
                   hls::stream<ap_uint<1>> &i_last,
                   hls::stream<t_output> &o_data) {
  const int c_par = c_bits / 8;
  const int c_index = c_och * c_oh * c_ow / c_par;
  const int c_out_pad = (c_och * c_oh * c_ow) % c_par;

  for (int s_index = 0; s_index < c_index; s_index++) {
    t_output tmp;
    for (int s_par = 0; s_par < c_par; s_par++) {
#pragma HLS pipeline
      t_input s_read = i_data.read();
      tmp.data((s_par + 1) * 8 - 1, s_par * 8) = s_read;
    }
    tmp.keep = -1;
    tmp.last = false;
    o_data.write(tmp);
  }

  t_output tmp;
  tmp.data = 0;
  for (int s_out_pad = 0; s_out_pad < c_out_pad; s_out_pad++) {
#pragma HLS pipeline
    t_input s_read = i_data.read();
    tmp.data((s_out_pad + 1) * 8 - 1, s_out_pad * 8) = s_read;
  }
  /* The input last stream doesn't count the number of activations streamed but
   * the */
  /* number of batches analyzed */
  tmp.keep = -1;
  tmp.last = i_last.read();
  o_data.write(tmp);
}

///////////////////////////// FOR INPUT WEIGHTS //////////////////////////
// For input weights
template <class t_input, class t_output, int c_ich, int c_och, int c_iw,
          int c_ih, int c_ow, int c_oh, int c_str>
void ProduceStream(t_input *i_data, hls::stream<t_output> &s_i_data) {
  const int c_index = c_och * c_ich * c_ih * c_iw;

  for (int s_oh = 0; s_oh < c_oh; s_oh++) {
    for (int s_ow = 0; s_ow < c_ow; s_ow++) {
    PRODSTR:
      for (int s_index = 0; s_index < c_index; s_index++) {
        s_i_data.write((t_output)(i_data[s_index]));
      }
    }
  }
}
//
// For input weights
template <class t_input, class t_output, int c_ich, int c_och, int c_iw,
          int c_ih, int c_ow, int c_oh>
void ProduceStream(const t_input i_data[c_och * c_ich * c_iw * c_ih],
                   hls::stream<t_output> o_data[c_ih * c_iw]) {
  const int c_index = c_oh * c_ow;
  const int c_stream_sel = c_ih * c_iw;
  const int c_ch = c_ich * c_och;
#pragma HLS array_partition type = cyclic factor = c_stream_sel variable = \
    i_data

  for (uint16_t s_index = 0; s_index < c_index; s_index++) {
    uint16_t s_addr = 0;
    for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline
      for (uint8_t s_stream_sel = 0; s_stream_sel < c_stream_sel;
           s_stream_sel++) {
        o_data[s_stream_sel].write((t_output)(i_data[s_addr]));
        s_addr++;
      }
    }
  }
}

template <class t_input, class t_output, int c_ich, int c_och, int c_iw,
          int c_ih, int c_ow, int c_oh, int c_ops>
void ProduceStream(const t_input i_data[c_och * c_ich * c_iw * c_ih],
                   hls::stream<t_output> o_data[c_ih * c_iw]) {
  const int c_index = c_oh * c_ow;
  const int c_stream_sel = c_ih * c_iw;
  const int c_ch = c_ich * c_och / c_ops;
#pragma HLS array_partition type = cyclic factor = c_stream_sel variable = \
    i_data

  for (uint16_t s_index = 0; s_index < c_index; s_index++) {
    uint16_t s_addr = 0;
    for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
      for (uint8_t s_stream_sel = 0; s_stream_sel < c_stream_sel;
           s_stream_sel++) {
#pragma HLS pipeline
        for (uint16_t s_ops = 0; s_ops < c_ops; s_ops++) {
          o_data[s_stream_sel].write((t_output)(i_data[s_addr]));
          s_addr++;
        }
      }
    }
  }
}

template <class t_input, class t_output, int c_ich, int c_och, int c_ow,
          int c_oh>
void ProduceStream(const t_input i_data[c_och * c_ich],
                   hls::stream<t_output> &o_data) {
  const int c_index = c_oh * c_ow;
  const int c_ch = c_ich * c_och;

  for (uint16_t s_index = 0; s_index < c_index; s_index++) {
    for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline
      o_data.write((t_output)(i_data[s_ch]));
    }
  }
}

template <class t_input, class t_output, int c_ich, int c_och, int c_ow,
          int c_oh, int c_ops>
void ProduceStream(const t_input i_data[c_och * c_ich],
                   hls::stream<t_output> &o_data) {
  const int c_index = c_oh * c_ow;
  const int c_ch = c_ich * c_och / c_ops;
  const uint8_t c_log_ops = (uint8_t)(log2(c_ops));

  for (uint16_t s_index = 0; s_index < c_index; s_index++) {
    for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline
      t_output s_data = 0;
      for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
        s_data(8 * (s_ops + 1) - 1, 8 * s_ops) =
            i_data[(s_ch << c_log_ops) + s_ops];
      }
      o_data.write((t_output)(s_data));
    }
  }
}

template <class t_input, class t_output, int c_ich, int c_och, int c_ow,
          int c_oh, int c_fw, int c_fh, int c_ops, int c_reuse>
void ProduceStream(
    const t_input i_data[c_fh * c_fw][c_och * c_ich / c_ops][c_ops],
    hls::stream<t_output> o_data[c_fh * c_fw]) {
  const int c_index = c_fh * c_fw;
  const int c_ch = c_ich * c_och / c_ops;
  const int c_o_index = c_oh * c_ow * c_ch / c_reuse;

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS pipeline
    auto s_ch = s_o_index % c_ch;
    for (auto s_index = 0; s_index < c_index; s_index++) {
      t_output s_output;
      for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
        s_output[s_ops] = i_data[s_index][s_ch][s_ops];
      }
      o_data[s_index].write(s_output);
    }
  }
}

template <class t_input, class t_output, int c_ich, int c_och, int c_ow,
          int c_oh, int c_fw, int c_fh, int c_ops>
void ProduceStream(hls::stream<t_input> i_data[c_fh * c_fw],
                   hls::stream<t_output> o_data[c_fh * c_fw]) {
  const int c_index = c_fh * c_fw;
  const int c_ch = c_ich * c_och / c_ops;
  const int c_iter = 64 / c_ops;
  const int c_o_index = c_oh * c_ow * c_ch;
  t_input s_data[c_index];
  for (uint32_t s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS pipeline style = frp
    uint8_t s_iter = s_o_index % c_iter;
    if (s_iter == 0) {
      for (uint8_t s_index = 0; s_index < c_index; s_index++)
        s_data[s_index] = i_data[s_index].read();
    }

    for (uint8_t s_index = 0; s_index < c_index; s_index++) {
      t_output s_output;
      for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
        uint8_t s_part = s_iter * c_ops + c_ops;
        s_output[s_ops] = s_data[s_index](8 * (s_part + 1) - 1, 8 * s_part);
      }
      o_data[s_index].write(s_output);
    }
  }
}

///////////////////////////// FROM STREAM TO STREAM ///////////////////////////

template <class t_input, class t_output, int c_ich, int c_och, int c_iw,
          int c_ih, int c_ow, int c_oh, int c_pad>
void PadStream(hls::stream<t_input> &i_data, hls::stream<t_output> &o_data) {
  if (c_pad == 0) {
    for (uint8_t s_ih = 0; s_ih < c_ih; s_ih++) {
      for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
        for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
          t_input s_input = i_data.read();
          t_output s_output;
          s_output.data = s_input.data;
          s_output.last = s_input.last;
          o_data.write(s_output);
        }
      }
    }
  }

  if (c_pad == 1) {
    for (uint8_t s_ih = 0; s_ih < c_ih; s_ih++) {
      for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
        for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
          t_input s_input = i_data.read();
          t_output s_output;
          s_output.data = s_input.data;
          s_output.last = s_input.last;
          o_data.write(s_output);
        }
      }
    }
  }
}

//////////////////////////// BLOCK INTERFACES /////////////////////////////////

template <class t_input, int c_ich, int c_iw, int c_ih, int c_fw, int c_fh,
          int c_pad>
void PadInput(hls::stream<t_input> &i_data, hls::stream<t_input> &o_data) {
  /* #pragma HLS inline */

  /* This handles padding aware inputs */

  const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  const int c_pad_index_w = c_pad * (c_fw - 1) / 2;

  const int c_ih_pad = c_ih + c_pad_index_h * 2;
  const int c_iw_pad = c_iw + c_pad_index_w * 2;

  const t_input s_zero_false = {0, false};
  bool s_last = false;
  t_input s_zero = s_zero_false;

  /* OPTIMIZATION */
  const int c_i_index = c_ih_pad * c_iw_pad * c_ich;

  for (uint32_t s_ih = 0; s_ih < c_ih_pad; s_ih++) {
    for (uint32_t s_iw = 0; s_iw < c_iw_pad; s_iw++) {
      for (uint32_t s_ich = 0; s_ich < c_ich; s_ich++) {
#pragma HLS pipeline style = flp

        bool s_data_read = true;

        s_data_read &= (s_ih >= c_pad_index_h);
        s_data_read &= (s_ih < (c_ih_pad - c_pad_index_h));
        s_data_read &= (s_iw >= c_pad_index_w);
        s_data_read &= (s_iw < (c_iw_pad - c_pad_index_w));

        t_input s_input = s_zero;
        if (s_data_read) s_input = i_data.read();

        s_zero.last = s_input.last;

        o_data.write(s_input);
      }
    }
  }
}

template <class t_input, int c_ich, int c_iw, int c_ih, int c_fw, int c_fh,
          int c_pad>
void ForwardStream(hls::stream<t_input> &i_data) {
  const int c_pad_index_h = c_pad * (c_fh - 1);
  const int c_pad_index_w = c_pad * (c_fw - 1);
  const int c_ih_end = c_ih + c_pad_index_h;
  const int c_iw_end = c_iw + c_pad_index_w;

  for (uint8_t s_ih = 0; s_ih < c_ih_end; s_ih++) {
    for (uint8_t s_iw = 0; s_iw < c_iw_end; s_iw++) {
      for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
        t_input s_tmp = i_data.read();
      }
    }
  }
}

template <class t_input, int c_ich, int c_iw, int c_ih, int c_fw, int c_fh,
          int c_pad>
void ForwardStream(hls::stream<t_input> &i_data,
                   hls::stream<t_input> &o_forward) {
  const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
  const int c_ih_start = -1 * c_pad_index_h;
  const int c_iw_start = -1 * c_pad_index_w;
  const int c_ih_end = c_ih + c_pad_index_h;
  const int c_iw_end = c_iw + c_pad_index_w;

  for (int8_t s_ih = c_ih_start; s_ih < c_ih_end; s_ih++) {
    for (int8_t s_iw = c_iw_start; s_iw < c_iw_end; s_iw++) {
      for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
        t_input s_tmp = i_data.read();
        if ((s_ih > -1) & (s_iw > -1) & (s_ih < c_ih) & (s_iw < c_iw))
          o_forward.write(s_tmp);
      }
    }
  }
}

template <int c_split>
void SplitStream(hls::stream<ap_uint<1>> &i_data,
                 hls::stream<ap_uint<1>> o_data[c_split]) {
  ap_uint<1> s_data = i_data.read();
  for (uint8_t s_split = 0; s_split < c_split; s_split++) {
#pragma HLS unroll
    o_data[s_split].write(s_data);
  }
}

template <class t_output, int c_och, int c_ow, int c_oh, int c_split>
void SplitStream(hls::stream<t_output> &i_data,
                 hls::stream<t_output> o_data[c_split]) {
  for (uint8_t s_oh = 0; s_oh < c_oh; s_oh++) {
    for (uint8_t s_ow = 0; s_ow < c_ow; s_ow++) {
      for (uint8_t s_och = 0; s_och < c_och; s_och++) {
        t_output s_out = i_data.read();
        for (uint8_t s_split = 0; s_split < c_split; s_split++) {
#pragma HLS unroll
          o_data[s_split].write(s_out);
        }
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
/* Rearranging activations for weights reusage generation */

template <class t_data, int c_och, int c_ops, int c_reuse>
void StoreNCHW(hls::stream<t_data> &i_data, t_data o_data[c_reuse][c_och]) {
  const int c_och_ops = c_och / c_ops;
  for (auto s_och = 0; s_och < c_och_ops; s_och++) {
    for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) {
      for (auto s_ops = 0; s_ops < c_ops; s_ops++) {
#pragma HLS pipeline style = frp
        t_data s_data = i_data.read();
        o_data[s_reuse][s_och * c_ops + s_ops] = s_data;
      }
    }
  }
}

template <class t_data, int c_och, int c_ops, int c_reuse>
void StreamNHWC(t_data i_data[c_reuse][c_och], hls::stream<t_data> &o_data) {
  for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) {
    for (auto s_och = 0; s_och < c_och; s_och++) {
#pragma HLS pipeline style = frp
      t_data s_data = i_data[s_reuse][s_och];
      o_data.write(s_data);
    }
  }
}

template <class t_data, int c_ich, int c_och, int c_oh, int c_ow, int c_index,
          int c_str, int c_ops, int c_reuse>
void RearrangeOp(hls::stream<t_data> &i_data, hls::stream<t_data> &o_data) {
  /* #pragma HLS inline */

  /* Fix c_ops different than 1 case */
  const int c_o_index = c_oh * c_ow / c_reuse;

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS dataflow

    t_data s_reuse_buffer[c_reuse][c_och];
#pragma HLS array_partition variable = s_reuse_buffer type = complete dim = 1
#pragma HLS stream variable = s_reuse_buffer type = shared

    StoreNCHW<t_data, c_och, c_ops, c_reuse>(i_data, s_reuse_buffer);

    StreamNHWC<t_data, c_och, c_ops, c_reuse>(s_reuse_buffer, o_data);
  }
}

template <class t_data, int c_ich, int c_index, int c_reuse>
void StoreNHWC(hls::stream<t_data> i_data[c_index],
               t_data o_data[c_reuse][c_ich][c_index]) {
  for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) {
    for (auto s_ich = 0; s_ich < c_ich; s_ich++) {
      for (auto s_index = 0; s_index < c_index; s_index++) {
#pragma HLS pipeline style = frp
        t_data s_data = i_data[s_index].read();
        o_data[s_reuse][s_ich][s_index] = s_data;
      }
    }
  }
}

template <class t_data, int c_ich, int c_index, int c_reuse>
void StreamNCHW(t_data i_data[c_reuse][c_ich][c_index],
                hls::stream<t_data> o_data[c_index]) {
  for (auto s_ich = 0; s_ich < c_ich; s_ich++) {
    for (auto s_reuse = 0; s_reuse < c_reuse; s_reuse++) {
      for (auto s_index = 0; s_index < c_index; s_index++) {
#pragma HLS pipeline style = frp
        t_data s_data = i_data[s_reuse][s_ich][s_index];
        o_data[s_index].write(s_data);
      }
    }
  }
}

template <class t_data, int c_ich, int c_och, int c_oh, int c_ow, int c_index,
          int c_str, int c_ops, int c_reuse>
void ArrangeOp(hls::stream<t_data> i_data[c_index],
               hls::stream<t_data> o_data[c_index]) {
  /* #pragma HLS inline */

  /* Fix c_ops different than 1 case */
  const int c_o_index = c_oh * c_ow / c_reuse;

  for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
#pragma HLS dataflow

    t_data s_reuse_buffer[c_reuse][c_ich][c_index];
#pragma HLS stream variable = s_reuse_buffer type = shared
#pragma HLS array_partition variable = s_reuse_buffer type = complete dim = 3

    StoreNHWC<t_data, c_ich, c_index, c_reuse>(i_data, s_reuse_buffer);

    StreamNCHW<t_data, c_ich, c_index, c_reuse>(s_reuse_buffer, o_data);
  }
}

//////////////////////////////////////////////////////////////////////////////
/* Line Buffers generation */
template <class t_input, int c_ich, int c_och, int c_ih, int c_iw, int c_oh,
          int c_ow, int c_fh, int c_fw, int c_str, int c_pad, int c_pos_h,
          int c_pos_w>
void ShiftOp(hls::stream<t_input> &i_data, hls::stream<t_input> &o_compute) {
  /* #pragma HLS inline */

  const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
  const int c_ih_pad = c_ih + c_pad_index_h * 2;
  const int c_iw_pad = c_iw + c_pad_index_w * 2;
  const int c_paddingh_shift = c_pos_h;
  const int c_paddingw_shift = c_pos_w;
  const int c_strideh_shift = (c_str - 1);
  const int c_stridew_shift = (c_str - 1);
  const int c_end_paddingh_shift = (c_fh - 1 - c_pos_h);
  const int c_end_paddingw_shift = (c_fw - 1 - c_pos_w);

  /* Constants for new version */
  const int c_i_index = c_ih_pad * c_iw_pad * c_ich;

  const int c_starth = c_pad_index_h;
  const int c_startw = c_pad_index_w;
  const int c_endh = c_ih_pad - c_pad_index_h;
  const int c_endw = c_iw_pad - c_pad_index_w;

  for (auto s_index_h = c_starth; s_index_h < c_endh; s_index_h++) {
    for (auto s_index_w = c_startw; s_index_w < c_endw; s_index_w++) {
      for (auto s_index_ich = 0; s_index_ich < c_ich; s_index_ich++) {
#pragma HLS pipeline style = frp
        bool s_compute_write = true;
        uint16_t s_index_h_str = s_index_h % c_str;
        uint16_t s_index_w_str = s_index_w % c_str;

        s_compute_write &= (s_index_h >= c_paddingh_shift);
        s_compute_write &= (s_index_h < (c_ih_pad - c_end_paddingh_shift));
        s_compute_write &= (s_index_w >= c_paddingw_shift);
        s_compute_write &= (s_index_w < (c_iw_pad - c_end_paddingw_shift));
        s_compute_write &= (s_index_h_str == (c_paddingh_shift % c_str));
        s_compute_write &= (s_index_w_str == (c_paddingw_shift % c_str));

        t_input s_input = i_data.read();
        if (s_compute_write) o_compute.write(s_input);
      }
    }
  }
}

template <class t_input, int c_ich, int c_iw, int c_ih, int c_fw, int c_fh,
          int c_str, int c_pad>
void PadInput(hls::stream<t_input> i_data[c_fw * c_fh],
              hls::stream<t_input> o_data[c_fw * c_fh]) {
  /* #pragma HLS inline */

  /* This handles padding aware inputs */

  const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
  const int c_ih_pad = c_ih + c_pad_index_h * 2;
  const int c_iw_pad = c_iw + c_pad_index_w * 2;
  const int c_index = c_fh * c_fw;

  bool s_last;
  for (auto s_index_h = 0; s_index_h < c_ih; s_index_h += c_str) {
    for (auto s_index_w = 0; s_index_w < c_iw; s_index_w += c_str) {
      for (auto s_index_ich = 0; s_index_ich < c_ich; s_index_ich++) {
#pragma HLS pipeline style = frp
        for (auto s_fh = 0; s_fh < c_fh; s_fh++) {
          for (auto s_fw = 0; s_fw < c_fw; s_fw++) {
            t_input s_write;

            auto s_index = s_fh * c_fw + s_fw;

            bool s_data_read = true;

            s_data_read &= (s_index_h >= (c_pad_index_h - s_fh));
            s_data_read &= (s_index_h < (c_ih + c_pad_index_h - s_fh));
            s_data_read &= (s_index_w >= (c_pad_index_w - s_fw));
            s_data_read &= (s_index_w < (c_iw + c_pad_index_w - s_fw));

            if (s_data_read) {
              s_write = i_data[c_index - s_index - 1].read();
              if (s_index == c_index - 1) s_last = s_write.last;
            } else {
              s_write.data = 0;
              s_write.last = s_last;
            }
            o_data[c_index - s_index - 1].write(s_write);
          }
        }
      }
    }
  }
}

template <class t_input, int c_ich, int c_och, int c_ih, int c_iw, int c_oh,
          int c_ow, int c_fh, int c_fw, int c_str, int c_pad, int c_pos_h,
          int c_pos_w>
void ShiftOp(hls::stream<t_input> &i_data, hls::stream<t_input> &o_compute,
             hls::stream<t_input> &o_data) {
  /* #pragma HLS inline */

  const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
  const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
  const int c_ih_pad = c_ih + c_pad_index_h * 2;
  const int c_iw_pad = c_iw + c_pad_index_w * 2;
  const int c_paddingh_shift = c_pos_h;
  const int c_paddingw_shift = c_pos_w;
  const int c_strideh_shift = (c_str - 1);
  const int c_stridew_shift = (c_str - 1);
  const int c_end_paddingh_shift = (c_fh - 1 - c_pos_h);
  const int c_end_paddingw_shift = (c_fw - 1 - c_pos_w);

  /* Constants for new version */
  const int c_i_index = c_ih_pad * c_iw_pad * c_ich;

  const int c_starth = c_pad_index_h;
  const int c_startw = c_pad_index_w;
  const int c_endh = c_ih_pad - c_pad_index_h;
  const int c_endw = c_iw_pad - c_pad_index_w;

  hls::stream<t_input> s_compute;
#pragma HLS stream variable = s_compute depth = 2 type = fifo

  for (auto s_index_h = c_starth; s_index_h < c_endh; s_index_h++) {
    for (auto s_index_w = c_startw; s_index_w < c_endw; s_index_w++) {
      for (auto s_index_ich = 0; s_index_ich < c_ich; s_index_ich++) {
#pragma HLS pipeline style = frp
        bool s_compute_write = true;
        uint16_t s_index_h_str = s_index_h % c_str;
        uint16_t s_index_w_str = s_index_w % c_str;

        s_compute_write &= (s_index_h >= c_paddingh_shift);
        s_compute_write &= (s_index_h < (c_ih_pad - c_end_paddingh_shift));
        s_compute_write &= (s_index_w >= c_paddingw_shift);
        s_compute_write &= (s_index_w < (c_iw_pad - c_end_paddingw_shift));
        s_compute_write &= (s_index_h_str == (c_paddingh_shift % c_str));
        s_compute_write &= (s_index_w_str == (c_paddingw_shift % c_str));

        t_input s_input = i_data.read();
        if (s_compute_write) o_compute.write(s_input);
        o_data.write(s_input);
      }
    }
  }
}
/* template < */
/* 	class t_input, */
/* 	int c_ich, */
/* 	int c_och, */
/* 	int c_ih, */
/* 	int c_iw, */
/* 	int c_oh, */
/* 	int c_ow, */
/* 	int c_fh, */
/* 	int c_fw, */
/* 	int c_str, */
/* 	int c_pad */
/* > void ShiftOp( */
/* 	hls::stream<t_input> &i_data, */
/* 	hls::stream<t_input> o_compute[c_fh*c_fw] */
/* ) { */
/* /1* #pragma HLS inline *1/ */

/* 	const auto c_starth = (c_fh-1)*(1-c_pad); */
/* 	const auto c_startw = (c_fw-1)*(1-c_pad); */
/* 	const auto c_pad_index_h = c_pad * (c_fh - 1) / 2; */
/* 	const auto c_pad_index_w = c_pad * (c_fw - 1) / 2; */
/* 	const auto c_ih_pad = c_ih + c_pad_index_h*2; */
/* 	const auto c_iw_pad = c_iw + c_pad_index_w*2; */
/* 	const auto c_strideh_shift = (c_str-1); */
/* 	const auto c_stridew_shift = (c_str-1); */

/* 	/1* Constants for new version *1/ */
/* 	const auto c_i_index = c_ih_pad*c_iw_pad*c_ich; */
/* 	const auto c_index = c_fh*c_fw; */

/*   const auto c_size = (c_fh-1*c_iw+fw-1)*c_ich; */
/* 	t_input s_data[c_size]; */
/*   const auto s_start = -1*c_size; */
/*   auto s_address = s_start; */

/* 	for (auto s_index_h = 0; s_index_h < c_ih; s_index_h++) { */
/* 		for (auto s_index_w = 0; s_index_w < c_iw; s_index_w++) { */
/* 			for (auto s_index_ich = 0; s_index_ich < c_ich; s_index_ich++) { */
/* #pragma HLS pipeline style=frp */
/* 				for (auto s_fh=0; s_fh<c_fh; s_fh++) { */
/*           auto s_addr_h = s_address+s_fh*c_iw*c_ich */
/* 					for (auto s_fw=0; s_fw<c_fw; s_fw++) { */
/* 						auto s_addr_w = (s_addr_h+s_fw*c_ich) % c_size; */
/* 						t_input s_input; */
/*             auto s_index = s_fh*c_fw+s_fw; */
/*             s_input = s_data[s_index] */
/* 						if (s_addr_w > 0) */
/* 							o_compute[s_index] = i_data.read(); */

/* 					} */
/* 				} */
/*         s_address++; */
/* 			} */
/* 		} */
/* 	} */

/* } */

#endif
