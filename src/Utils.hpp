#ifndef __UTILS__
#define __UTILS__

#include "hls_math.h"
#include "Debug.hpp"
#include "LineBuffer.hpp"

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

template <
	class t_input,
	class t_output,
	int c_ich,
	int c_iw,
	int c_ih
> void ProduceStream(
	t_input i_data[c_ich*c_ih*c_iw],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_ich*c_ih*c_iw;

	/* while(1) { */
		PRODSTR: for (int s_index = 0; s_index < c_index; s_index++) {
			o_data.write((t_output)(i_data[s_index]));
		}
		ap_uint<1> s_last = i_last.read();
		/* if (s_last) */
		/* 	break; */
	/* } */

}

template <
	class t_input,
	class t_output,
	int c_ich,
	int c_iw,
	int c_ih,
	int c_bits
> void ProduceStream(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

	const int c_par = c_i_data/8;
	const int c_index = (c_ich*c_ih*c_iw)/c_par;

#ifndef __SYNTHESIS__

	while(i_data.empty());

#endif

	/* while(1) { */

#ifndef __SYNTHESIS__
		int s_dbg_count_bytes = 0;
#endif

		t_input tmp_r;
		PRODSTR: for (int s_index = 0; s_index < c_index; s_index++) {
			tmp_r = i_data.read();
			ap_uint<64> tmp_r_par = tmp_r.data;
			
			for (uint8_t s_par = 0; s_par < c_par; s_par++) {
#pragma HLS pipeline off
				/* t_output tmp_w = (t_output)(tmp_r.data(8*(s_par+1)-1,8*s_par)); */
				t_output tmp_w = (t_output)(tmp_r_par & 0xff);
#ifndef __SYNTHESIS__
#ifdef DEBUG
					std::cout << (ap_uint<8>)(tmp_w) << " ";
#endif
#endif
				o_data.write(tmp_w);
				tmp_r_par = tmp_r_par >> 8;
			}

#ifndef __SYNTHESIS__
			s_dbg_count_bytes += c_par;
#endif
			
		}

#ifndef __SYNTHESIS__
#ifdef DEBUG
		std::cout << std::endl;
#endif
#endif

		o_last.write(tmp_r.last);

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Read " << s_dbg_count_bytes << " bytes" << std::endl;
		std::cout << "Producing last signal" << std::endl;
#endif

#endif

		/* if (tmp_r.last) */
		/* 	break; */

	/* } */

#ifndef __SYNTHESIS__
	EmptyStream<t_input>(i_data);
#ifdef DEBUG
	std::cout << "PRODUCESTREAM: " << c_ih << " " << c_iw << " " << c_ich << std::endl;
#endif
#endif

}

// For input weights
template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_str
> void ProduceStream(
	t_input *i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_output> &s_i_data
) {

	const int c_index = c_och*c_ich*c_ih*c_iw;

	/* while(1) { */
		for (int s_oh = 0; s_oh < c_oh; s_oh++) {
			for (int s_ow = 0; s_ow < c_ow; s_ow++) {
				PRODSTR: for (int s_index = 0; s_index < c_index; s_index++) {
					s_i_data.write((t_output)(i_data[s_index]));
				}
			}
		}

		ap_uint<1> s_last;
		s_last = i_last.read();
		/* if (s_last) */
		/* 	break; */
	/* } */

}
//
// For input weights
template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh
> void ProduceStream(
	const t_input i_data[c_och*c_ich*c_iw*c_ih],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_output> o_data[c_ih*c_iw]
) {

	const int c_index = c_oh*c_ow;
	const int c_stream_sel = c_ih*c_iw;
	const int c_ch = c_ich*c_och;
#pragma HLS array_partition type=cyclic factor=c_stream_sel variable=i_data

	/* while(1) { */

		for (uint16_t s_index = 0; s_index < c_index; s_index++) {
			uint16_t s_addr = 0;
			for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
				#pragma HLS pipeline
				for (uint8_t s_stream_sel = 0; s_stream_sel < c_stream_sel; s_stream_sel++) {
					o_data[s_stream_sel].write((t_output)(i_data[s_addr]));
					s_addr++;
				}
			}
		}

		ap_uint<1> s_last;
		s_last = i_last.read();
		/* if (i_last.read()) */
		/* 	break; */

	/* } */
}

template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_ops
> void ProduceStream(
	const t_input i_data[c_och*c_ich*c_iw*c_ih],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_output> o_data[c_ih*c_iw]
) {

	const int c_index = c_oh*c_ow;
	const int c_stream_sel = c_ih*c_iw;
	const int c_ch = c_ich*c_och/c_ops;
#pragma HLS array_partition type=cyclic factor=c_stream_sel variable=i_data

	/* while(1) { */

		for (uint16_t s_index = 0; s_index < c_index; s_index++) {
			uint16_t s_addr = 0;
			for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
				for (uint8_t s_stream_sel = 0; s_stream_sel < c_stream_sel; s_stream_sel++) {
					#pragma HLS pipeline
					for (uint16_t s_ops = 0; s_ops < c_ops; s_ops++) {
						o_data[s_stream_sel].write((t_output)(i_data[s_addr]));
						s_addr++;
					}
				}
			}
		}

		ap_uint<1> s_last;
		s_last = i_last.read();
		/* if (i_last.read()) */
		/* 	break; */

	/* } */
}

template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_ow,
	int c_oh
> void ProduceStream(
	const t_input i_data[c_och*c_ich],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_oh*c_ow;
	const int c_ch = c_ich*c_och;

	/* while(1) { */
		for (uint16_t s_index = 0; s_index < c_index; s_index++) {
			for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
#pragma HLS pipeline
				o_data.write((t_output)(i_data[s_ch]));
			}
		}
		ap_uint<1> s_last;
		s_last = i_last.read();
		/* if (i_last.read()) */
		/* 	break; */
	/* } */

}

template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_ow,
	int c_oh,
	int c_ops
> void ProduceStream(
	const t_input i_data[c_och*c_ich],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_output> &o_data
) {

	const int c_index = c_oh*c_ow;
	const int c_ch = c_ich*c_och/c_ops;
	const uint8_t c_log_ops = (uint8_t)(log2(c_ops));

	/* while(1) { */
		for (uint16_t s_index = 0; s_index < c_index; s_index++) {
			for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
/* #pragma HLS loop_merge */
#pragma HLS pipeline
				t_output s_data = 0;
				for (uint8_t s_ops = 0; s_ops < c_ops; s_ops++) {
					s_data(8*(s_ops+1)-1, 8*s_ops) = i_data[(s_ch << c_log_ops) + s_ops];
				}
				o_data.write((t_output)(s_data));
			}
		}
		ap_uint<1> s_last;
		s_last = i_last.read();
		/* if (i_last.read()) */
		/* 	break; */
	/* } */

}

template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_ow,
	int c_oh,
	int c_fw,
	int c_fh,
	int c_ops
> void ProduceStream(
	const t_input i_data[c_fh*c_fw][c_och*c_ich/c_ops+1],
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_output> o_data[c_fh*c_fw]
) {

	const int c_o_index = c_oh*c_ow;
	const int c_index = c_fh*c_fw;
	const int c_ch = c_ich*c_och/c_ops;
	/* while(1) { */
	for (uint16_t s_o_index = 0; s_o_index < c_o_index; s_o_index++) {
		uint16_t s_ch = 0;
		for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
			for (uint8_t s_och = 0; s_och < c_och/c_ops; s_och++) {
	/* #pragma HLS loop_merge */
#pragma HLS pipeline
				for (uint8_t s_index = 0; s_index < c_index; s_index++) {
#pragma HLS pipeline
					o_data[s_index].write((t_output)(i_data[s_index][s_ch]));
				}
				s_ch++;
			}
		}
	}
	ap_uint<1> s_last;
	s_last = i_last.read();
		/* if (i_last.read()) */
		/* 	break; */
	/* } */

}

///////////////////////////// FROM STREAM TO POINTER ////////////////////////// 

/* // For output activations */
/* template < */
/* 	class t_input, */
/* 	class t_output, */
/* 	int c_och, */
/* 	int c_ow, */
/* 	int c_oh */
/* > void ConsumeStream( */
/* 	hls::stream<t_input> &i_data, */
/* 	t_output *o_data */
/* ) { */

/* #ifndef __SYNTHESIS__ */

/* 	if (i_data.empty()) */
/* 		return; */

/* #endif */

/* 	t_input s_read; */
/* 	const int c_index = c_och*c_oh*c_ow; */

/* 	for (int s_index = 0; s_index < c_index; s_index++) { */

/* 		s_read = i_data.read(); */
/* 		o_data[s_index] = (t_output)(s_read); */

/* 	} */

/* } */

// For output activations
template <
	class t_input,
	class t_output,
	int c_och,
	int c_ow,
	int c_oh
> void ConsumeStream(
	hls::stream<t_input> &i_data,
	t_output o_data[c_och*c_ow*c_oh]
) {

#ifndef __SYNTHESIS__

	if (i_data.empty())
		return;

#endif

	t_input s_read;
	const int c_index = c_och*c_oh*c_ow;

	for (int s_index = 0; s_index < c_index; s_index++) {

		s_read = i_data.read();
		o_data[s_index] = (t_output)(s_read);

	}

}

template <
	class t_input,
	class t_output,
	int c_och,
	int c_ow,
	int c_oh
> void ConsumeStream(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_output> &o_data
) {

#ifndef __SYNTHESIS__

	if (i_data.empty())
		return;

#endif

	const int c_index = c_och*c_oh*c_ow;

	/* while(1) { */

		for (int s_index = 0; s_index < c_index-1; s_index++) {

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

		/* if (tmp.last) */
		/* 	break; */

	/* } */

}

/* template < */
/* 	class t_input, */
/* 	class t_output, */
/* 	int c_och, */
/* 	int c_ow, */
/* 	int c_oh */
/* > void ConsumeStream( */
/* 	hls::stream<t_input> &i_data, */
/* 	hls::stream<ap_uint<1>> &i_last, */
/* 	hls::stream<t_output> &o_data */
/* ) { */

/* #ifndef __SYNTHESIS__ */

/* 	if (i_data.empty()) */
/* 		return; */

/* #endif */

/* 	const int c_par = c_bits/8; */
/* 	const int c_index = c_och*c_oh*c_ow/c_par; */
/* 	const int c_out_pad = (c_och*c_oh*c_ow)%c_par; */

/* 	for (int s_index = 0; s_index < c_index-1; s_index++) { */

/* 		t_input s_read = i_data.read(); */
/* 		tmp.data = s_read; */
/* 		tmp.last = false; */
/* 		o_data.write(tmp); */

/* 	} */

/* 	t_input s_read = i_data.read(); */
/* 	tmp.data = s_read; */
/* 	tmp.last = i_last.read(); */
/* 	o_data.write(tmp); */

/* } */

template <
	class t_input,
	class t_output,
	int c_och,
	int c_ow,
	int c_oh,
	int c_bits
> void ConsumeStream(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_output> &o_data
) {

#ifndef __SYNTHESIS__

	if (i_data.empty())
		return;

#endif

	const int c_par = c_bits/8;
	const int c_index = c_och*c_oh*c_ow/c_par;
	const int c_out_pad = (c_och*c_oh*c_ow)%c_par;

	for (int s_index = 0; s_index < c_index; s_index++) {

		t_output tmp;
		for (int s_par = 0; s_par < c_par; s_par++) {
			#pragma HLS pipeline
			t_input s_read = i_data.read();
			tmp.data((s_par + 1)*8-1,s_par*8) = s_read;
		}
		tmp.last = false;
		o_data.write(tmp);

	}

	t_output tmp;
	tmp.data = 0;
	for (int s_out_pad = 0; s_out_pad < c_out_pad; s_out_pad++) {
		#pragma HLS pipeline
		t_input s_read = i_data.read();
		tmp.data((s_out_pad + 1)*8-1,s_out_pad*8) = s_read;
	}
	/* The input last stream doesn't count the number of activations streamed but the */
	/* number of batches analyzed */
	tmp.last = i_last.read();
	o_data.write(tmp);

}

///////////////////////////// FROM STREAM TO STREAM ///////////////////////////

template <
	class t_input,
	class t_output,
	int c_ich,
	int c_och,
	int c_iw,
	int c_ih,
	int c_ow,
	int c_oh,
	int c_pad
> void PadStream(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> &o_data
) {

	/* while(1) { */

#ifndef __SYNTHESIS__

		while(i_data.empty());

#endif

		if (c_pad == 0) {
			for (uint8_t s_ih = 0; s_ih < c_ih; s_ih++) {
				for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
					for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
							o_data.write((t_output)(i_data.read()));
					}
				}
			}
		}

		if (c_pad == 1) {
			for (uint8_t s_ih = 0; s_ih < c_ih; s_ih++) {
				for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
					for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
							o_data.write((t_output)(i_data.read()));
					}
				}
			}
		}

		ap_uint<1> s_last = i_last.read();
		o_last.write(s_last);
		/* if (s_last) */
		/* 	break; */

	/* } */

}


//////////////////////////// BLOCK INTERFACES /////////////////////////////////

template <
	class t_acc,
	class t_output,
	int c_och,
	int c_ow,
	int c_fh,
	int c_pad
> void WriteOutputKernel(
	uint16_t s_oh,
	uint16_t s_ow,
	t_acc s_acc_buffer[c_fh*2][c_ow][c_och],
	hls::stream<t_output> &o_data
) {

	for (uint8_t s_och = 0; s_och < c_och; s_och++) {

			t_output s_out_buffer = (t_output)(s_acc_buffer[s_oh][s_ow][s_och]);
			o_data.write(s_out_buffer);
			s_acc_buffer[s_oh][s_ow][s_och] = 0;

	}

}

template <
	class t_input,
	int c_ich,
	int c_iw,
	int c_ih
> void PadInput(
	hls::stream<t_input> &o_data
) {

	const int c_index = c_ich*c_ih*c_iw;

	PRODSTR: for (int s_index = 0; s_index < c_index; s_index++) {
		o_data.write(0);
	}

}

template <
	class t_input,
	int c_ich,
	int c_iw,
	int c_ih,
	int c_fw,
	int c_fh,
	int c_pad
> void PadInput(
	hls::stream<t_input> &o_data
) {

/* #pragma HLS pipeline */

	/* This handles padding aware inputs */

	const int c_pad_index_h = c_pad * (c_fh - 1);
	const int c_pad_index_w = c_pad * (c_fw - 1);
	const int c_ih_pad = c_ih + c_pad_index_h;
	const int c_iw_pad = c_iw + c_pad_index_w;

	/* Top padding */
	for (uint8_t s_ih = 0; s_ih < c_ih_pad; s_ih++){
		for (uint8_t s_iw = 0; s_iw < c_iw_pad; s_iw++) {
			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
#pragma HLS loop_flatten
				o_data.write(0);
			}
		}
	}

}

template <
	class t_input,
	int c_ich,
	int c_iw,
	int c_ih,
	int c_fw,
	int c_fh,
	int c_pad
> void PadInput(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_input> &o_data
) {

/* #pragma HLS pipeline */

	/* This handles padding aware inputs */

	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;

	const int c_ih_pad = c_ih + c_pad_index_h*2;
	const int c_iw_pad = c_iw + c_pad_index_w*2;

#ifndef __SYNTHESIS__
	while(i_data.empty());
#endif

	/* while(1) { */

		/* Top padding */
		for (uint8_t s_pad = 0; s_pad < c_pad_index_h; s_pad++){
			for (uint8_t s_iw = 0; s_iw < c_iw_pad; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
#ifndef __SYNTHESIS__
#ifdef DEBUG
					std::cout << (ap_uint<8>)(0) << " ";
#endif
#endif
				}
			}
		}

		for (uint8_t s_ih = 0; s_ih < c_ih; s_ih++) {

			/* Right padding */
			for (uint8_t s_pad = 0; s_pad < c_pad_index_w; s_pad++){
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
#ifndef __SYNTHESIS__
#ifdef DEBUG
					std::cout << (ap_uint<8>)(0) << " ";
#endif
#endif
				}
			}

			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					o_data.write(s_input);
#ifndef __SYNTHESIS__
#ifdef DEBUG
					std::cout << (ap_uint<8>)(s_input) << " ";
#endif
#endif
				}
			}

			/* Left padding */
			for (uint8_t s_pad = 0; s_pad < c_pad_index_w; s_pad++){
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
#ifndef __SYNTHESIS__
#ifdef DEBUG
					std::cout << (ap_uint<8>)(0) << " ";
#endif
#endif
				}
			}

		}

		for (uint8_t s_pad = 0; s_pad < c_pad_index_h; s_pad++){
			for (uint8_t s_iw = 0; s_iw < c_iw_pad; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
#ifndef __SYNTHESIS__
#ifdef DEBUG
					std::cout << (ap_uint<8>)(0) << " ";
#endif
#endif
				}
			}
		}

#ifndef __SYNTHESIS__
#ifdef DEBUG
					std::cout << std::endl;
#endif
#endif
/* // TODO: Problem seems multiple reads */
		ap_uint<1> s_last = i_last.read();
		o_last.write(s_last);
		/* if (s_last) */
		/* 	break; */

	/* } */

#ifndef __SYNTHESIS__
	EmptyStream<t_input>(i_data);
#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG
	std::cout << "PADINPUT: " << c_ih << " " << c_iw << " " << c_ich << std::endl;
#endif
#endif

}

template <
	class t_input,
	int c_ich,
	int c_iw,
	int c_ih,
	int c_fw,
	int c_fh,
	int c_pad
> void ForwardStream(
	hls::stream<t_input> &i_data
) {

	/* while(1) { */

#ifndef __SYNTHESIS__

		while(i_data.empty());

#endif

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

	/* } */

}

template <
	class t_input,
	int c_ich,
	int c_iw,
	int c_ih,
	int c_fw,
	int c_fh,
	int c_pad
> void ForwardStream(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_input> &o_forward
) {

	/* while(1) { */

#ifndef __SYNTHESIS__

		while(i_data.empty());

#endif

		const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
		const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
		const int c_ih_start = -1*c_pad_index_h;
		const int c_iw_start = -1*c_pad_index_w;
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

		ap_uint<1> s_last = i_last.read();
		o_last.write(s_last);
		/* if (s_last) */
		/* 	break; */

	/* } */

}

template <
	int c_split
> void SplitStream(
	hls::stream<ap_uint<1>> &i_data,
	hls::stream<ap_uint<1>> o_data[c_split]
) {

	/* while(1) { */

		ap_uint<1> s_data = i_data.read();
		for (uint8_t s_split = 0; s_split < c_split; s_split++) {
#pragma HLS unroll
			o_data[s_split].write(s_data);
		}
		/* if (s_data) */
		/* 	break; */

	/* } */

}

template <
	class t_output,
	int c_och,
	int c_ow,
	int c_oh,
	int c_split
> void SplitStream(
	hls::stream<t_output> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<ap_uint<1>> &o_last,
	hls::stream<t_output> o_data[c_split]
) {

#ifndef __SYNTHESIS__

	while(i_data.empty());

#endif

	/* while(1) { */
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

		ap_uint<1> s_last = i_last.read();
		o_last.write(s_last);
		/* if (s_last) */
		/* 	break; */
	/* } */

}

//////////////////////////////////////////////////////////////////////////////

/* Line Buffers generation */
template <
	class t_input,
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
	int c_bypass,
	int c_bypass_w
> void ShiftOp(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_input> &o_compute
) {

/* #pragma HLS pipeline */

	const int c_starth = (c_fh-1)*(1-c_pad);
	const int c_startw = (c_fw-1)*(1-c_pad);
	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_pad = c_ih + c_pad_index_h*2;
	const int c_iw_pad = c_iw + c_pad_index_w*2;
	const int c_paddingh_shift = c_bypass*c_iw_pad*c_ich;
	const int c_paddingw_shift = c_bypass_w*c_ich;
	const int c_strideh_shift = (c_str-1)*c_iw_pad*c_ich;
	const int c_stridew_shift = (c_str-1)*c_ich;
	const int c_end_paddingh_shift = (c_fh - 1 - c_bypass)*c_iw_pad*c_ich;
	const int c_end_paddingw_shift = (c_fw - 1 - c_bypass_w)*c_ich;

	/* while(1) { */
#ifndef __SYNTHESIS__
		while(i_data.empty());
#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG
		std::cout << "START SHIFTOP" << std::endl;
#endif
#endif
		/* Shifting first lines through the fifo chain */
		/* After this shift, all the useless computations with data at the borders are */
		/* skipped */
		for (uint16_t s_index = 0; s_index < c_paddingh_shift; s_index++) {
			t_input s_input = i_data.read();
#ifndef __SYNTHESIS__
#ifdef DEBUG
			std::cout << (ap_uint<8>)(s_input) << " ";
#endif
#endif
		}

#ifndef __SYNTHESIS__
#ifdef DEBUG
		std::cout << std::endl;
		std::cout << "DISCARDER FIRST " << c_paddingh_shift << "LINES" << std::endl;
#endif
#endif

		for (uint8_t s_ih = c_starth; s_ih < c_ih; s_ih+=c_str) {

			/* Start shifting for padding */
			/* After this shift, the first row data are shifted forward */
			for (uint8_t s_index = 0; s_index < c_paddingw_shift; s_index++) {
				t_input s_input = i_data.read();
			}

#ifndef __SYNTHESIS__
#ifdef DEBUG
			std::cout << "DISCARDER FIRST " << c_paddingw_shift << "PIXELS" << std::endl;
#endif
#endif

			for (uint8_t s_iw = c_startw; s_iw < c_iw; s_iw+=c_str) {

				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					
					o_compute.write(s_input);
				}

				for (uint8_t s_index = 0; s_index < c_stridew_shift; s_index++) {
					t_input s_input = i_data.read();
				}

			}

			for (uint16_t s_index = 0; s_index < c_end_paddingw_shift; s_index++) {
				t_input s_input = i_data.read();
			}

			/* Start shifting for h stride */
			for (uint16_t s_index = 0; s_index < c_strideh_shift; s_index++) {
				t_input s_input = i_data.read();
			}

		}

		for (uint16_t s_index = 0; s_index < c_end_paddingh_shift; s_index++) {
			t_input s_input = i_data.read();
		}

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Waiting for last signal" << std::endl;
#endif

#endif

		ap_uint<1> s_last = i_last.read();
		/* if (s_last) */
		/* 	break; */

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Starting new image" << std::endl;
#endif

#endif

	/* } */

#ifndef __SYNTHESIS__
	EmptyStream<t_input>(i_data);
#ifdef DEBUG
	std::cout << "SHIFTOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif
#endif

}

template <
	class t_input,
	int c_ich,
	int c_och,
	int c_ih,
	int c_iw,
	int c_oh,
	int c_ow,
	int c_fh,
	int c_fw,
	int c_str,
	int c_pad
> void ShiftOp(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_input> o_compute[c_fh*c_fw]
) {

/* #pragma HLS pipeline */

	const int c_starth = (c_fh-1)*(1-c_pad);
	const int c_startw = (c_fw-1)*(1-c_pad);
	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_pad = c_ih + c_pad_index_h*2;
	const int c_iw_pad = c_iw + c_pad_index_w*2;
	const int c_paddingh_shift = (c_fh - 1)*c_iw_pad*c_ich;
	const int c_paddingw_shift = (c_fw - 1)*c_ich;
	const int c_strideh_shift = (c_str-1)*c_iw_pad*c_ich;
	const int c_stridew_shift = (c_str-1)*c_ich;
	const int c_end_paddingh_shift = (c_fh - 1)*c_iw_pad*c_ich;
	const int c_end_paddingw_shift = (c_fw - 1)*c_ich;
	const int c_index = c_fh*c_fw;

	LineBuffer<t_input, c_fh, c_fw, c_ich, c_iw> s_data;
	
#ifndef __SYNTHESIS__
		while(i_data.empty());
#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG
		std::cout << "START SHIFTOP" << std::endl;
#endif
#endif
		/* Shifting first lines through the fifo chain */
		/* After this shift, all the useless computations with data at the borders are */
		/* skipped */

		s_data.FillLineBuffer(i_data);

		for (uint8_t s_ih = c_starth; s_ih < c_ih; s_ih+=c_str) {

			for (uint8_t s_iw = c_startw; s_iw < c_iw; s_iw+=c_str) {

				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
#pragma HLS pipeline
					t_input s_input = s_data.PopFirst();
					for (uint8_t s_index = c_index-1; s_index > 0; s_index--) {
						o_compute[s_index].write(s_input);
						s_input = s_data.ShiftLineBuffer(s_index);
					}
					s_input = i_data.read();
					o_compute[0].write(s_input);
					s_data.PushFirst(s_input);
				}

				for (uint8_t s_stride = 0; s_stride < c_stridew_shift; s_stride++) {
#pragma HLS pipeline
					t_input s_input = s_data.PopFirst();
					for (uint8_t s_index = c_index-1; s_index > 0; s_index--) {
						s_input = s_data.ShiftLineBuffer(s_index);
					}
					s_input = i_data.read();
					s_data.PushFirst(s_input);
				}

			}

			if (s_ih < (c_ih-1-c_str+1)) {
#pragma HLS pipeline
				for (uint16_t s_pad = 0; s_pad < (c_paddingw_shift + c_strideh_shift); s_pad++) {
					t_input s_input = s_data.PopFirst();
					for (uint8_t s_index = c_index-1; s_index > 0; s_index--) {
						s_input = s_data.ShiftLineBuffer(s_index);
					}
					s_input = i_data.read();
					s_data.PushFirst(s_input);
				}
			}

		}

		s_data.EmptyLineBuffer();

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Waiting for last signal" << std::endl;
#endif

#endif

		ap_uint<1> s_last = i_last.read();
		/* if (s_last) */
		/* 	break; */

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Starting new image" << std::endl;
#endif

#endif


	/* } */

#ifndef __SYNTHESIS__
	EmptyStream<t_input>(i_data);
#ifdef DEBUG
	std::cout << "SHIFTOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif
#endif

}

template <
	class t_input,
	int c_ich,
	int c_och,
	int c_ih,
	int c_iw,
	int c_oh,
	int c_ow,
	int c_fh,
	int c_fw,
	int c_str,
	int c_pad
> void ShiftOp(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_input> o_compute[c_fh*c_fw],
	hls::stream<t_input> &o_data
) {

/* #pragma HLS pipeline */

	const int c_starth = (c_fh-1)*(1-c_pad);
	const int c_startw = (c_fw-1)*(1-c_pad);
	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_pad = c_ih + c_pad_index_h*2;
	const int c_iw_pad = c_iw + c_pad_index_w*2;
	const int c_paddingh_shift = (c_fh - 1)*c_iw_pad*c_ich;
	const int c_paddingw_shift = (c_fw - 1)*c_ich;
	const int c_strideh_shift = (c_str-1)*c_iw_pad*c_ich;
	const int c_stridew_shift = (c_str-1)*c_ich;
	const int c_end_paddingh_shift = (c_fh - 1)*c_iw_pad*c_ich;
	const int c_end_paddingw_shift = (c_fw - 1)*c_ich;
	const int c_index = c_fh*c_fw;

	LineBuffer<t_input, c_fh, c_fw, c_ich, c_iw> s_data;
	
#ifndef __SYNTHESIS__
		while(i_data.empty());
#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG
		std::cout << "START SHIFTOP" << std::endl;
#endif
#endif
		/* Shifting first lines through the fifo chain */
		/* After this shift, all the useless computations with data at the borders are */
		/* skipped */

		s_data.FillLineBuffer(i_data);

		for (uint8_t s_ih = c_starth; s_ih < c_ih; s_ih+=c_str) {

			for (uint8_t s_iw = c_startw; s_iw < c_iw; s_iw+=c_str) {

				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
#pragma HLS pipeline
					t_input s_input = s_data.PopFirst();
					o_data.write(s_input);
					for (uint8_t s_index = c_index-1; s_index > 0; s_index--) {
						o_compute[s_index].write(s_input);
						s_input = s_data.ShiftLineBuffer(s_index);
					}
					s_input = i_data.read();
					o_compute[0].write(s_input);
					s_data.PushFirst(s_input);
				}

				for (uint8_t s_stride = 0; s_stride < c_stridew_shift; s_stride++) {
#pragma HLS pipeline
					t_input s_input = s_data.PopFirst();
					o_data.write(s_input);
					for (uint8_t s_index = c_index-1; s_index > 0; s_index--) {
						s_input = s_data.ShiftLineBuffer(s_index);
					}
					s_input = i_data.read();
					s_data.PushFirst(s_input);
				}

			}

			for (uint16_t s_pad = 0; s_pad < c_strideh_shift; s_pad++) {
#pragma HLS pipeline
				t_input s_input = s_data.PopFirst();
				o_data.write(s_input);
				for (uint8_t s_index = c_index-1; s_index > 0; s_index--) {
					s_input = s_data.ShiftLineBuffer(s_index);
				}
				s_input = i_data.read();
				s_data.PushFirst(s_input);
			}

			if (s_ih < (c_ih-1-c_str+1)) {
				for (uint16_t s_pad = 0; s_pad < c_paddingw_shift; s_pad++) {
					t_input s_input = s_data.PopFirst();
					o_data.write(s_input);
					for (uint8_t s_index = c_index-1; s_index > 0; s_index--) {
						s_input = s_data.ShiftLineBuffer(s_index);
					}
					s_input = i_data.read();
					s_data.PushFirst(s_input);
				}
			}

		}

		s_data.EmptyLineBuffer(o_data);

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Waiting for last signal" << std::endl;
#endif

#endif

		ap_uint<1> s_last = i_last.read();
		/* if (s_last) */
		/* 	break; */

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Starting new image" << std::endl;
#endif

#endif


	/* } */

#ifndef __SYNTHESIS__
	EmptyStream<t_input>(i_data);
#ifdef DEBUG
	std::cout << "SHIFTOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif
#endif

}

template <
	class t_input,
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
	int c_bypass,
	int c_bypass_w
> void ShiftOp(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_input> &o_compute,
	hls::stream<t_input> &o_data
) {

/* #pragma HLS pipeline */

	const int c_starth = (c_fh-1)*(1-c_pad);
	const int c_startw = (c_fw-1)*(1-c_pad);
	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_pad = c_ih + c_pad_index_h*2;
	const int c_iw_pad = c_iw + c_pad_index_w*2;
	const int c_paddingh_shift = c_bypass*c_iw_pad*c_ich;
	const int c_paddingw_shift = c_bypass_w*c_ich;
	const int c_strideh_shift = (c_str-1)*c_iw_pad*c_ich;
	const int c_stridew_shift = (c_str-1)*c_ich;
	const int c_end_paddingh_shift = (c_fh - 1 - c_bypass)*c_iw_pad*c_ich;
	const int c_end_paddingw_shift = (c_fw - 1 - c_bypass_w)*c_ich;

	/* while(1) { */

#ifndef __SYNTHESIS__
		while(i_data.empty());
#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG
		std::cout << "START SHIFTOP" << std::endl;
#endif
#endif
		/* Shifting first lines through the fifo chain */
		/* After this shift, all the useless computations with data at the borders are */
		/* skipped */
		for (uint16_t s_index = 0; s_index < c_paddingh_shift; s_index++) {
			t_input s_input = i_data.read();
#ifndef __SYNTHESIS__
#ifdef DEBUG
			std::cout << (ap_uint<8>)(s_input) << " ";
#endif
#endif
			o_data.write(s_input);
		}

#ifndef __SYNTHESIS__
#ifdef DEBUG
		std::cout << std::endl;
		std::cout << "DISCARDER FIRST " << c_paddingh_shift << "LINES" << std::endl;
#endif
#endif

		for (uint8_t s_ih = c_starth; s_ih < c_ih; s_ih+=c_str) {

			/* Start shifting for padding */
			/* After this shift, the first row data are shifted forward */
			for (uint8_t s_index = 0; s_index < c_paddingw_shift; s_index++) {
				t_input s_input = i_data.read();
#ifndef __SYNTHESIS__
#ifdef DEBUG
				std::cout << (ap_uint<8>)(s_input) << " ";
#endif
#endif
				o_data.write(s_input);
			}

#ifndef __SYNTHESIS__
#ifdef DEBUG
			std::cout << "DISCARDER FIRST " << c_paddingw_shift << "PIXELS" << std::endl;
#endif
#endif
			for (uint8_t s_iw = c_startw; s_iw < c_iw; s_iw+=c_str) {

				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					o_compute.write(s_input);
					o_data.write(s_input);
				}

				for (uint8_t s_index = 0; s_index < c_stridew_shift; s_index++) {
					t_input s_input = i_data.read();
					o_data.write(s_input);
				}

			}

			for (uint16_t s_index = 0; s_index < c_end_paddingw_shift; s_index++) {
				t_input s_input = i_data.read();
				o_data.write(s_input);
			}

			/* Start shifting for h stride */
			for (uint16_t s_index = 0; s_index < c_strideh_shift; s_index++) {
				t_input s_input = i_data.read();
				o_data.write(s_input);
			}

		}

		for (uint16_t s_index = 0; s_index < c_end_paddingh_shift; s_index++) {
			t_input s_input = i_data.read();
			o_data.write(s_input);
		}


#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Waiting for last signal" << std::endl;
#endif

#endif

		ap_uint<1> s_last = i_last.read();
		/* if (s_last) */
		/* 	break; */

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Starting new image" << std::endl;
#endif

#endif


	/* } */

#ifndef __SYNTHESIS__
	EmptyStream<t_input>(i_data);
#ifdef DEBUG
	std::cout << "SHIFTOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif
#endif

}

template <
	class t_input,
	int c_ich,
	int c_och,
	int c_ih,
	int c_iw,
	int c_oh,
	int c_ow,
	int c_fh,
	int c_fw,
	int c_str,
	int c_pad
> void ShiftOp(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_input> &o_compute,
	hls::stream<t_input> &o_data,
	const int c_bypass,
	const int c_bypass_w
) {

/* #pragma HLS pipeline */

	const int c_starth = (c_fh-1)*(1-c_pad);
	const int c_startw = (c_fw-1)*(1-c_pad);
	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;
	const int c_ih_pad = c_ih + c_pad_index_h*2;
	const int c_iw_pad = c_iw + c_pad_index_w*2;
	const int c_paddingh_shift = c_bypass*c_iw_pad*c_ich;
	const int c_paddingw_shift = c_bypass_w*c_ich;
	const int c_strideh_shift = (c_str-1)*c_iw_pad*c_ich;
	const int c_stridew_shift = (c_str-1)*c_ich;
	const int c_end_paddingh_shift = (c_fh - 1 - c_bypass)*c_iw_pad*c_ich;
	const int c_end_paddingw_shift = (c_fw - 1 - c_bypass_w)*c_ich;

	/* while(1) { */

#ifndef __SYNTHESIS__
		while(i_data.empty());
#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG
		std::cout << "START SHIFTOP" << std::endl;
#endif
#endif
		/* Shifting first lines through the fifo chain */
		/* After this shift, all the useless computations with data at the borders are */
		/* skipped */
		for (uint16_t s_index = 0; s_index < c_paddingh_shift; s_index++) {
			t_input s_input = i_data.read();
			o_data.write(s_input);
		}

		for (uint8_t s_ih = c_starth; s_ih < c_ih; s_ih+=c_str) {

			/* Start shifting for padding */
			/* After this shift, the first row data are shifted forward */
			for (uint16_t s_index = 0; s_index < c_paddingw_shift; s_index++) {
				t_input s_input = i_data.read();
				o_data.write(s_input);
			}

			for (uint8_t s_iw = c_startw; s_iw < c_iw; s_iw+=c_str) {

				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					t_input s_input = i_data.read();
					o_compute.write(s_input);
					o_data.write(s_input);
				}

				for (uint8_t s_index = 0; s_index < c_stridew_shift; s_index++) {
					t_input s_input = i_data.read();
					o_data.write(s_input);
				}

			}

			for (uint16_t s_index = 0; s_index < c_end_paddingw_shift; s_index++) {
				t_input s_input = i_data.read();
				o_data.write(s_input);
			}

			/* Start shifting for h stride */
			for (uint16_t s_index = 0; s_index < c_strideh_shift; s_index++) {
				t_input s_input = i_data.read();
				o_data.write(s_input);
			}

		}

		for (uint16_t s_index = 0; s_index < c_end_paddingh_shift; s_index++) {
			t_input s_input = i_data.read();
			o_data.write(s_input);
		}


#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Waiting for last signal" << std::endl;
#endif

#endif

		ap_uint<1> s_last = i_last.read();
		/* if (s_last) */
		/* 	break; */

#ifndef __SYNTHESIS__

#ifdef DEBUG
		std::cout << "Starting new image" << std::endl;
#endif

#endif


	/* } */

#ifndef __SYNTHESIS__
	EmptyStream<t_input>(i_data);
#ifdef DEBUG
	std::cout << "SHIFTOP: " << c_ih << " " << c_iw << " " << c_ich << " " << c_str << " " << c_pad << " " << std::endl;
#endif
#endif

}

#endif
