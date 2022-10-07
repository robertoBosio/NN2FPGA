#ifndef __UTILS__
#define __UTILS__

#include "Debug.hpp"

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

	while(1) {
		PRODSTR: for (int s_index = 0; s_index < c_index; s_index++) {
			o_data.write((t_output)(i_data[s_index]));
		}
		ap_uint<1> s_last = i_last.read();
		if (s_last)
			break;
	}

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
	hls::stream<t_output> &s_i_data
) {

	const int c_par = c_i_data/8;
	const int c_index = (c_ich*c_ih*c_iw)/c_par;

	while(1) {

		t_input tmp_r;
		PRODSTR: for (int s_index = 0; s_index < c_index; s_index++) {
			tmp_r = i_data.read();
			for (int s_par = 0; s_par < c_par; s_par++) {
#pragma HLS pipeline
				t_output tmp_w = (t_output)(tmp_r.data(8*(s_par+1)-1,8*s_par));
				s_i_data.write(tmp_w);
			}
		}
		o_last.write(tmp_r.last);

		if (tmp_r.last)
			break;

	}

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

	for (int s_oh = 0; s_oh < c_oh; s_oh++) {
		for (int s_ow = 0; s_ow < c_ow; s_ow++) {
			PRODSTR: for (int s_index = 0; s_index < c_index; s_index++) {
				s_i_data.write((t_output)(i_data[s_index]));
			}
		}
	}

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

	while(1) {

		for (uint16_t s_index = 0; s_index < c_index; s_index++) {
			uint16_t s_addr = 0;
			for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
				for (uint8_t s_stream_sel = 0; s_stream_sel < c_stream_sel; s_stream_sel++) {
					#pragma HLS UNROLL
					o_data[s_stream_sel].write((t_output)(i_data[s_addr]));
					s_addr++;
				}
			}
		}

		if (i_last.read())
			break;

	}
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

	while(1) {
		for (uint16_t s_index = 0; s_index < c_index; s_index++) {
			for (uint16_t s_ch = 0; s_ch < c_ch; s_ch++) {
				o_data.write((t_output)(i_data[s_ch]));
			}
		}
		if (i_last.read())
			break;
	}

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

	while(1) {

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

		if (tmp.last)
			break;

	}

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
	hls::stream<t_output> &o_data
) {

	while(1) {

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
		if (s_last)
			break;

	}

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
	hls::stream<t_input> &o_data
) {

	/* This handles padding aware inputs */

	const int c_pad_index_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_index_w = c_pad * (c_fw - 1) / 2;

	const int c_ih_pad = c_ih + c_pad_index_h*2;
	const int c_iw_pad = c_iw + c_pad_index_w*2;

	while(1) {

		/* Top padding */
		for (uint8_t s_pad = 0; s_pad < c_pad_index_h; s_pad++){
			for (uint8_t s_iw = 0; s_iw < c_iw_pad; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
				}
			}
		}

		for (uint8_t s_ih = 0; s_ih < c_ih; s_ih++) {

			/* Right padding */
			for (uint8_t s_pad = 0; s_pad < c_pad_index_w; s_pad++){
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
				}
			}

			for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(i_data.read());
				}
			}

			/* Left padding */
			for (uint8_t s_pad = 0; s_pad < c_pad_index_w; s_pad++){
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
				}
			}

		}

		for (uint8_t s_pad = 0; s_pad < c_pad_index_h; s_pad++){
			for (uint8_t s_iw = 0; s_iw < c_iw_pad; s_iw++) {
				for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
					o_data.write(0);
				}
			}
		}

#ifndef __SYNTHESIS__
		EmptyStream<t_input>(i_data);
#endif

#ifndef __SYNTHESIS__
		std::cout << "PADINPUT: " << c_ih << " " << c_iw << " " << c_ich << std::endl;
#endif

		ap_uint<1> s_last = i_last.read();
		if (s_last)
			break;

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
> void ForwardStream(
	hls::stream<t_input> &i_data
) {

	while(1) {

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
> void ForwardStream(
	hls::stream<t_input> &i_data,
	hls::stream<ap_uint<1>> &i_last,
	hls::stream<t_input> &o_forward
) {

	while(1) {

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
		if (s_last)
			break;

	}

}

template <
	int c_split
> void SplitStream(
	hls::stream<ap_uint<1>> &i_data,
	hls::stream<ap_uint<1>> o_data[c_split]
) {

	while(1) {

		ap_uint<1> s_data = i_data.read();
		for (uint8_t s_split = 0; s_split < c_split; s_split++) {
#pragma HLS unroll
			o_data[s_split].write(s_data);
		}
		if (s_data)
			break;

	}

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
	hls::stream<t_output> o_data[c_split]
) {

	while(1) {
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
		if (s_last)
			break;
	}

}

#endif
