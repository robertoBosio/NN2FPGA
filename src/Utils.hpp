#ifndef __UTILS__
#define __UTILS__

#include "Debug.hpp"

//////////////////////////// FROM POINTER TO STREAM /////////////////////////// 
// For input activations
template <
	class t_input,
	class t_output,
	int c_ich,
	int c_iw,
	int c_ih
> void ProduceStream(
	t_input *i_data,
	hls::stream<t_output> &s_i_data
) {

	const int c_index = c_ich*c_ih*c_iw;

	PRODSTR: for (int s_index = 0; s_index < c_index; s_index++) {
		s_i_data.write((t_output)(i_data[s_index]));
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
	hls::stream<t_output> o_data[c_ih*c_iw]
) {

	const int c_index = c_oh*c_ow;
	const int c_stream_sel = c_ih*c_iw;
	const int c_ch = c_ich*c_och;
#pragma HLS array_partition type=cyclic factor=c_stream_sel variable=i_data

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

}

///////////////////////////// FROM STREAM TO POINTER ////////////////////////// 

// For output activations
template <
	class t_input,
	class t_output,
	int c_och,
	int c_ow,
	int c_oh
> void ConsumeStream(
	hls::stream<t_input> &i_data,
	t_output *o_data
) {

	if (i_data.empty())
		return;

	t_input s_read;
	const int c_index = c_och*c_oh*c_ow;

	for (int s_index = 0; s_index < c_index; s_index++) {

		s_read = i_data.read();
		o_data[s_index] = (t_output)(s_read);

	}

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
	hls::stream<t_output> &o_data
) {

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
#pragma loop_flatten
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
	hls::stream<t_input> &o_data
) {

	/* This handles padding aware inputs */

	const int c_pad_index_h = c_pad * (c_fh - 1);
	const int c_pad_index_w = c_pad * (c_fw - 1);
	const int c_ih_pad = c_ih + c_pad_index_h;
	const int c_iw_pad = c_iw + c_pad_index_w;

	/* Top padding */
	for (uint8_t s_ih = 0; s_ih < c_ih; s_ih++){
		for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) {
			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
#pragma loop_flatten
				t_input s_data = i_data.read();
			}
		}
	}

	for (uint8_t s_ih = 0; s_ih < c_ih_pad; s_ih++){
		for (uint8_t s_iw = 0; s_iw < c_iw_pad; s_iw++) {
			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) {
#pragma loop_flatten
				o_data.write(0);
			}
		}
	}

	EmptyStream<t_input>(i_data);

#ifndef __SYNTHESIS__
	std::cout << "PADINPUT: " << c_ih << " " << c_iw << " " << c_ich << std::endl;
#endif

	/* const int c_pad_index_h = c_pad * (c_fh - 1) / 2; */
	/* const int c_pad_index_w = c_pad * (c_fw - 1) / 2; */

	/* /1* Top padding *1/ */
	/* for (uint8_t s_pad = 0; s_pad < c_pad_index_h; s_pad++){ */
	/* 	for (uint8_t s_iw = 0; s_iw < c_iw+c_pad_index_w*2; s_iw++) { */
	/* 		for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) { */
	/* 			o_data.write(0); */
	/* 		} */
	/* 	} */
	/* } */

	/* for (uint8_t s_ih = 0; s_ih < c_ih; s_ih++) { */

	/* 	/1* Right padding *1/ */
	/* 	for (uint8_t s_pad = 0; s_pad < c_pad_index_w; s_pad++){ */
	/* 		for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) { */
	/* 			o_data.write(0); */
	/* 		} */
	/* 	} */

	/* 	for (uint8_t s_iw = 0; s_iw < c_iw; s_iw++) { */
	/* 		for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) { */
	/* 			o_data.write(i_data.read()); */
	/* 		} */
	/* 	} */

	/* 	/1* Left padding *1/ */
	/* 	for (uint8_t s_pad = 0; s_pad < c_pad_index_w; s_pad++){ */
	/* 		for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) { */
	/* 			o_data.write(0); */
	/* 		} */
	/* 	} */

	/* } */

	/* for (uint8_t s_pad = 0; s_pad < c_pad_index_h; s_pad++){ */
	/* 	for (uint8_t s_iw = 0; s_iw < c_iw+c_pad_index_w*2; s_iw++) { */
	/* 		for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) { */
	/* 			o_data.write(0); */
	/* 		} */
	/* 	} */
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
	hls::stream<t_input> &i_data
) {

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
	hls::stream<t_input> &o_forward
) {

#ifndef __SYNTHESIS__

	while(i_data.empty());

#endif

	const int c_pad_index_h = c_pad * (c_fh - 1);
	const int c_pad_index_w = c_pad * (c_fw - 1);
	const int c_ih_end = c_ih + c_pad_index_h;
	const int c_iw_end = c_iw + c_pad_index_w;
	const int c_pad_h = c_pad * (c_fh - 1) / 2;
	const int c_pad_w = c_pad * (c_fw - 1) / 2;
	
	for (uint8_t s_ih = 0; s_ih < c_ih_end; s_ih++) { 
		for (uint8_t s_iw = 0; s_iw < c_iw_end; s_iw++) { 
			for (uint8_t s_ich = 0; s_ich < c_ich; s_ich++) { 
				t_input s_tmp = i_data.read();
				if ((s_ih >= c_pad_h) & (s_iw >= c_pad_w))
					o_forward.write(s_tmp);
			}
		}
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
	hls::stream<t_output> o_data[c_split]
) {

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

#endif
