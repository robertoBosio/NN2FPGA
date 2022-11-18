#ifndef __LINEBUFFER__
#define __LINEBUFFER__

#include "ap_int.h"
#include "hls_stream.h"
#include "Debug.hpp"

template <class t_stream, int c_depth>
class LineStream {
	public:
#ifndef __SYNTHESIS__
		int n_elems;
#endif
		hls::stream<t_stream, c_depth> s_stream;

		LineStream() { 
#ifndef __SYNTHESIS__
			n_elems = 0;
#endif
#pragma HLS STREAM variable=s_stream depth=c_depth type=fifo
		}

		void write(t_stream i_write) {
			s_stream.write(i_write);
#ifndef __SYNTHESIS__
			if (!(n_elems < c_depth))
				std::cout << "WRITING FULL BUFFER" << std::endl;
			n_elems++;
#endif
		}

		t_stream read() {
#ifndef __SYNTHESIS__
			if (n_elems > 0) 
				n_elems--;
#endif
			return s_stream.read();
		}

		bool full() {
#ifndef __SYNTHESIS__
			return !(n_elems < c_depth);
#else
			return s_stream.full();
#endif
		}

		bool empty() {
			return s_stream.empty();
		}

};

template <
	class t_stream,
	int c_fh,
	int c_fw,
	int c_ich,
	int c_iw
> class LineBuffer{
	public:
		const int c_index = c_fh*c_fw;
		LineStream<t_stream, c_ich> s_stream_c[c_fh][c_fw-1];
		LineStream<t_stream, c_ich*c_iw> s_stream_r[c_fh-1];
		const ap_uint<8> c_addr[c_fh*c_fw-1] = {0xff, 0x10, 0x21, 0x42, 0x54, 0x65, 0x86, 0x98};

		LineBuffer() {}

		/* Fill the buffer for init */
		void ShiftIn(t_stream i_write) {

			for (ap_int<4> s_fh = c_fh-1; s_fh > -1; s_fh--) {

				for (ap_int<4> s_fw = c_fw-2; s_fw > -1; s_fw--) {
					if (!s_stream_c[s_fh][s_fw].full()){
						s_stream_c[s_fh][s_fw].write(i_write);
						return;
					}
				}

				if (s_fh > 0) {
					if (!s_stream_r[s_fh-1].full()){
						s_stream_r[s_fh-1].write(i_write);
						return;
					}
				}

			}
			
		}

		/* Fill and retrieve */
		t_stream PopFirst() {
			return s_stream_c[c_fh-1][c_fw-2].read();
		}

		void PushFirst(t_stream i_write) {
			s_stream_c[0][0].write(i_write);
		}

		/* Fill and retrieve */
		t_stream GetLineBuffer(
			ap_uint<2> i_fh,
			ap_uint<2> i_fw
		) {

#pragma HLS inline
			if (i_fh == 3)
				return 0;

			if (i_fw(1,1)==1)
				return s_stream_r[i_fh].read();
			else
				return s_stream_c[i_fh][i_fw].read();
			
		}

		void SetLineBuffer(
			ap_uint<2> i_fh,
			ap_uint<2> i_fw,
			t_stream i_write
		) {

#pragma HLS inline
			if (i_fh == 3)
				return;

			if (i_fw(1,1)==1)
				s_stream_r[i_fh].write(i_write);
			else
				s_stream_c[i_fh][i_fw].write(i_write);
			
		}

		t_stream ShiftLineBuffer(uint8_t i_index) {

#pragma HLS inline
			ap_uint<8> s_addr = c_addr[i_index-1];
			t_stream s_stream = GetLineBuffer(s_addr(3,2), s_addr(1,0));
			SetLineBuffer(s_addr(7,6), s_addr(5,4), s_stream);

			return s_stream;
		}

		/* Empty the buffer */
		t_stream ShiftOut() {

			t_stream s_stream = 0;
			bool s_shift = true;
			for (ap_int<4> s_fh = c_fh-1; s_fh > -1; s_fh--) {

				for (ap_int<4> s_fw = c_fw-2; s_fw > -1; s_fw--) {
					if (!s_stream_c[s_fh][s_fw].empty() && s_shift){
						s_stream = s_stream_c[s_fh][s_fw].read();
						s_shift = false;
					}
				}

				if (s_fh > 0) {
					if (!s_stream_r[s_fh-1].empty() && s_shift){
						return s_stream_r[s_fh-1].read();
						s_shift = false;
					}
				}

			}
			
			return s_stream;
			
		}

#ifndef __SYNTHESIS__
		void PrintNumData() {
			std::cout << "-----------------------------" << std::endl;
			for (ap_int<4> s_fh = c_fh-1; s_fh > -1; s_fh--) {

				for (ap_int<4> s_fw = c_fw-2; s_fw > -1; s_fw--) {
					std::cout <<s_stream_c[s_fh][s_fw].n_elems << std::endl;
				}

				if (s_fh > 0) {
					std::cout << s_stream_r[s_fh-1].n_elems << std::endl;
				}

			}
		}
#endif

};

#endif
