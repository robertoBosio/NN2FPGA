#ifndef __LINEBUFFER__
#define __LINEBUFFER__

#include "ap_int.h"
#include "hls_stream.h"
#include "Debug.hpp"

template <class t_stream, int c_depth>
class LineStream {
	private:
#ifndef __SYNTHESIS__
		int n_elems;
#endif
		hls::stream<t_stream, c_depth> s_stream;

	public:
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
		ap_int<4> s_fh_w;
		ap_int<4> s_fw_w;
		ap_int<4> s_fh_r;
		ap_int<4> s_fw_r;

		LineBuffer() { 
			s_fh_w = c_fh-1;
			s_fw_w = c_fw-2;
			s_fh_r = c_fh-1;
			s_fw_r = c_fw-2;
		}

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
			s_fh_r = c_fh-1;
			s_fw_r = c_fw-3;
			s_fh_w = c_fh-1;
			s_fw_w = c_fw-2;
			return s_stream_c[c_fh-1][c_fw-2].read();
		}

		void PushFirst(t_stream i_write) {
			s_fh_r = c_fh-1;
			s_fw_r = c_fw-2;
			s_fh_w = c_fh-1;
			s_fw_w = c_fw-2;
			s_stream_c[0][0].write(i_write);
		}

		/* Fill and retrieve */
		t_stream ShiftLineBuffer() {
			t_stream s_stream = 0;

			if (s_fw_r > -1) {
				s_stream = s_stream_c[s_fh_r][s_fw_r].read();
				s_fw_r--;
			} else {
				s_fw_r = c_fw-2;
				if (s_fh_r > 0) {
					s_stream = s_stream_r[s_fh_r-1].read();
					s_fh_r--;
				} else {
					s_fh_r = c_fh-1;
				}
			}
			
			if (s_fw_w > 0 || ((s_fw_w > -1) && (s_fh_w > 0))) {
				s_stream_c[s_fh_w][s_fw_w].write(s_stream);
				s_fw_w--;
			} else {
				s_fw_w = c_fw-2;
				if (s_fh_w > 0) {
					s_stream_r[s_fh_w-1].write(s_stream);
					s_fh_w--;
				} else {
					s_fh_w = c_fh-1;
				}
			}
			
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
