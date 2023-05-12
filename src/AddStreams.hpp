#ifndef __ADDSTREAM__
#define __ADDSTREAM__

#include "ap_int.h"
#include "hls_stream.h"

template <class t_input, class t_output, const int c_ich, const int c_ih,
          const int c_iw>
void AddStreams(hls::stream<t_input> &i_data1, hls::stream<t_input> &i_data2,
                hls::stream<t_output> &o_data) {
  if (i_data1.empty() | i_data2.empty()) return;

  const int c_index = c_ich * c_ih * c_iw;

  for (int s_index = 0; s_index < c_index; s_index++) {
    t_input s_data1, s_data2;
    t_output s_o_data;

    i_data1.read(s_data1);
    i_data2.read(s_data2);

    s_o_data = s_data1 + s_data2;

    o_data.write(s_o_data);
  }
}

#endif
