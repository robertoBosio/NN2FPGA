#ifndef NN2FPGA_POOL_STREAM_H_
#define NN2FPGA_POOL_STREAM_H_

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "nn2fpga/debug.h"
#include "nn2fpga/line_buffer_utils.h"

namespace nn2fpga {

  template <class t_input_struct,
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
  pool_op(hls::stream<t_input_struct> i_data[1],
          hls::stream<t_output_struct> o_data[c_ow_ops])
  {

    static_assert(c_ops <= c_in_ops, "c_ops <= c_in_ops");
    static_assert(c_in_ops % c_ops == 0, "c_in_ops \% c_ops != 0");

    const int c_index = c_fh * c_fw;
    const int c_o_index = (c_oh * c_ow) / c_ow_ops;
    
    t_acc c_quant;
    if constexpr(c_pool == 0)
    {
      c_quant = 0;
    }
    else
    {
      c_quant = INT8_MIN;
    }

    bool s_last;
    t_acc s_acc_buff[c_ow_ops];

#ifndef __SYNTHESIS__
    std::cout << "INFO: Call to pool_op" << std::endl;
    std::cout << "\t\tc_ich = " << c_ich << std::endl;
    std::cout << "\t\tc_och = " << c_och << std::endl;
    std::cout << "\t\tc_ih = " << c_ih << std::endl;
    std::cout << "\t\tc_iw = " << c_iw << std::endl;
    std::cout << "\t\tc_oh = " << c_oh << std::endl;
    std::cout << "\t\tc_ow = " << c_ow << std::endl;
    std::cout << "\t\tc_fh = " << c_fh << std::endl;
    std::cout << "\t\tc_fw = " << c_fw << std::endl;
    std::cout << "\t\tc_str = " << c_str << std::endl;
    std::cout << "\t\tc_pad = " << c_pad << std::endl;
    std::cout << "\t\tc_pool = " << c_pool << std::endl;
    std::cout << "\t\tc_ow_ops = " << c_ow_ops << std::endl;
    std::cout << "\t\tc_ops = " << c_ops << std::endl;
    std::cout << "\t\tc_in_ops = " << c_in_ops << std::endl;

    for (auto i = 0; i < c_ow_ops; i++)
    {
      std::cout << "i_data[" << i << "].size() = " << i_data[i].size() << std::endl;
    }
#endif
    
    t_output_struct s_output_struct;
    t_input_struct s_input_struct;
    
    for (auto s_o_index = 0; s_o_index < c_o_index; s_o_index++)
    {

      for (auto s_och = 0; s_och < c_och; s_och += c_in_ops)
      {

        for (auto s_in_ops = 0; s_in_ops < c_in_ops; s_in_ops += c_ops)
        {
#pragma HLS pipeline style = stp II = 1

          for (auto s_ow_ops = 0; s_ow_ops < c_ow_ops; s_ow_ops++)
          {

            for (auto s_ops = 0; s_ops < c_ops; s_ops++)
            {
              
              // std::cout << "computing max between: ";
              for (auto s_fh = 0; s_fh < c_fh; s_fh++)
              {

                for (auto s_fw = 0; s_fw < c_fw; s_fw++)
                {
                  unsigned int s_index = s_fh * c_fw + s_fw + (c_ow_ops - s_ow_ops - 1) * c_str;
                  unsigned int s_acc_index = s_och + s_in_ops + s_ops;
                  
                  // Initializing the accumulator for each window
                  if ((s_fh == 0) && (s_fw == 0)){
                    s_acc_buff[s_ow_ops] = c_quant;
                  }

                  // Reading packets of s_in_ops channels
                  if ((s_in_ops == 0) && (s_ops == 0) && (s_index == 0))
                  {
                    s_input_struct = i_data[0].read();
                    s_last = s_input_struct.last;
                  }

                  // Pooling operation
                  if constexpr (c_pool == 0)
                  { // Average Pool
                    s_acc_buff[s_ow_ops] += s_input_struct.data[s_index][s_in_ops + s_ops];
                  }
                  if constexpr (c_pool == 1)
                  { // Max Pool
                    // std::cout << s_input_struct.data[s_index][s_in_ops + s_ops] << " ";
                    if (s_input_struct.data[s_index][s_in_ops + s_ops] > s_acc_buff[s_ow_ops])
                    {
                      s_acc_buff[s_ow_ops] = s_input_struct.data[s_index][s_in_ops + s_ops];
                    }
                  }

                  if (s_index == (c_index - 1))
                  {
                    t_div s_acc = s_acc_buff[s_ow_ops];
                    
                    if constexpr (c_pool == 0)
                    { // Average Pool
                      t_div s_divisor = c_index;
                      s_acc = s_acc / s_divisor;
                    }

                    s_output_struct.data[0][s_ops] = t_output(s_acc);
                    if (s_ops == (c_ops - 1))
                    {
                      s_output_struct.last = s_last;
                      o_data[0].write(s_output_struct);
                    }
                  }
                }
              }
              // std::cout << std::endl;
            }
          }
        }
      }
    }

#ifndef __SYNTHESIS__
#ifndef SKIP_ASSERTION
  for (auto i = 0; i < c_ow_ops; i++) {
    if (i_data[i].size() > 0)
      std::cout << "ERROR: pool_op: i_data[" << i << "].size() "
                << i_data[i].size() << " > 0\n" << std::flush;
    // assert(i_data[i].size() == 0);
  }
  assert(i_data[0].size() == 0);
  if (o_data[0].size() == 0) {
    std::cout << "ERROR: pool_op: o_data[0].size() " << o_data[0].size()
              << " == 0\n";
    assert(o_data[0].size() > 0);
  }
  std::cout << "pool_op: o_data[0].size() " << o_data[0].size() << "\n";
#endif /* SKIP_ASSERTION */
  std::cout << "end pool_op " << c_ich << std::endl;
#endif
  }
  
  template <class t_input_struct,
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
            int c_pool,
            int c_ops,
            int c_in_ops>
  void
  global_pool_op(hls::stream<t_input_struct> i_data[1],
          hls::stream<t_output_struct> o_data[1])
  {

    static_assert(c_ops <= c_in_ops, "c_ops <= c_in_ops");
    static_assert(c_in_ops % c_ops == 0, "c_in_ops \% c_ops != 0");

    t_acc c_quant = 0;
    if constexpr(c_pool == 0)
    {
      c_quant = 0;
    }
    else
    {
      c_quant = INT8_MIN;
    }

    bool s_last;
    t_acc s_acc_buff[c_och];
    const t_div s_divisor = c_ih * c_iw;

#ifndef __SYNTHESIS__
    std::cout << "INFO: Call to global_pool_op" << std::endl;
    std::cout << "\t\tc_ich = " << c_ich << std::endl;
    std::cout << "\t\tc_och = " << c_och << std::endl;
    std::cout << "\t\tc_ih = " << c_ih << std::endl;
    std::cout << "\t\tc_iw = " << c_iw << std::endl;
    std::cout << "\t\tc_oh = " << c_oh << std::endl;
    std::cout << "\t\tc_ow = " << c_ow << std::endl;
    std::cout << "\t\tc_pool = " << c_pool << std::endl;
    std::cout << "\t\tc_ops = " << c_ops << std::endl;
    std::cout << "\t\tc_in_ops = " << c_in_ops << std::endl;

    std::cout << "i_data[0].size() = " << i_data[0].size() << std::endl;
#endif
    
    t_output_struct s_output_struct;
    t_input_struct s_input_struct;
    
    for (auto s_fh = 0; s_fh < c_ih; s_fh++)
    {

      for (auto s_fw = 0; s_fw < c_iw; s_fw++)
      {
    
        for (auto s_och = 0; s_och < c_och; s_och += c_in_ops)
        {

          for (auto s_in_ops = 0; s_in_ops < c_in_ops; s_in_ops += c_ops)
          {
  #pragma HLS pipeline style = stp II = 1

            for (auto s_ops = 0; s_ops < c_ops; s_ops++)
            {
              bool s_pool_write;
              unsigned int s_acc_index = s_och + s_in_ops + s_ops;

              // Initializing the accumulator for each window
              if ((s_fh == 0) && (s_fw == 0)){
                s_acc_buff[s_acc_index] = c_quant;
              }

              // Reading packets of s_in_ops channels
              if (((s_in_ops) == 0) && (s_ops == 0))
              {
                s_input_struct = i_data[0].read();
                s_last = s_input_struct.last;
              }

              // Pooling operation
              if constexpr (c_pool == 0)
              { // Average Pool
                s_acc_buff[s_acc_index] += s_input_struct.data[0][s_in_ops + s_ops];
              }
              else {
                // Max Pool
                if (s_input_struct.data[0][s_in_ops + s_ops] > s_acc_buff[s_acc_index])
                {
                  s_acc_buff[s_acc_index] = s_input_struct.data[0][s_in_ops + s_ops];
                }
              }

              // Writing the output at the end of the window
              if (s_fh == (c_ih - 1) && s_fw == (c_iw - 1))
              {
                t_div s_acc = s_acc_buff[s_acc_index];
                if constexpr (c_pool == 0)
                { // Average Pool
                  s_acc = s_acc / s_divisor;
                }

                s_output_struct.data[0][s_ops] = t_output(s_acc);
                if (s_ops == (c_ops - 1))
                {
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

#endif // NN2FPGA_POOL_STREAM_H_
