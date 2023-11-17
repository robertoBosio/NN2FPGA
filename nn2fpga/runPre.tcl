config_compile -pipeline_style stp -enable_auto_rewind=false
config_interface -m_axi_latency=1
create_clock -period 3.3333
set_param hls.enable_fifo_io_regslice true