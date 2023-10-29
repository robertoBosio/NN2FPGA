// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
 `timescale 1ns/1ps


`define AUTOTB_DUT      two_layers
`define AUTOTB_DUT_INST AESL_inst_two_layers
`define AUTOTB_TOP      apatb_two_layers_top
`define AUTOTB_LAT_RESULT_FILE "two_layers.result.lat.rb"
`define AUTOTB_PER_RESULT_TRANS_FILE "two_layers.performance.result.transaction.xml"
`define AUTOTB_TOP_INST AESL_inst_apatb_two_layers_top
`define AUTOTB_MAX_ALLOW_LATENCY  15000000
`define AUTOTB_CLOCK_PERIOD_DIV2 2.50
`define AUTOTB_II 49164
`define AUTOTB_LATENCY 49184

`define CLOCK_PERIOD_RAZOR 2.40

`define AESL_DEPTH_i_inp_1_V_data_V 1
`define AESL_DEPTH_i_inp_1_V_keep_V 1
`define AESL_DEPTH_i_inp_1_V_strb_V 1
`define AESL_DEPTH_i_inp_1_V_last_V 1
`define AESL_DEPTH_o_outp1_V_data_V 1
`define AESL_DEPTH_o_outp1_V_keep_V 1
`define AESL_DEPTH_o_outp1_V_strb_V 1
`define AESL_DEPTH_o_outp1_V_last_V 1
`define AUTOTB_TVIN_i_inp_1_V_data_V  "./c.two_layers.autotvin_i_inp_1_V_data_V.dat"
`define AUTOTB_TVIN_i_inp_1_V_keep_V  "./c.two_layers.autotvin_i_inp_1_V_keep_V.dat"
`define AUTOTB_TVIN_i_inp_1_V_strb_V  "./c.two_layers.autotvin_i_inp_1_V_strb_V.dat"
`define AUTOTB_TVIN_i_inp_1_V_last_V  "./c.two_layers.autotvin_i_inp_1_V_last_V.dat"
`define AUTOTB_TVIN_i_inp_1_V_data_V_out_wrapc  "./rtl.two_layers.autotvin_i_inp_1_V_data_V.dat"
`define AUTOTB_TVIN_i_inp_1_V_keep_V_out_wrapc  "./rtl.two_layers.autotvin_i_inp_1_V_keep_V.dat"
`define AUTOTB_TVIN_i_inp_1_V_strb_V_out_wrapc  "./rtl.two_layers.autotvin_i_inp_1_V_strb_V.dat"
`define AUTOTB_TVIN_i_inp_1_V_last_V_out_wrapc  "./rtl.two_layers.autotvin_i_inp_1_V_last_V.dat"
`define AUTOTB_TVOUT_o_outp1_V_data_V  "./c.two_layers.autotvout_o_outp1_V_data_V.dat"
`define AUTOTB_TVOUT_o_outp1_V_keep_V  "./c.two_layers.autotvout_o_outp1_V_keep_V.dat"
`define AUTOTB_TVOUT_o_outp1_V_strb_V  "./c.two_layers.autotvout_o_outp1_V_strb_V.dat"
`define AUTOTB_TVOUT_o_outp1_V_last_V  "./c.two_layers.autotvout_o_outp1_V_last_V.dat"
`define AUTOTB_TVOUT_o_outp1_V_data_V_out_wrapc  "./rtl.two_layers.autotvout_o_outp1_V_data_V.dat"
`define AUTOTB_TVOUT_o_outp1_V_keep_V_out_wrapc  "./rtl.two_layers.autotvout_o_outp1_V_keep_V.dat"
`define AUTOTB_TVOUT_o_outp1_V_strb_V_out_wrapc  "./rtl.two_layers.autotvout_o_outp1_V_strb_V.dat"
`define AUTOTB_TVOUT_o_outp1_V_last_V_out_wrapc  "./rtl.two_layers.autotvout_o_outp1_V_last_V.dat"
module `AUTOTB_TOP;



parameter AUTOTB_TRANSACTION_NUM = 2;
parameter PROGRESS_TIMEOUT = 10000000;
parameter LATENCY_ESTIMATION = 49184;
parameter LENGTH_i_inp_1_V_data_V = 384;
parameter LENGTH_i_inp_1_V_keep_V = 384;
parameter LENGTH_i_inp_1_V_last_V = 384;
parameter LENGTH_i_inp_1_V_strb_V = 384;
parameter LENGTH_o_outp1_V_data_V = 10;
parameter LENGTH_o_outp1_V_keep_V = 10;
parameter LENGTH_o_outp1_V_last_V = 10;
parameter LENGTH_o_outp1_V_strb_V = 10;

task read_token;
    input integer fp;
    output reg [151 : 0] token;
    integer ret;
    begin
        token = "";
        ret = 0;
        ret = $fscanf(fp,"%s",token);
    end
endtask

reg AESL_clock;
reg rst;
reg dut_rst;
reg start;
reg ce;
reg tb_continue;
wire AESL_start;
wire AESL_reset;
wire AESL_ce;
wire AESL_ready;
wire AESL_idle;
wire AESL_continue;
reg AESL_done = 0;
reg AESL_done_delay = 0;
reg AESL_done_delay2 = 0;
reg AESL_ready_delay = 0;
wire ready;
wire ready_wire;
wire [63 : 0] i_inp_1_TDATA;
wire [7 : 0] i_inp_1_TKEEP;
wire [7 : 0] i_inp_1_TSTRB;
wire [0 : 0] i_inp_1_TLAST;
wire [7 : 0] o_outp1_TDATA;
wire [0 : 0] o_outp1_TKEEP;
wire [0 : 0] o_outp1_TSTRB;
wire [0 : 0] o_outp1_TLAST;
wire  i_inp_1_TVALID;
wire  i_inp_1_TREADY;
wire  o_outp1_TVALID;
wire  o_outp1_TREADY;
reg [31:0] ap_c_n_tvin_trans_num_i_inp_1_V_data_V;
reg [31:0] ap_c_n_tvin_trans_num_i_inp_1_V_keep_V;
reg [31:0] ap_c_n_tvin_trans_num_i_inp_1_V_strb_V;
reg [31:0] ap_c_n_tvin_trans_num_i_inp_1_V_last_V;
reg [31:0] ap_c_n_tvout_trans_num_o_outp1_V_data_V;
reg [31:0] ap_c_n_tvout_trans_num_o_outp1_V_keep_V;
reg [31:0] ap_c_n_tvout_trans_num_o_outp1_V_strb_V;
reg [31:0] ap_c_n_tvout_trans_num_o_outp1_V_last_V;
integer done_cnt = 0;
integer AESL_ready_cnt = 0;
integer ready_cnt = 0;
reg ready_initial;
reg ready_initial_n;
reg ready_last_n;
reg ready_delay_last_n;
reg done_delay_last_n;
reg interface_done = 0;


wire ap_clk;
wire ap_rst_n;
wire ap_rst_n_n;
wire clk1;
wire clk2;
wire clk1_rst_n;
reg clk1_tb;
reg clk2_tb = 0;
reg clk1_rst_n_tb = 0;
wire error_sim;

`AUTOTB_DUT `AUTOTB_DUT_INST(
    .i_inp_1_TDATA(i_inp_1_TDATA),
    .i_inp_1_TKEEP(i_inp_1_TKEEP),
    .i_inp_1_TSTRB(i_inp_1_TSTRB),
    .i_inp_1_TLAST(i_inp_1_TLAST),
    .o_outp1_TDATA(o_outp1_TDATA),
    .o_outp1_TKEEP(o_outp1_TKEEP),
    .o_outp1_TSTRB(o_outp1_TSTRB),
    .o_outp1_TLAST(o_outp1_TLAST),
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .clk1(clk1),
    .clk2(clk2),
    .clk1_rst_n(clk1_rst_n),
    .error_sim(error_sim),
    .i_inp_1_TVALID(i_inp_1_TVALID),
    .i_inp_1_TREADY(i_inp_1_TREADY),
    .o_outp1_TVALID(o_outp1_TVALID),
    .o_outp1_TREADY(o_outp1_TREADY));

// Assignment for control signal
assign ap_clk = AESL_clock;
assign ap_rst_n = dut_rst;
assign ap_rst_n_n = ~dut_rst;
assign clk1 = clk1_tb;
assign clk2 = clk2_tb;
assign clk1_rst_n = clk1_rst_n_tb;
assign AESL_reset = rst;
assign AESL_start = start;
assign AESL_ce = ce;
assign AESL_continue = tb_continue;
assign AESL_ready = ready;


wire i_inp_1_ready;
wire i_inp_1_done;
wire [31:0] i_inp_1_transaction;
wire axi_s_i_inp_1_TVALID;
wire axi_s_i_inp_1_TREADY;

AESL_axi_s_i_inp_1 AESL_AXI_S_i_inp_1(
    .clk(AESL_clock),
    .reset(AESL_reset),
    .TRAN_i_inp_1_TDATA(i_inp_1_TDATA),
    .TRAN_i_inp_1_TKEEP(i_inp_1_TKEEP),
    .TRAN_i_inp_1_TSTRB(i_inp_1_TSTRB),
    .TRAN_i_inp_1_TLAST(i_inp_1_TLAST),
    .TRAN_i_inp_1_TVALID(i_inp_1_TVALID),
    .TRAN_i_inp_1_TREADY(i_inp_1_TREADY),
    .ready(i_inp_1_ready),
    .done(i_inp_1_done),
    .transaction(i_inp_1_transaction));

assign i_inp_1_ready = ready;
assign i_inp_1_done = 0;
wire o_outp1_ready;
wire o_outp1_done;
wire [31:0] o_outp1_transaction;
wire axi_s_o_outp1_TVALID;
wire axi_s_o_outp1_TREADY;

AESL_axi_s_o_outp1 AESL_AXI_S_o_outp1(
    .clk(AESL_clock),
    .reset(AESL_reset),
    .TRAN_o_outp1_TDATA(o_outp1_TDATA),
    .TRAN_o_outp1_TKEEP(o_outp1_TKEEP),
    .TRAN_o_outp1_TSTRB(o_outp1_TSTRB),
    .TRAN_o_outp1_TLAST(o_outp1_TLAST),
    .TRAN_o_outp1_TVALID(o_outp1_TVALID),
    .TRAN_o_outp1_TREADY(o_outp1_TREADY),
    .ready(o_outp1_ready),
    .done(o_outp1_done),
    .transaction(o_outp1_transaction));

assign o_outp1_ready = 0;
assign o_outp1_done = AESL_done;

initial begin : generate_AESL_ready_cnt_proc
    AESL_ready_cnt = 0;
    wait(AESL_reset === 1);
    while(AESL_ready_cnt != AUTOTB_TRANSACTION_NUM) begin
        while(AESL_ready !== 1) begin
            @(posedge AESL_clock);
            # 0.4;
        end
        @(negedge AESL_clock);
        AESL_ready_cnt = AESL_ready_cnt + 1;
        @(posedge AESL_clock);
        # 0.4;
    end
end

    event next_trigger_ready_cnt;
    
    initial begin : gen_ready_cnt
        ready_cnt = 0;
        wait (AESL_reset === 1);
        forever begin
            @ (posedge AESL_clock);
            if (ready == 1) begin
                if (ready_cnt < AUTOTB_TRANSACTION_NUM) begin
                    ready_cnt = ready_cnt + 1;
                end
            end
            -> next_trigger_ready_cnt;
        end
    end
    
    wire all_finish = (done_cnt == AUTOTB_TRANSACTION_NUM);
    
    // done_cnt
    always @ (posedge AESL_clock) begin
        if (~AESL_reset) begin
            done_cnt <= 0;
        end else begin
            if (AESL_done == 1) begin
                if (done_cnt < AUTOTB_TRANSACTION_NUM) begin
                    done_cnt <= done_cnt + 1;
                end
            end
        end
    end
    
    initial begin : finish_simulation
        wait (all_finish == 1);
        // last transaction is saved at negedge right after last done
        repeat(6) @ (posedge AESL_clock);
        $finish;
    end
    
initial begin
    AESL_clock = 0;
    forever #`AUTOTB_CLOCK_PERIOD_DIV2 AESL_clock = ~AESL_clock;
end

initial begin : clock1
    clk1_tb = 0;
    forever #`CLOCK_PERIOD_RAZOR clk1_tb = ~clk1_tb;
end

initial begin : clock2  
    wait(clk1_tb === 1);
    #`CLOCK_PERIOD_RAZOR;
    clk2_tb = 1;
    forever #`CLOCK_PERIOD_RAZOR clk2_tb = ~clk2_tb;
end

reg end_i_inp_1_V_data_V;
reg [31:0] size_i_inp_1_V_data_V;
reg [31:0] size_i_inp_1_V_data_V_backup;
reg end_i_inp_1_V_keep_V;
reg [31:0] size_i_inp_1_V_keep_V;
reg [31:0] size_i_inp_1_V_keep_V_backup;
reg end_i_inp_1_V_strb_V;
reg [31:0] size_i_inp_1_V_strb_V;
reg [31:0] size_i_inp_1_V_strb_V_backup;
reg end_i_inp_1_V_last_V;
reg [31:0] size_i_inp_1_V_last_V;
reg [31:0] size_i_inp_1_V_last_V_backup;
reg end_o_outp1_V_data_V;
reg [31:0] size_o_outp1_V_data_V;
reg [31:0] size_o_outp1_V_data_V_backup;
reg end_o_outp1_V_keep_V;
reg [31:0] size_o_outp1_V_keep_V;
reg [31:0] size_o_outp1_V_keep_V_backup;
reg end_o_outp1_V_strb_V;
reg [31:0] size_o_outp1_V_strb_V;
reg [31:0] size_o_outp1_V_strb_V_backup;
reg end_o_outp1_V_last_V;
reg [31:0] size_o_outp1_V_last_V;
reg [31:0] size_o_outp1_V_last_V_backup;

initial begin : initial_process_clk1_rst
    integer proc_rand;
    clk1_rst_n_tb = 0;
    # 100;
    repeat(3) @ (posedge clk1_tb);
    clk1_rst_n_tb = 1;
end

initial begin : initial_process
    integer proc_rand;
    rst = 0;
    # 100;
    repeat(0+3) @ (posedge AESL_clock);
    rst = 1;
end
initial begin : initial_process_for_dut_rst
    integer proc_rand;
    dut_rst = 0;
    # 100;
    repeat(3) @ (posedge AESL_clock);
    dut_rst = 1;
end
    // i_inp_1_V_data_V : axi_s
    reg ready_i_inp_1_V_data_V;
    
    always @ (*) begin
        if (end_i_inp_1_V_data_V) begin
            ready_i_inp_1_V_data_V <= 1;
        end else if (ap_c_n_tvin_trans_num_i_inp_1_V_data_V > ready_cnt) begin
            ready_i_inp_1_V_data_V <= 1;
        end else begin
            ready_i_inp_1_V_data_V <= 0;
        end
    end
    
    // start
    always @ (*) begin
        if (~AESL_reset) begin
            start <= 0;
        end else begin
            start <= ready_i_inp_1_V_data_V;
        end
    end

always @(AESL_done)
begin
    tb_continue = AESL_done;
end

initial begin : ready_initial_process
    ready_initial = 0;
    wait(AESL_reset === 1);
    ready_initial = 1;
    @(posedge AESL_clock);
    ready_initial = 0;
end

initial begin : ready_last_n_process
  ready_last_n = 1;
  wait(ready_cnt == AUTOTB_TRANSACTION_NUM)
  @(posedge AESL_clock);
  ready_last_n <= 0;
end

assign ready = AESL_start;
assign ready_wire = ready;
initial begin : done_delay_last_n_process
  done_delay_last_n = 1;
  while(done_cnt < AUTOTB_TRANSACTION_NUM)
      @(posedge AESL_clock);
  # 0.1;
  done_delay_last_n = 0;
end

always @(posedge AESL_clock)
begin
    if(AESL_reset === 0)
  begin
      AESL_done_delay <= 0;
      AESL_done_delay2 <= 0;
  end
  else begin
      AESL_done_delay <= AESL_done & done_delay_last_n;
      AESL_done_delay2 <= AESL_done_delay;
  end
end
always @(posedge AESL_clock)
begin
    if(AESL_reset === 0)
      interface_done = 0;
  else begin
      # 0.01;
      if(ready === 1 && ready_cnt > 0 && ready_cnt < AUTOTB_TRANSACTION_NUM)
          interface_done = 1;
      else if(AESL_done_delay === 1 && done_cnt == AUTOTB_TRANSACTION_NUM)
          interface_done = 1;
      else
          interface_done = 0;
  end
end
    // o_outp1_V_data_V : axi_s
    reg done_o_outp1_V_data_V;
    
    always @ (*) begin
        if (end_o_outp1_V_data_V) begin
            done_o_outp1_V_data_V <= 1;
        end else if (ap_c_n_tvout_trans_num_o_outp1_V_data_V > done_cnt + 1) begin
            done_o_outp1_V_data_V <= 1;
        end else if (size_o_outp1_V_data_V > 1) begin
            done_o_outp1_V_data_V <= 0;
        end else if (o_outp1_TVALID == 1) begin
            done_o_outp1_V_data_V <= 1;
        end else begin
            done_o_outp1_V_data_V <= 0;
        end
    end
    
    // o_outp1_V_keep_V : axi_s
    reg done_o_outp1_V_keep_V;
    
    always @ (*) begin
        if (end_o_outp1_V_keep_V) begin
            done_o_outp1_V_keep_V <= 1;
        end else if (ap_c_n_tvout_trans_num_o_outp1_V_keep_V > done_cnt + 1) begin
            done_o_outp1_V_keep_V <= 1;
        end else if (size_o_outp1_V_keep_V > 1) begin
            done_o_outp1_V_keep_V <= 0;
        end else if (o_outp1_TVALID == 1) begin
            done_o_outp1_V_keep_V <= 1;
        end else begin
            done_o_outp1_V_keep_V <= 0;
        end
    end
    
    // o_outp1_V_strb_V : axi_s
    reg done_o_outp1_V_strb_V;
    
    always @ (*) begin
        if (end_o_outp1_V_strb_V) begin
            done_o_outp1_V_strb_V <= 1;
        end else if (ap_c_n_tvout_trans_num_o_outp1_V_strb_V > done_cnt + 1) begin
            done_o_outp1_V_strb_V <= 1;
        end else if (size_o_outp1_V_strb_V > 1) begin
            done_o_outp1_V_strb_V <= 0;
        end else if (o_outp1_TVALID == 1) begin
            done_o_outp1_V_strb_V <= 1;
        end else begin
            done_o_outp1_V_strb_V <= 0;
        end
    end
    
    // o_outp1_V_last_V : axi_s
    reg done_o_outp1_V_last_V;
    
    always @ (*) begin
        if (end_o_outp1_V_last_V) begin
            done_o_outp1_V_last_V <= 1;
        end else if (ap_c_n_tvout_trans_num_o_outp1_V_last_V > done_cnt + 1) begin
            done_o_outp1_V_last_V <= 1;
        end else if (size_o_outp1_V_last_V > 1) begin
            done_o_outp1_V_last_V <= 0;
        end else if (o_outp1_TVALID == 1) begin
            done_o_outp1_V_last_V <= 1;
        end else begin
            done_o_outp1_V_last_V <= 0;
        end
    end
    
    // AESL_done
    always @ (*) begin
        if (~AESL_reset) begin
            AESL_done <= 0;
        end else begin
            AESL_done <= done_o_outp1_V_data_V && done_o_outp1_V_keep_V && done_o_outp1_V_strb_V && done_o_outp1_V_last_V;
        end
    end
    
    `define STREAM_SIZE_IN_i_inp_1_V_data_V "./stream_size_in_i_inp_1_V_data_V.dat"
    
    initial begin : gen_ap_c_n_tvin_trans_num_i_inp_1_V_data_V
        integer fp_i_inp_1_V_data_V;
        reg [127:0] token_i_inp_1_V_data_V;
        integer ret;
        
        ap_c_n_tvin_trans_num_i_inp_1_V_data_V = 0;
        end_i_inp_1_V_data_V = 0;
        wait (AESL_reset === 1);
        
        fp_i_inp_1_V_data_V = $fopen(`STREAM_SIZE_IN_i_inp_1_V_data_V, "r");
        if(fp_i_inp_1_V_data_V == 0) begin
            $display("Failed to open file \"%s\"!", `STREAM_SIZE_IN_i_inp_1_V_data_V);
            $finish;
        end
        read_token(fp_i_inp_1_V_data_V, token_i_inp_1_V_data_V); // should be [[[runtime]]]
        if (token_i_inp_1_V_data_V != "[[[runtime]]]") begin
            $display("ERROR: token_i_inp_1_V_data_V != \"[[[runtime]]]\"");
            $finish;
        end
        size_i_inp_1_V_data_V = 0;
        size_i_inp_1_V_data_V_backup = 0;
        while (size_i_inp_1_V_data_V == 0 && end_i_inp_1_V_data_V == 0) begin
            ap_c_n_tvin_trans_num_i_inp_1_V_data_V = ap_c_n_tvin_trans_num_i_inp_1_V_data_V + 1;
            read_token(fp_i_inp_1_V_data_V, token_i_inp_1_V_data_V); // should be [[transaction]] or [[[/runtime]]]
            if (token_i_inp_1_V_data_V == "[[transaction]]") begin
                read_token(fp_i_inp_1_V_data_V, token_i_inp_1_V_data_V); // should be transaction number
                read_token(fp_i_inp_1_V_data_V, token_i_inp_1_V_data_V); // should be size for hls::stream
                ret = $sscanf(token_i_inp_1_V_data_V, "%d", size_i_inp_1_V_data_V);
                if (size_i_inp_1_V_data_V > 0) begin
                    size_i_inp_1_V_data_V_backup = size_i_inp_1_V_data_V;
                end
                read_token(fp_i_inp_1_V_data_V, token_i_inp_1_V_data_V); // should be [[/transaction]]
            end else if (token_i_inp_1_V_data_V == "[[[/runtime]]]") begin
                $fclose(fp_i_inp_1_V_data_V);
                end_i_inp_1_V_data_V = 1;
            end else begin
                $display("ERROR: unknown token_i_inp_1_V_data_V");
                $finish;
            end
        end
        forever begin
            @ (posedge AESL_clock);
            if (end_i_inp_1_V_data_V == 0) begin
                if ((i_inp_1_TREADY & i_inp_1_TVALID) == 1) begin
                    if (size_i_inp_1_V_data_V > 0) begin
                        size_i_inp_1_V_data_V = size_i_inp_1_V_data_V - 1;
                        while (size_i_inp_1_V_data_V == 0 && end_i_inp_1_V_data_V == 0) begin
                            ap_c_n_tvin_trans_num_i_inp_1_V_data_V = ap_c_n_tvin_trans_num_i_inp_1_V_data_V + 1;
                            read_token(fp_i_inp_1_V_data_V, token_i_inp_1_V_data_V); // should be [[transaction]] or [[[/runtime]]]
                            if (token_i_inp_1_V_data_V == "[[transaction]]") begin
                                read_token(fp_i_inp_1_V_data_V, token_i_inp_1_V_data_V); // should be transaction number
                                read_token(fp_i_inp_1_V_data_V, token_i_inp_1_V_data_V); // should be size for hls::stream
                                ret = $sscanf(token_i_inp_1_V_data_V, "%d", size_i_inp_1_V_data_V);
                                if (size_i_inp_1_V_data_V > 0) begin
                                    size_i_inp_1_V_data_V_backup = size_i_inp_1_V_data_V;
                                end
                                read_token(fp_i_inp_1_V_data_V, token_i_inp_1_V_data_V); // should be [[/transaction]]
                            end else if (token_i_inp_1_V_data_V == "[[[/runtime]]]") begin
                                size_i_inp_1_V_data_V = size_i_inp_1_V_data_V_backup;
                                $fclose(fp_i_inp_1_V_data_V);
                                end_i_inp_1_V_data_V = 1;
                            end else begin
                                $display("ERROR: unknown token_i_inp_1_V_data_V");
                                $finish;
                            end
                        end
                    end
                end
            end else begin
                if ((i_inp_1_TREADY & i_inp_1_TVALID) == 1) begin
                    if (size_i_inp_1_V_data_V > 0) begin
                        size_i_inp_1_V_data_V = size_i_inp_1_V_data_V - 1;
                        if (size_i_inp_1_V_data_V == 0) begin
                            ap_c_n_tvin_trans_num_i_inp_1_V_data_V = ap_c_n_tvin_trans_num_i_inp_1_V_data_V + 1;
                            size_i_inp_1_V_data_V = size_i_inp_1_V_data_V_backup;
                        end
                    end
                end
            end
        end
    end
    
    `define STREAM_SIZE_OUT_o_outp1_V_data_V "./stream_size_out_o_outp1_V_data_V.dat"
    
    initial begin : proc_ap_c_n_tvout_trans_num_o_outp1_V_data_V
        integer fp_o_outp1_V_data_V;
        reg [127:0] token_o_outp1_V_data_V;
        integer ret;
        
        ap_c_n_tvout_trans_num_o_outp1_V_data_V = 0;
        end_o_outp1_V_data_V = 0;
        wait (AESL_reset === 1);
        
        fp_o_outp1_V_data_V = $fopen(`STREAM_SIZE_OUT_o_outp1_V_data_V, "r");
        if (fp_o_outp1_V_data_V == 0) begin // Failed to open file
            $display("Failed to open size file for \"%s\"!", `STREAM_SIZE_OUT_o_outp1_V_data_V);
            $finish;
        end
        read_token(fp_o_outp1_V_data_V, token_o_outp1_V_data_V);
        if (token_o_outp1_V_data_V != "[[[runtime]]]") begin
            $display("ERROR: token_o_outp1_V_data_V != \"[[[runtime]]]\"");
            $finish;
        end
        
        size_o_outp1_V_data_V = 0;
        while (size_o_outp1_V_data_V == 0 && end_o_outp1_V_data_V == 0) begin
            ap_c_n_tvout_trans_num_o_outp1_V_data_V = ap_c_n_tvout_trans_num_o_outp1_V_data_V + 1;
            read_token(fp_o_outp1_V_data_V, token_o_outp1_V_data_V); // should be [[transaction]] or [[[/runtime]]]
            if (token_o_outp1_V_data_V == "[[transaction]]") begin
                read_token(fp_o_outp1_V_data_V, token_o_outp1_V_data_V); // should be transaction number
                read_token(fp_o_outp1_V_data_V, token_o_outp1_V_data_V); // should be size for hls::stream
                ret = $sscanf(token_o_outp1_V_data_V, "%d", size_o_outp1_V_data_V);
                read_token(fp_o_outp1_V_data_V, token_o_outp1_V_data_V); // should be [[/transaction]]
            end else if (token_o_outp1_V_data_V == "[[[/runtime]]]") begin
                $fclose(fp_o_outp1_V_data_V);
                end_o_outp1_V_data_V = 1;
            end else begin
                $display("ERROR: unknown token_o_outp1_V_data_V");
                $finish;
            end
        end
        forever begin
            @ (posedge AESL_clock);
            if (end_o_outp1_V_data_V == 0) begin 
                if (o_outp1_TVALID == 1) begin
                    if (size_o_outp1_V_data_V > 0) begin
                        size_o_outp1_V_data_V = size_o_outp1_V_data_V - 1;
                        while (size_o_outp1_V_data_V == 0 && end_o_outp1_V_data_V == 0) begin
                            ap_c_n_tvout_trans_num_o_outp1_V_data_V = ap_c_n_tvout_trans_num_o_outp1_V_data_V + 1;
                            read_token(fp_o_outp1_V_data_V, token_o_outp1_V_data_V); // should be [[transaction]] or [[[/runtime]]]
                            if (token_o_outp1_V_data_V == "[[transaction]]") begin
                                read_token(fp_o_outp1_V_data_V, token_o_outp1_V_data_V); // should be transaction number
                                read_token(fp_o_outp1_V_data_V, token_o_outp1_V_data_V); // should be size for hls::stream
                                ret = $sscanf(token_o_outp1_V_data_V, "%d", size_o_outp1_V_data_V);
                                read_token(fp_o_outp1_V_data_V, token_o_outp1_V_data_V); // should be [[/transaction]]
                            end else if (token_o_outp1_V_data_V == "[[[/runtime]]]") begin
                                $fclose(fp_o_outp1_V_data_V);
                                end_o_outp1_V_data_V = 1;
                            end else begin
                                $display("ERROR: unknown token_o_outp1_V_data_V");
                                $finish;
                            end
                        end
                    end
                end
            end
        end
    end
    `define STREAM_SIZE_OUT_o_outp1_V_keep_V "./stream_size_out_o_outp1_V_keep_V.dat"
    
    initial begin : proc_ap_c_n_tvout_trans_num_o_outp1_V_keep_V
        integer fp_o_outp1_V_keep_V;
        reg [127:0] token_o_outp1_V_keep_V;
        integer ret;
        
        ap_c_n_tvout_trans_num_o_outp1_V_keep_V = 0;
        end_o_outp1_V_keep_V = 0;
        wait (AESL_reset === 1);
        
        fp_o_outp1_V_keep_V = $fopen(`STREAM_SIZE_OUT_o_outp1_V_keep_V, "r");
        if (fp_o_outp1_V_keep_V == 0) begin // Failed to open file
            $display("Failed to open size file for \"%s\"!", `STREAM_SIZE_OUT_o_outp1_V_keep_V);
            $finish;
        end
        read_token(fp_o_outp1_V_keep_V, token_o_outp1_V_keep_V);
        if (token_o_outp1_V_keep_V != "[[[runtime]]]") begin
            $display("ERROR: token_o_outp1_V_keep_V != \"[[[runtime]]]\"");
            $finish;
        end
        
        size_o_outp1_V_keep_V = 0;
        while (size_o_outp1_V_keep_V == 0 && end_o_outp1_V_keep_V == 0) begin
            ap_c_n_tvout_trans_num_o_outp1_V_keep_V = ap_c_n_tvout_trans_num_o_outp1_V_keep_V + 1;
            read_token(fp_o_outp1_V_keep_V, token_o_outp1_V_keep_V); // should be [[transaction]] or [[[/runtime]]]
            if (token_o_outp1_V_keep_V == "[[transaction]]") begin
                read_token(fp_o_outp1_V_keep_V, token_o_outp1_V_keep_V); // should be transaction number
                read_token(fp_o_outp1_V_keep_V, token_o_outp1_V_keep_V); // should be size for hls::stream
                ret = $sscanf(token_o_outp1_V_keep_V, "%d", size_o_outp1_V_keep_V);
                read_token(fp_o_outp1_V_keep_V, token_o_outp1_V_keep_V); // should be [[/transaction]]
            end else if (token_o_outp1_V_keep_V == "[[[/runtime]]]") begin
                $fclose(fp_o_outp1_V_keep_V);
                end_o_outp1_V_keep_V = 1;
            end else begin
                $display("ERROR: unknown token_o_outp1_V_keep_V");
                $finish;
            end
        end
        forever begin
            @ (posedge AESL_clock);
            if (end_o_outp1_V_keep_V == 0) begin 
                if (o_outp1_TVALID == 1) begin
                    if (size_o_outp1_V_keep_V > 0) begin
                        size_o_outp1_V_keep_V = size_o_outp1_V_keep_V - 1;
                        while (size_o_outp1_V_keep_V == 0 && end_o_outp1_V_keep_V == 0) begin
                            ap_c_n_tvout_trans_num_o_outp1_V_keep_V = ap_c_n_tvout_trans_num_o_outp1_V_keep_V + 1;
                            read_token(fp_o_outp1_V_keep_V, token_o_outp1_V_keep_V); // should be [[transaction]] or [[[/runtime]]]
                            if (token_o_outp1_V_keep_V == "[[transaction]]") begin
                                read_token(fp_o_outp1_V_keep_V, token_o_outp1_V_keep_V); // should be transaction number
                                read_token(fp_o_outp1_V_keep_V, token_o_outp1_V_keep_V); // should be size for hls::stream
                                ret = $sscanf(token_o_outp1_V_keep_V, "%d", size_o_outp1_V_keep_V);
                                read_token(fp_o_outp1_V_keep_V, token_o_outp1_V_keep_V); // should be [[/transaction]]
                            end else if (token_o_outp1_V_keep_V == "[[[/runtime]]]") begin
                                $fclose(fp_o_outp1_V_keep_V);
                                end_o_outp1_V_keep_V = 1;
                            end else begin
                                $display("ERROR: unknown token_o_outp1_V_keep_V");
                                $finish;
                            end
                        end
                    end
                end
            end
        end
    end
    `define STREAM_SIZE_OUT_o_outp1_V_strb_V "./stream_size_out_o_outp1_V_strb_V.dat"
    
    initial begin : proc_ap_c_n_tvout_trans_num_o_outp1_V_strb_V
        integer fp_o_outp1_V_strb_V;
        reg [127:0] token_o_outp1_V_strb_V;
        integer ret;
        
        ap_c_n_tvout_trans_num_o_outp1_V_strb_V = 0;
        end_o_outp1_V_strb_V = 0;
        wait (AESL_reset === 1);
        
        fp_o_outp1_V_strb_V = $fopen(`STREAM_SIZE_OUT_o_outp1_V_strb_V, "r");
        if (fp_o_outp1_V_strb_V == 0) begin // Failed to open file
            $display("Failed to open size file for \"%s\"!", `STREAM_SIZE_OUT_o_outp1_V_strb_V);
            $finish;
        end
        read_token(fp_o_outp1_V_strb_V, token_o_outp1_V_strb_V);
        if (token_o_outp1_V_strb_V != "[[[runtime]]]") begin
            $display("ERROR: token_o_outp1_V_strb_V != \"[[[runtime]]]\"");
            $finish;
        end
        
        size_o_outp1_V_strb_V = 0;
        while (size_o_outp1_V_strb_V == 0 && end_o_outp1_V_strb_V == 0) begin
            ap_c_n_tvout_trans_num_o_outp1_V_strb_V = ap_c_n_tvout_trans_num_o_outp1_V_strb_V + 1;
            read_token(fp_o_outp1_V_strb_V, token_o_outp1_V_strb_V); // should be [[transaction]] or [[[/runtime]]]
            if (token_o_outp1_V_strb_V == "[[transaction]]") begin
                read_token(fp_o_outp1_V_strb_V, token_o_outp1_V_strb_V); // should be transaction number
                read_token(fp_o_outp1_V_strb_V, token_o_outp1_V_strb_V); // should be size for hls::stream
                ret = $sscanf(token_o_outp1_V_strb_V, "%d", size_o_outp1_V_strb_V);
                read_token(fp_o_outp1_V_strb_V, token_o_outp1_V_strb_V); // should be [[/transaction]]
            end else if (token_o_outp1_V_strb_V == "[[[/runtime]]]") begin
                $fclose(fp_o_outp1_V_strb_V);
                end_o_outp1_V_strb_V = 1;
            end else begin
                $display("ERROR: unknown token_o_outp1_V_strb_V");
                $finish;
            end
        end
        forever begin
            @ (posedge AESL_clock);
            if (end_o_outp1_V_strb_V == 0) begin 
                if (o_outp1_TVALID == 1) begin
                    if (size_o_outp1_V_strb_V > 0) begin
                        size_o_outp1_V_strb_V = size_o_outp1_V_strb_V - 1;
                        while (size_o_outp1_V_strb_V == 0 && end_o_outp1_V_strb_V == 0) begin
                            ap_c_n_tvout_trans_num_o_outp1_V_strb_V = ap_c_n_tvout_trans_num_o_outp1_V_strb_V + 1;
                            read_token(fp_o_outp1_V_strb_V, token_o_outp1_V_strb_V); // should be [[transaction]] or [[[/runtime]]]
                            if (token_o_outp1_V_strb_V == "[[transaction]]") begin
                                read_token(fp_o_outp1_V_strb_V, token_o_outp1_V_strb_V); // should be transaction number
                                read_token(fp_o_outp1_V_strb_V, token_o_outp1_V_strb_V); // should be size for hls::stream
                                ret = $sscanf(token_o_outp1_V_strb_V, "%d", size_o_outp1_V_strb_V);
                                read_token(fp_o_outp1_V_strb_V, token_o_outp1_V_strb_V); // should be [[/transaction]]
                            end else if (token_o_outp1_V_strb_V == "[[[/runtime]]]") begin
                                $fclose(fp_o_outp1_V_strb_V);
                                end_o_outp1_V_strb_V = 1;
                            end else begin
                                $display("ERROR: unknown token_o_outp1_V_strb_V");
                                $finish;
                            end
                        end
                    end
                end
            end
        end
    end
    `define STREAM_SIZE_OUT_o_outp1_V_last_V "./stream_size_out_o_outp1_V_last_V.dat"
    
    initial begin : proc_ap_c_n_tvout_trans_num_o_outp1_V_last_V
        integer fp_o_outp1_V_last_V;
        reg [127:0] token_o_outp1_V_last_V;
        integer ret;
        
        ap_c_n_tvout_trans_num_o_outp1_V_last_V = 0;
        end_o_outp1_V_last_V = 0;
        wait (AESL_reset === 1);
        
        fp_o_outp1_V_last_V = $fopen(`STREAM_SIZE_OUT_o_outp1_V_last_V, "r");
        if (fp_o_outp1_V_last_V == 0) begin // Failed to open file
            $display("Failed to open size file for \"%s\"!", `STREAM_SIZE_OUT_o_outp1_V_last_V);
            $finish;
        end
        read_token(fp_o_outp1_V_last_V, token_o_outp1_V_last_V);
        if (token_o_outp1_V_last_V != "[[[runtime]]]") begin
            $display("ERROR: token_o_outp1_V_last_V != \"[[[runtime]]]\"");
            $finish;
        end
        
        size_o_outp1_V_last_V = 0;
        while (size_o_outp1_V_last_V == 0 && end_o_outp1_V_last_V == 0) begin
            ap_c_n_tvout_trans_num_o_outp1_V_last_V = ap_c_n_tvout_trans_num_o_outp1_V_last_V + 1;
            read_token(fp_o_outp1_V_last_V, token_o_outp1_V_last_V); // should be [[transaction]] or [[[/runtime]]]
            if (token_o_outp1_V_last_V == "[[transaction]]") begin
                read_token(fp_o_outp1_V_last_V, token_o_outp1_V_last_V); // should be transaction number
                read_token(fp_o_outp1_V_last_V, token_o_outp1_V_last_V); // should be size for hls::stream
                ret = $sscanf(token_o_outp1_V_last_V, "%d", size_o_outp1_V_last_V);
                read_token(fp_o_outp1_V_last_V, token_o_outp1_V_last_V); // should be [[/transaction]]
            end else if (token_o_outp1_V_last_V == "[[[/runtime]]]") begin
                $fclose(fp_o_outp1_V_last_V);
                end_o_outp1_V_last_V = 1;
            end else begin
                $display("ERROR: unknown token_o_outp1_V_last_V");
                $finish;
            end
        end
        forever begin
            @ (posedge AESL_clock);
            if (end_o_outp1_V_last_V == 0) begin 
                if (o_outp1_TVALID == 1) begin
                    if (size_o_outp1_V_last_V > 0) begin
                        size_o_outp1_V_last_V = size_o_outp1_V_last_V - 1;
                        while (size_o_outp1_V_last_V == 0 && end_o_outp1_V_last_V == 0) begin
                            ap_c_n_tvout_trans_num_o_outp1_V_last_V = ap_c_n_tvout_trans_num_o_outp1_V_last_V + 1;
                            read_token(fp_o_outp1_V_last_V, token_o_outp1_V_last_V); // should be [[transaction]] or [[[/runtime]]]
                            if (token_o_outp1_V_last_V == "[[transaction]]") begin
                                read_token(fp_o_outp1_V_last_V, token_o_outp1_V_last_V); // should be transaction number
                                read_token(fp_o_outp1_V_last_V, token_o_outp1_V_last_V); // should be size for hls::stream
                                ret = $sscanf(token_o_outp1_V_last_V, "%d", size_o_outp1_V_last_V);
                                read_token(fp_o_outp1_V_last_V, token_o_outp1_V_last_V); // should be [[/transaction]]
                            end else if (token_o_outp1_V_last_V == "[[[/runtime]]]") begin
                                $fclose(fp_o_outp1_V_last_V);
                                end_o_outp1_V_last_V = 1;
                            end else begin
                                $display("ERROR: unknown token_o_outp1_V_last_V");
                                $finish;
                            end
                        end
                    end
                end
            end
        end
    end
task write_binary;
    input integer fp;
    input reg[64-1:0] in;
    input integer in_bw;
    reg [63:0] tmp_long;
    reg[64-1:0] local_in;
    integer char_num;
    integer long_num;
    integer i;
    integer j;
    begin
        long_num = (in_bw + 63) / 64;
        char_num = ((in_bw - 1) % 64 + 7) / 8;
        for(i=long_num;i>0;i=i-1) begin
             local_in = in;
             tmp_long = local_in >> ((i-1)*64);
             for(j=0;j<64;j=j+1)
                 if (tmp_long[j] === 1'bx)
                     tmp_long[j] = 1'b0;
             if (i == long_num) begin
                 case(char_num)
                     1: $fwrite(fp,"%c",tmp_long[7:0]);
                     2: $fwrite(fp,"%c%c",tmp_long[15:8],tmp_long[7:0]);
                     3: $fwrite(fp,"%c%c%c",tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     4: $fwrite(fp,"%c%c%c%c",tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     5: $fwrite(fp,"%c%c%c%c%c",tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     6: $fwrite(fp,"%c%c%c%c%c%c",tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     7: $fwrite(fp,"%c%c%c%c%c%c%c",tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     8: $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     default: ;
                 endcase
             end
             else begin
                 $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
             end
        end
    end
endtask;

reg dump_tvout_finish_o_outp1_V_data_V;

initial begin : dump_tvout_runtime_sign_o_outp1_V_data_V
    integer fp;
    dump_tvout_finish_o_outp1_V_data_V = 0;
    fp = $fopen(`AUTOTB_TVOUT_o_outp1_V_data_V_out_wrapc, "w");
    if (fp == 0) begin
        $display("Failed to open file \"%s\"!", `AUTOTB_TVOUT_o_outp1_V_data_V_out_wrapc);
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end
    $fdisplay(fp,"[[[runtime]]]");
    $fclose(fp);
    wait (done_cnt == AUTOTB_TRANSACTION_NUM);
    // last transaction is saved at negedge right after last done
    repeat(5) @ (posedge AESL_clock);
    fp = $fopen(`AUTOTB_TVOUT_o_outp1_V_data_V_out_wrapc, "a");
    if (fp == 0) begin
        $display("Failed to open file \"%s\"!", `AUTOTB_TVOUT_o_outp1_V_data_V_out_wrapc);
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end
    $fdisplay(fp,"[[[/runtime]]]");
    $fclose(fp);
    dump_tvout_finish_o_outp1_V_data_V = 1;
end


reg dump_tvout_finish_o_outp1_V_keep_V;

initial begin : dump_tvout_runtime_sign_o_outp1_V_keep_V
    integer fp;
    dump_tvout_finish_o_outp1_V_keep_V = 0;
    fp = $fopen(`AUTOTB_TVOUT_o_outp1_V_keep_V_out_wrapc, "w");
    if (fp == 0) begin
        $display("Failed to open file \"%s\"!", `AUTOTB_TVOUT_o_outp1_V_keep_V_out_wrapc);
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end
    $fdisplay(fp,"[[[runtime]]]");
    $fclose(fp);
    wait (done_cnt == AUTOTB_TRANSACTION_NUM);
    // last transaction is saved at negedge right after last done
    repeat(5) @ (posedge AESL_clock);
    fp = $fopen(`AUTOTB_TVOUT_o_outp1_V_keep_V_out_wrapc, "a");
    if (fp == 0) begin
        $display("Failed to open file \"%s\"!", `AUTOTB_TVOUT_o_outp1_V_keep_V_out_wrapc);
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end
    $fdisplay(fp,"[[[/runtime]]]");
    $fclose(fp);
    dump_tvout_finish_o_outp1_V_keep_V = 1;
end


reg dump_tvout_finish_o_outp1_V_strb_V;

initial begin : dump_tvout_runtime_sign_o_outp1_V_strb_V
    integer fp;
    dump_tvout_finish_o_outp1_V_strb_V = 0;
    fp = $fopen(`AUTOTB_TVOUT_o_outp1_V_strb_V_out_wrapc, "w");
    if (fp == 0) begin
        $display("Failed to open file \"%s\"!", `AUTOTB_TVOUT_o_outp1_V_strb_V_out_wrapc);
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end
    $fdisplay(fp,"[[[runtime]]]");
    $fclose(fp);
    wait (done_cnt == AUTOTB_TRANSACTION_NUM);
    // last transaction is saved at negedge right after last done
    repeat(5) @ (posedge AESL_clock);
    fp = $fopen(`AUTOTB_TVOUT_o_outp1_V_strb_V_out_wrapc, "a");
    if (fp == 0) begin
        $display("Failed to open file \"%s\"!", `AUTOTB_TVOUT_o_outp1_V_strb_V_out_wrapc);
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end
    $fdisplay(fp,"[[[/runtime]]]");
    $fclose(fp);
    dump_tvout_finish_o_outp1_V_strb_V = 1;
end


reg dump_tvout_finish_o_outp1_V_last_V;

initial begin : dump_tvout_runtime_sign_o_outp1_V_last_V
    integer fp;
    dump_tvout_finish_o_outp1_V_last_V = 0;
    fp = $fopen(`AUTOTB_TVOUT_o_outp1_V_last_V_out_wrapc, "w");
    if (fp == 0) begin
        $display("Failed to open file \"%s\"!", `AUTOTB_TVOUT_o_outp1_V_last_V_out_wrapc);
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end
    $fdisplay(fp,"[[[runtime]]]");
    $fclose(fp);
    wait (done_cnt == AUTOTB_TRANSACTION_NUM);
    // last transaction is saved at negedge right after last done
    repeat(5) @ (posedge AESL_clock);
    fp = $fopen(`AUTOTB_TVOUT_o_outp1_V_last_V_out_wrapc, "a");
    if (fp == 0) begin
        $display("Failed to open file \"%s\"!", `AUTOTB_TVOUT_o_outp1_V_last_V_out_wrapc);
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end
    $fdisplay(fp,"[[[/runtime]]]");
    $fclose(fp);
    dump_tvout_finish_o_outp1_V_last_V = 1;
end


////////////////////////////////////////////
// progress and performance
////////////////////////////////////////////

task wait_start();
    while (~AESL_start) begin
        @ (posedge AESL_clock);
    end
endtask

reg [31:0] clk_cnt = 0;
reg AESL_ready_p1;
reg AESL_start_p1;

always @ (posedge AESL_clock) begin
    if (AESL_reset == 0) begin
        clk_cnt <= 32'h0;
        AESL_ready_p1 <= 1'b0;
        AESL_start_p1 <= 1'b0;
    end
    else begin
        clk_cnt <= clk_cnt + 1;
        AESL_ready_p1 <= AESL_ready;
        AESL_start_p1 <= AESL_start;
    end
end

reg [31:0] start_timestamp [0:AUTOTB_TRANSACTION_NUM - 1];
reg [31:0] start_cnt;
reg [31:0] ready_timestamp [0:AUTOTB_TRANSACTION_NUM - 1];
reg [31:0] ap_ready_cnt;
reg [31:0] finish_timestamp [0:AUTOTB_TRANSACTION_NUM - 1];
reg [31:0] finish_cnt;
reg [31:0] lat_total;
event report_progress;

always @(posedge AESL_clock)
begin
    if (finish_cnt == AUTOTB_TRANSACTION_NUM - 1 && AESL_done == 1'b1)
        lat_total = clk_cnt - start_timestamp[0];
end

initial begin
    start_cnt = 0;
    finish_cnt = 0;
    ap_ready_cnt = 0;
    wait (AESL_reset == 1);
    wait_start();
    start_timestamp[start_cnt] = clk_cnt;
    start_cnt = start_cnt + 1;
    if (AESL_done) begin
        finish_timestamp[finish_cnt] = clk_cnt;
        finish_cnt = finish_cnt + 1;
    end
    -> report_progress;
    forever begin
        @ (posedge AESL_clock);
        if (start_cnt < AUTOTB_TRANSACTION_NUM) begin
            if ((AESL_start && AESL_ready_p1)||(AESL_start && ~AESL_start_p1)) begin
                start_timestamp[start_cnt] = clk_cnt;
                start_cnt = start_cnt + 1;
            end
        end
        if (ap_ready_cnt < AUTOTB_TRANSACTION_NUM) begin
            if (AESL_start_p1 && AESL_ready_p1) begin
                ready_timestamp[ap_ready_cnt] = clk_cnt;
                ap_ready_cnt = ap_ready_cnt + 1;
            end
        end
        if (finish_cnt < AUTOTB_TRANSACTION_NUM) begin
            if (AESL_done) begin
                finish_timestamp[finish_cnt] = clk_cnt;
                finish_cnt = finish_cnt + 1;
            end
        end
        -> report_progress;
    end
end

reg [31:0] progress_timeout;

initial begin : simulation_progress
    real intra_progress;
    wait (AESL_reset == 1);
    progress_timeout = PROGRESS_TIMEOUT;
    $display("////////////////////////////////////////////////////////////////////////////////////");
    $display("// Inter-Transaction Progress: Completed Transaction / Total Transaction");
    $display("// Intra-Transaction Progress: Measured Latency / Latency Estimation * 100%%");
    $display("//");
    $display("// RTL Simulation : \"Inter-Transaction Progress\" [\"Intra-Transaction Progress\"] @ \"Simulation Time\"");
    $display("////////////////////////////////////////////////////////////////////////////////////");
    print_progress();
    while (finish_cnt < AUTOTB_TRANSACTION_NUM) begin
        @ (report_progress);
        if (finish_cnt < AUTOTB_TRANSACTION_NUM) begin
            if (AESL_done) begin
                print_progress();
                progress_timeout = PROGRESS_TIMEOUT;
            end else begin
                if (progress_timeout == 0) begin
                    print_progress();
                    progress_timeout = PROGRESS_TIMEOUT;
                end else begin
                    progress_timeout = progress_timeout - 1;
                end
            end
        end
    end
    print_progress();
    $display("////////////////////////////////////////////////////////////////////////////////////");
    calculate_performance();
end

task get_intra_progress(output real intra_progress);
    begin
        if (start_cnt > finish_cnt) begin
            intra_progress = clk_cnt - start_timestamp[finish_cnt];
        end else if(finish_cnt > 0) begin
            intra_progress = LATENCY_ESTIMATION;
        end else begin
            intra_progress = 0;
        end
        intra_progress = intra_progress / LATENCY_ESTIMATION;
    end
endtask

task print_progress();
    real intra_progress;
    begin
        if (LATENCY_ESTIMATION > 0) begin
            get_intra_progress(intra_progress);
            $display("// RTL Simulation : %0d / %0d [%2.2f%%] @ \"%0t\"", finish_cnt, AUTOTB_TRANSACTION_NUM, intra_progress * 100, $time);
        end else begin
            $display("// RTL Simulation : %0d / %0d [n/a] @ \"%0t\"", finish_cnt, AUTOTB_TRANSACTION_NUM, $time);
        end
    end
endtask

task calculate_performance();
    integer i;
    integer fp;
    integer real_cnt;
    integer valid_cnt;
    reg [31:0] latency [0:AUTOTB_TRANSACTION_NUM - 1];
    reg [31:0] latency_min;
    reg [31:0] latency_max;
    reg [31:0] latency_total;
    reg [31:0] latency_average;
    reg [31:0] interval [0:AUTOTB_TRANSACTION_NUM - 2];
    reg [31:0] interval_min;
    reg [31:0] interval_max;
    reg [31:0] interval_total;
    reg [31:0] interval_average;
    reg [31:0] total_execute_time;
    begin
        latency_min = -1;
        latency_max = 0;
        latency_total = 0;
        interval_min = -1;
        interval_max = 0;
        interval_total = 0;
        total_execute_time = lat_total;
        real_cnt = (start_cnt < finish_cnt) ? start_cnt : AUTOTB_TRANSACTION_NUM;
        valid_cnt = 0;

        for (i = 0; i < real_cnt; i = i + 1) begin
            // calculate latency
            if (finish_timestamp[i] >= start_timestamp[i]) begin
                latency[i] = finish_timestamp[i] - start_timestamp[i];
                if (latency[i] > latency_max) latency_max = latency[i];
                if (latency[i] < latency_min) latency_min = latency[i];
                latency_total = latency_total + latency[i];
                valid_cnt = valid_cnt + 1;
            end else begin
                latency[i] = 0;
            end
            // calculate interval
            if (real_cnt == 1) begin
                interval[i] = 0;
                interval_max = 0;
                interval_min = 0;
                interval_total = 0;
            end else if (i < real_cnt - 1) begin
                interval[i] = start_timestamp[i + 1] - start_timestamp[i];
                if (interval[i] > interval_max) interval_max = interval[i];
                if (interval[i] < interval_min) interval_min = interval[i];
                interval_total = interval_total + interval[i];
            end
        end

        if (valid_cnt > 0)
            latency_average = latency_total / valid_cnt;
        else
            latency_average = latency_total;
        if (real_cnt == 1) begin
            interval_average = 0;
        end else begin
            interval_average = interval_total / (real_cnt - 1);
        end

        fp = $fopen(`AUTOTB_LAT_RESULT_FILE, "w");

        $fdisplay(fp, "$MAX_LATENCY = \"%0d\"", latency_max);
        $fdisplay(fp, "$MIN_LATENCY = \"%0d\"", latency_min);
        $fdisplay(fp, "$AVER_LATENCY = \"%0d\"", latency_average);
        $fdisplay(fp, "$MAX_THROUGHPUT = \"%0d\"", interval_max);
        $fdisplay(fp, "$MIN_THROUGHPUT = \"%0d\"", interval_min);
        $fdisplay(fp, "$AVER_THROUGHPUT = \"%0d\"", interval_average);
        $fdisplay(fp, "$TOTAL_EXECUTE_TIME = \"%0d\"", total_execute_time);

        $fclose(fp);

        fp = $fopen(`AUTOTB_PER_RESULT_TRANS_FILE, "w");

        $fdisplay(fp, "%20s%16s%16s", "", "latency", "interval");
        if (real_cnt == 1) begin
            i = 0;
            $fdisplay(fp, "transaction%8d:%16d%16d", i, latency[i], interval[i]);
        end else begin
            for (i = 0; i < real_cnt; i = i + 1) begin
                if (i < real_cnt - 1) begin
                    $fdisplay(fp, "transaction%8d:%16d%16d", i, latency[i], interval[i]);
                end else begin
                    $fdisplay(fp, "transaction%8d:%16d               x", i, latency[i]);
                end
            end
        end

        $fclose(fp);
        if (start_cnt < finish_cnt)
            $display("Note: for this 'ap_ctrl_none' design the last %0d transactions have no output. In order to save runtime, cosim will end eariler after getting all needed output. So performance report only covers the first %0d transactions.",(finish_cnt - start_cnt), start_cnt);
    end
endtask

///////////////////////////////////////////////////////
// dataflow status monitor
///////////////////////////////////////////////////////
//dataflow_monitor U_dataflow_monitor(
//    .clock(AESL_clock),
//    .reset(~rst),
//   .finish(all_finish));

`include "fifo_para.vh"

endmodule
