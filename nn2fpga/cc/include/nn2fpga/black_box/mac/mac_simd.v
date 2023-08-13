// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2.0 (64-bit)
// Tool Version Limit: 2023.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
`timescale 1 ns / 1 ps

module top_ama_addmuladd_26s_26s_18s_28s_45_4_1_DSP48_0 (
    input clk,
    input rst,
    input ce,
    input  [27 - 1:0] a,
    input  [27 - 1:0] d,
    input  [18 - 1:0] b,
    input  [48 - 1:0] c,
    output signed [48 - 1:0]  dout);


wire signed [48 - 1:0]     p;
reg  signed [48 - 1:0]     p_reg;

assign p = (d + a) * b + c;

// always @(posedge clk) begin
//     if (ce) begin
//         p_reg  <= p;
//     end
// end

assign dout = p;

endmodule

(* use_dsp = "yes" *)
module mac_simd(
    clk,
    reset,
    ce,
    A,
    B,
    C,
    D,
    ap_return);

parameter ID = 32'd1;
input clk;
input reset;
input ce;
input[27 - 1:0] A;
input[18 - 1:0] B;
input[48 - 1:0] C;
input[27 - 1:0] D;
output[48 - 1:0] ap_return;



top_ama_addmuladd_26s_26s_18s_28s_45_4_1_DSP48_0 top_ama_addmuladd_26s_26s_18s_28s_45_4_1_DSP48_0_U(
    .clk( clk ),
    .rst( reset ),
    .ce( ce ),
    .a( A ),
    .b( B ),
    .c( C ),
    .d( D ),
    .dout( ap_return ));
endmodule
