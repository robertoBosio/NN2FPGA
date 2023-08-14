`timescale 1ns/100ps

// `timescale 100ps/100ps
// This module describes SIMD Inference 
// 4 adders packed into single DSP block
(* dont_touch = "1" *)
module mult_simd (
  input [7:0] a, d, b,
  input apClk, apRst, apCE, apStart, apContinue,
  output reg [15:0] ab, bd, 
  output ab_apVld, bd_apVld,
  output apIdle, apDone, apReady
); 

wire signed [26:0] aDSP;
wire signed [26:0] dDSP;
wire signed [17:0] bDSP;
wire signed [47:0] pDSP;
reg dly;
wire apReadyDSP;
wire apVldDSP;
wire ce;

assign aDSP = {a[7], a, 18'b0};
assign dDSP[26:8] = {19{d[7]}}; assign dDSP[7:0] = d;
assign bDSP[17:8] = {10{b[7]}}; assign bDSP[7:0] = b;
assign apReady = apReadyDSP;
assign ab_apVld = dly;
assign bd_apVld = dly;
assign apDone = dly;
assign apIdle = ~apStart;
assign ce = apCE ; // & start;

dsp_macro_fab dsp0 (
  .apClk(apClk), 
  .apRst(apRst), 
  .apStart(apStart),
  .apCE(ce), 
  .a(aDSP), 
  .d(dDSP), 
  .b(bDSP),
  .apReady(apReadyDSP), 
  .p_apVld(p_apVldDSP),
  .p(pDSP)
);

always @ (posedge apClk) 
begin
  if (apRst) 
  begin
    bd <= 0;
    ab <= 0;
    dly <= 0;
  end 
  else if (apCE) 
  begin
    bd <= pDSP[15:0];
    ab <= pDSP[33:18] + pDSP[15];
    dly <= p_apVldDSP;
  end
end

endmodule // mult_simd
