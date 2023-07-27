`timescale 1ns/100ps

// `timescale 100ps/100ps
// This module describes SIMD Inference 
// Two MACs in one DSP48E2
(* dont_touch = "1" *)
module mac_simd (
  input [26:0] a, d,
  input [17:0] b,
  input apClk, apRst, apCE, apStart, apContinue,
  output reg [47:0] p, 
  output p_apVld,
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

assign apReady = apReadyDSP;
assign apDone = dly;
assign apIdle = ~apStart;
assign ce = apCE ; // & start;

dsp_macro dsp0 (
  .apClk(apClk), 
  .apRst(apRst), 
  .apStart(apStart),
  .apCE(ce), 
  .a(a), 
  .d(d), 
  .b(b),
  .apReady(apReadyDSP), 
  .p_apVld(p_apVldDSP),
  .p(pDSP)
);

always @ (posedge apClk) 
begin
  if (apRst) 
  begin
    p <= 0;
    dly <= 0;
  end 
  else if (apCE) 
  begin
    p <= pDSP;
    dly <= p_apVldDSP;
  end
end

endmodule // mac_simd
