`timescale 1ns/100ps

(* USE_DSP = "YES" *) module dsp_macro_fab (
  input apClk, apRst, apCE, apStart,
  input signed [26:0] a,
  input signed [26:0] d,
  input signed [17:0] b,
  output apReady,
  output p_apVld,
  output signed [47:0] p
);


reg apVldReg[3:0];
integer i;

always @ (posedge apClk)
begin
  if (apRst)
  begin
    for (i = 0; i < 4; i = i + 1)
      apVldReg[i] <= 0;
  end
  else
  begin
    if (apCE)
    begin
      apVldReg[0] <= apStart;
      for (i = 0; i < 3; i = i + 1)
        apVldReg[i + 1] <= apVldReg[i];
    end
  end
end

assign p_apVld = apVldReg[3];
assign apReady = apVldReg[1]; 

wire [29:0] aDSP = {a[26], a[26], a[26], a};

// Xilinx macro signals.
wire [29:0] aCout;
wire [17:0] bCout;
wire carryCascOut;
wire multSignOut;
wire [47:0] pCout;
wire overflow;
wire patternBDetect;
wire patternDetect;
wire underflow;
wire [3:0] carryOut;
wire [7:0] xorOut;
//  <-----Cut code below this line---->

   // DSP48E2: 48-bit Multi-Functional Arithmetic Block
   //          Kintex UltraScale
   // Xilinx HDL Language Template, version 2022.2

   DSP48E2 #(
      // Feature Control Attributes: Data Path Selection
      .AMULTSEL("AD"),                    // Selects A input to multiplier (A, AD)
      .A_INPUT("DIRECT"),                // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
      .BMULTSEL("B"),                    // Selects B input to multiplier (AD, B)
      .B_INPUT("DIRECT"),                // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
      .PREADDINSEL("A"),                 // Selects input to pre-adder (A, B)
      .RND(48'h000000000000),            // Rounding Constant
      .USE_MULT("MULTIPLY"),             // Select multiplier usage (DYNAMIC, MULTIPLY, NONE)
      .USE_SIMD("ONE48"),                // SIMD selection (FOUR12, ONE48, TWO24)
      .USE_WIDEXOR("FALSE"),             // Use the Wide XOR function (FALSE, TRUE)
      .XORSIMD("XOR24_48_96"),           // Mode of operation for the Wide XOR (XOR12, XOR24_48_96)
      // Pattern Detector Attributes: Pattern Detection Configuration
      .AUTORESET_PATDET("NO_RESET"),     // NO_RESET, RESET_MATCH, RESET_NOT_MATCH
      .AUTORESET_PRIORITY("RESET"),      // Priority of AUTORESET vs. CEP (CEP, RESET).
      .MASK(48'hffffffffffff),           // 48-bit mask value for pattern detect (1=ignore)
      .PATTERN(48'h000000000000),        // 48-bit pattern match for pattern detect
      .SEL_MASK("MASK"),                 // C, MASK, ROUNDING_MODE1, ROUNDING_MODE2
      .SEL_PATTERN("PATTERN"),           // Select pattern value (C, PATTERN)
      .USE_PATTERN_DETECT("NO_PATDET"),  // Enable pattern detect (NO_PATDET, PATDET)
      // Programmable Inversion Attributes: Specifies built-in programmable inversion on specific pins
      .IS_ALUMODE_INVERTED(4'b0000),     // Optional inversion for ALUMODE
      .IS_CARRYIN_INVERTED(1'b0),        // Optional inversion for CARRYIN
      .IS_CLK_INVERTED(1'b0),            // Optional inversion for CLK
      .IS_INMODE_INVERTED(5'b00000),     // Optional inversion for INMODE
      .IS_OPMODE_INVERTED(9'b000000000), // Optional inversion for OPMODE
      .IS_RSTALLCARRYIN_INVERTED(1'b0),  // Optional inversion for RSTALLCARRYIN
      .IS_RSTALUMODE_INVERTED(1'b0),     // Optional inversion for RSTALUMODE
      .IS_RSTA_INVERTED(1'b0),           // Optional inversion for RSTA
      .IS_RSTB_INVERTED(1'b0),           // Optional inversion for RSTB
      .IS_RSTCTRL_INVERTED(1'b0),        // Optional inversion for RSTCTRL
      .IS_RSTC_INVERTED(1'b0),           // Optional inversion for RSTC
      .IS_RSTD_INVERTED(1'b0),           // Optional inversion for RSTD
      .IS_RSTINMODE_INVERTED(1'b0),      // Optional inversion for RSTINMODE
      .IS_RSTM_INVERTED(1'b0),           // Optional inversion for RSTM
      .IS_RSTP_INVERTED(1'b0),           // Optional inversion for RSTP
      // Register Control Attributes: Pipeline Register Configuration
      .ACASCREG(1),                      // Number of pipeline stages between A/ACIN and ACOUT (0-2)
      .ADREG(1),                         // Pipeline stages for pre-adder (0-1)
      .ALUMODEREG(0),                    // Pipeline stages for ALUMODE (0-1)
      .AREG(1),                          // Pipeline stages for A (0-2)
      .BCASCREG(1),                      // Number of pipeline stages between B/BCIN and BCOUT (0-2)
      .BREG(2),                          // Pipeline stages for B (0-2)
      .CARRYINREG(0),                    // Pipeline stages for CARRYIN (0-1)
      .CARRYINSELREG(0),                 // Pipeline stages for CARRYINSEL (0-1)
      .CREG(0),                          // Pipeline stages for C (0-1)
      .DREG(1),                          // Pipeline stages for D (0-1)
      .INMODEREG(0),                     // Pipeline stages for INMODE (0-1)
      .MREG(1),                          // Multiplier pipeline stages (0-1)
      .OPMODEREG(0),                     // Pipeline stages for OPMODE (0-1)
      .PREG(1)                           // Number of pipeline stages for P (0-1)
   )
   eddai (
      // Cascade outputs: Cascade Ports
      .ACOUT(aCout),                   // 30-bit output: A port cascade
      .BCOUT(bCout),                   // 18-bit output: B cascade
      .CARRYCASCOUT(carryCascOut),     // 1-bit output: Cascade carry
      .MULTSIGNOUT(multSignOut),       // 1-bit output: Multiplier sign cascade
      .PCOUT(pCout),                   // 48-bit output: Cascade output
      // Control outputs: Control Inputs/Status Bits
      .OVERFLOW(overflow),             // 1-bit output: Overflow in add/acc
      .PATTERNBDETECT(patternBDetect), // 1-bit output: Pattern bar detect
      .PATTERNDETECT(patternDetect),   // 1-bit output: Pattern detect
      .UNDERFLOW(underflow),           // 1-bit output: Underflow in add/acc
      // Data outputs: Data Ports
      .CARRYOUT(carryOut),             // 4-bit output: Carry
      .P(p),                           // 48-bit output: Primary data
      .XOROUT(xorOut),                 // 8-bit output: XOR data
      // Cascade inputs: Cascade Ports
      .ACIN(30'h00000000),                     // 30-bit input: A cascade data
      .BCIN(18'h00000),                     // 18-bit input: B cascade
      .CARRYCASCIN(1'b0),       // 1-bit input: Cascade carry
      .MULTSIGNIN(1'b0),         // 1-bit input: Multiplier sign cascade
      .PCIN(48'h000000000000),                     // 48-bit input: P cascade
      // Control inputs: Control Inputs/Status Bits
      .ALUMODE(4'b0000),               // 4-bit input: ALU control
      .CARRYINSEL(3'b000),         // 3-bit input: Carry select
      .CLK(apClk),                       // 1-bit input: Clock
      .INMODE(5'b00101),                 // 5-bit input: INMODE control
      .OPMODE(9'b000000101),                 // 9-bit input: Operation mode
      // Data inputs: Data Ports
      .A(aDSP),                           // 30-bit input: A data
      .B(b),                           // 18-bit input: B data
      .C(48'h00000000),                           // 48-bit input: C data
      .CARRYIN(1'b0),               // 1-bit input: Carry-in
      .D(d),                           // 27-bit input: D data
      // Reset/Clock Enable inputs: Reset/Clock Enable Inputs
      .CEA1(apCE),                     // 1-bit input: Clock enable for 1st stage AREG
      .CEA2(1'b0),                     // 1-bit input: Clock enable for 2nd stage AREG
      .CEAD(apCE),                     // 1-bit input: Clock enable for ADREG
      .CEALUMODE(1'b0),           // 1-bit input: Clock enable for ALUMODE
      .CEB1(apCE),                     // 1-bit input: Clock enable for 1st stage BREG
      .CEB2(apCE),                     // 1-bit input: Clock enable for 2nd stage BREG
      .CEC(1'b0),                       // 1-bit input: Clock enable for CREG
      .CECARRYIN(1'b0),           // 1-bit input: Clock enable for CARRYINREG
      .CECTRL(1'b0),                 // 1-bit input: Clock enable for OPMODEREG and CARRYINSELREG
      .CED(apCE),                       // 1-bit input: Clock enable for DREG
      .CEINMODE(1'b0),             // 1-bit input: Clock enable for INMODEREG
      .CEM(apCE),                       // 1-bit input: Clock enable for MREG
      .CEP(apCE),                       // 1-bit input: Clock enable for PREG
      .RSTA(apRst),                     // 1-bit input: Reset for AREG
      .RSTALLCARRYIN(1'b1),   // 1-bit input: Reset for CARRYINREG
      .RSTALUMODE(1'b1),         // 1-bit input: Reset for ALUMODEREG
      .RSTB(apRst),                     // 1-bit input: Reset for BREG
      .RSTC(1'b1),                     // 1-bit input: Reset for CREG
      .RSTCTRL(1'b1),               // 1-bit input: Reset for OPMODEREG and CARRYINSELREG
      .RSTD(apRst),                     // 1-bit input: Reset for DREG and ADREG
      .RSTINMODE(1'b1),           // 1-bit input: Reset for INMODEREG
      .RSTM(apRst),                     // 1-bit input: Reset for MREG
      .RSTP(apRst)                      // 1-bit input: Reset for PREG
   );

   // End of DSP48E2_inst instantiation



endmodule // dsp_macro_fab
