// This file is part of www.nand2tetris.org
// and the book "The Elements of Computing Systems"
// by Nisan and Schocken, MIT Press.
// File name: projects/04/Mult.asm

// Multiplies R0 and R1 and stores the result in R2.
// (R0, R1, R2 refer to RAM[0], RAM[1], and RAM[2], respectively.)
//
// This program only needs to handle arguments that satisfy
// R0 >= 0, R1 >= 0, and R0*R1 < 32768.

// Put your code here.

// Initialise R2
@R2
M=0

// If either is 0 the answer is 0
@R0
D=M
@END
D;JEQ
@R1
D=M
@END
D;JEQ

// Find out if if R0 or R1 is bigger, to allow for shorter loops
@R0
D=M
@R1
D=D-M
@MULTR1LT
D;JGT

// Loop until R0 is 0
(MULTR0LT)
    @R1
    D=M
    @R2
    M=D+M
    @R0
    MD=M-1
    @MULTR0LT
    D;JGT
    @END
    0;JMP

// Loop until R1 is 0
(MULTR1LT)
    @R0
    D=M
    @R2
    M=D+M
    @R1
    MD=M-1
    @MULTR1LT
    D;JGT

(END)
    @END
    0;JMP