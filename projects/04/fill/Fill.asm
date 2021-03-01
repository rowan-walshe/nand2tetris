// This file is part of www.nand2tetris.org
// and the book "The Elements of Computing Systems"
// by Nisan and Schocken, MIT Press.
// File name: projects/04/Fill.asm

// Runs an infinite loop that listens to the keyboard input.
// When a key is pressed (any key), the program blackens the screen,
// i.e. writes "black" in every pixel;
// the screen should remain fully black as long as the key is pressed. 
// When no key is pressed, the program clears the screen, i.e. writes
// "white" in every pixel;
// the screen should remain fully clear as long as no key is pressed.

// Initialise variables
@8191
D=A
@screen_end
M=D
@screen_i
M=0

// TODO initialise screen if needed
// Note: Test passes without initialising screen to white, so not doing this for now

(START)
    // Figure out if key is pressed or not
    @KBD
    D=M
    @FILL_WHITE
    D;JEQ

(FILL_BLACK)
    @screen_i
    D=M
    @SCREEN
    A=D+A
    M=-1  // Set word to black
    @screen_end
    D=D-M
    @START
    D;JEQ  // Jump to the start if the index is at the end, i.e. the screen is all black
    @screen_i
    M=M+1
    @START
    0;JMP

(FILL_WHITE)
    @screen_i
    D=M
    @SCREEN
    A=D+A
    M=0  // Set word to white
    @START
    D;JEQ  // Jump to the start if the index is 0, i.e. the screen is all white
    @screen_i
    M=M-1
    @START
    0;JMP