import argparse
from enum import Enum

from os import path
from copy import deepcopy
from typing import List, Tuple


def is_int(val: str) -> bool:
    """Returns True if val is a valid int, False otherwise"""
    try:
        int(val)
        return True
    except ValueError:
        return False


class VM_Error(Exception):
    """Raised for invalid VM statements"""


class VMLine:
    """
    Used to represent a single VM line, and conver it to a hack equivalent.
    Will raise an
    """
    COMMANDS = {"push": "_vm_push",
                "pop": "_vm_pop",
                "add": "_vm_add",
                "sub": "_vm_sub",
                "neg": "_vm_neg",
                "eq": "_vm_eq",
                "gt": "_vm_gt",
                "lt": "_vm_lt",
                "and": "_vm_and",
                "or": "_vm_or",
                "not": "_vm_not"}

    MEMORY_SEGMENTS = {"local",
                       "argument",
                       "this",
                       "that",
                       "constant",
                       "static",
                       "pointer",
                       "temp"}
    
    INC_SP = ["@SP", "M=M+1"]
    DEC_SP = ["@SP", "M=M-1"]
    PUT_D_ON_STACK = ["@SP", "A=M", "M=D"]
    POP_STACK_ON_D = ["@SP", "A=M", "D=M"]
    TEMP = 5

    def __init__(self,
                 original_line_number: int,
                 original_input: str,
                 file_name: str,
                 add_comments: bool = False):
        # Init attributes
        self.__line_number: int = original_line_number
        self.__original_input: str = original_input
        self.__asm: List = []
        self.__file_name: str = file_name
        self.__add_comments: str = add_comments
        if add_comments:
            self.__asm.append(f"// {original_input}")

        # Convert origin input to asm
        self.__convert_vm_to_asm()

    def __convert_vm_to_asm(self):
        command = self.__original_input.split(" ")[0]
        if command in VMLine.COMMANDS:
            getattr(self, VMLine.COMMANDS[command])()
        else:
            raise VM_Error(f"Unrecognised VM command: '{self.__original_input}'")

    @property
    def line_number(self) -> int:
        return self.__line_number

    @property
    def original_input(self) -> str:
        return deepcopy(self.__original_input)

    def get_asm(self) -> Tuple[str]:
        if not self.__asm or len(self.__asm) == int(self.__add_comments):
            raise VM_Error(f"Failed to convert VM command: '{self.__original_input}'")
        return tuple(self.__asm)

    def __validate_push_pop(self) -> List[str]:
        # A push/pop should have three distinct parts:
        # 1. 'push' or 'pop'
        # 2. A memory segment name e.g. constant, local, etc.
        # 3. A positive integer (n >= 0)
        parts = self.__original_input.split(" ")
        if len(parts) != 3:
            raise VM_Error(f"Unrecognised '{parts[0]}'' command: '{self.__original_input}'")
        if parts[1] not in VMLine.MEMORY_SEGMENTS:
            raise VM_Error(f"Unrecognised memory segment: '{parts[1]}'")
        if not is_int(parts[2]):
            raise VM_Error(f"Invalid offset: '{parts[2]}'. Expected integer n>=0")
        return parts

    def _vm_push(self):
        parts = self.__validate_push_pop()
        if parts[1] == "local":
            self.__asm.extend([f"@{parts[2]}", "D=A", "@LCL", "A=D+M", "D=M"])  # Put RAM[*LCL + i] into D
            self.__asm.extend(self.PUT_D_ON_STACK)
        elif parts[1] == "argument":
            self.__asm.extend([f"@{parts[2]}", "D=A", "@ARG", "A=D+M", "D=M"])  # Put RAM[*ARG + i] into D
            self.__asm.extend(self.PUT_D_ON_STACK)
        elif parts[1] == "this":
            self.__asm.extend([f"@{parts[2]}", "D=A", "@THIS", "A=D+M", "D=M"])  # Put RAM[*THIS + i] into D
            self.__asm.extend(self.PUT_D_ON_STACK)
        elif parts[1] == "that":
            self.__asm.extend([f"@{parts[2]}", "D=A", "@THAT", "A=D+M", "D=M"])  # Put RAM[*THAT + i] into D
            self.__asm.extend(self.PUT_D_ON_STACK)
        elif parts[1] == "constant":
            self.__asm.extend([f"@{parts[2]}", "D=A"])
            self.__asm.extend(self.PUT_D_ON_STACK)
        elif parts[1] == "static":
            self.__asm.extend([f"@{self.__file_name}.{parts[2]}", "D=M"])
            self.__asm.extend(self.PUT_D_ON_STACK)
        elif parts[1] == "pointer":
            if parts[2] == '0':
                self.__asm.extend(["@THIS", "D=M"])
            elif parts[2] == '1':
                self.__asm.extend(["@THAT", "D=M"])
            else:
                raise VM_Error(f"Push from pointer '{parts[2]}', but only values 0 or 1 are valid")
            self.__asm.extend(self.PUT_D_ON_STACK)
        elif parts[1] == "temp":
            offset = int(parts[2])
            if 0 <= offset <= 7:
                address = self.TEMP + offset
                self.__asm.extend([f"@{address}", "D=M"])
                self.__asm.extend(self.PUT_D_ON_STACK)
            else:
                raise VM_Error(f"Trying to access temp memory segment out of range: 'RAM[*temp+{offset}]'")
        else:
            raise VM_Error(f"Push from memory segment '{parts[1]}' not yet implemented")
        self.__asm.extend(self.INC_SP)

    def _vm_pop(self):
        parts = self.__validate_push_pop()
        self.__asm.extend(self.DEC_SP)
        if parts[1] == "local":
            self.__asm.extend([f"@{parts[2]}", "D=A", "@LCL", "D=D+M"])  # Compute address for popping into
            self.__asm.extend(["@R13", "M=D"])  # Store computed address in R13
            self.__asm.extend(self.POP_STACK_ON_D)
            self.__asm.extend(["@R13", "A=M", "M=D"])  # Load R13 into A, and store D
        elif parts[1] == "argument":
            self.__asm.extend([f"@{parts[2]}", "D=A", "@ARG", "D=D+M"])  # Compute address for popping into
            self.__asm.extend(["@R13", "M=D"])  # Store computed address in R13
            self.__asm.extend(self.POP_STACK_ON_D)
            self.__asm.extend(["@R13", "A=M", "M=D"])  # Load R13 into A, and store D
        elif parts[1] == "this":
            self.__asm.extend([f"@{parts[2]}", "D=A", "@THIS", "D=D+M"])  # Compute address for popping into
            self.__asm.extend(["@R13", "M=D"])  # Store computed address in R13
            self.__asm.extend(self.POP_STACK_ON_D)
            self.__asm.extend(["@R13", "A=M", "M=D"])  # Load R13 into A, and store D
        elif parts[1] == "that":
            self.__asm.extend([f"@{parts[2]}", "D=A", "@THAT", "D=D+M"])  # Compute address for popping into
            self.__asm.extend(["@R13", "M=D"])  # Store computed address in R13
            self.__asm.extend(self.POP_STACK_ON_D)
            self.__asm.extend(["@R13", "A=M", "M=D"])  # Load R13 into A, and store D
        elif parts[1] == "constant":
            raise VM_Error(f"Pop to constant not allowed")
        elif parts[1] == "static":
            self.__asm.extend(self.POP_STACK_ON_D)
            self.__asm.extend([f"@{self.__file_name}.{parts[2]}", "M=D"])
        elif parts[1] == "pointer":
            self.__asm.extend(self.POP_STACK_ON_D)
            if parts[2] == '0':
                self.__asm.extend(["@THIS", "M=D"])
            elif parts[2] == '1':
                self.__asm.extend(["@THAT", "M=D"])
            else:
                raise VM_Error(f"Pop to pointer '{parts[2]}', but only values 0 or 1 are valid")
        elif parts[1] == "temp":
            offset = int(parts[2])
            if 0 <= offset <= 7:
                address = self.TEMP + offset
                self.__asm.extend(self.POP_STACK_ON_D)
                self.__asm.extend([f"@{address}", "M=D"])
            else:
                raise VM_Error(f"Trying to access temp memory segment out of range: 'RAM[*temp+{offset}]'")
        else:
            raise VM_Error(f"Pop from memory segment '{parts[1]}' not yet implemented")

    def _vm_add(self):
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "D=M"])  # Pop value at top of stack into D
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "M=D+M"])  # Add D to the value at the top of the stack, and push the result
        self.__asm.extend(self.INC_SP)

    def _vm_sub(self):
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "D=M"])  # Pop value at top of stack into D
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "M=M-D"])  # Substract D from the value at top of stack, and push the result
        self.__asm.extend(self.INC_SP)

    def _vm_neg(self):
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "M=-M"])  # Pop value at top of stack into D
        self.__asm.extend(self.INC_SP)

    def _vm_eq(self):
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "D=M"])  # Pop value at top of stack into D
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "D=M-D"])  # Substract D from the value at top of stack, and push the result
        self.__asm.extend([
            f"@TRUE.{self.__file_name}.{self.__line_number}",
            "D;JEQ",
            "@SP",
            "A=M",
            "M=0",
            f"@END.{self.__file_name}.{self.__line_number}",
            "0;JMP",
            f"(TRUE.{self.__file_name}.{self.__line_number})",
            "@SP",
            "A=M",
            "M=-1",
            f"(END.{self.__file_name}.{self.__line_number})",
        ])
        self.__asm.extend(self.INC_SP)

    def _vm_gt(self):
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "D=M"])  # Pop value at top of stack into D
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "D=M-D"])  # Substract D from the value at top of stack, and push the result
        self.__asm.extend([
            f"@TRUE.{self.__file_name}.{self.__line_number}",
            "D;JGT",
            "@SP",
            "A=M",
            "M=0",
            f"@END.{self.__file_name}.{self.__line_number}",
            "0;JMP",
            f"(TRUE.{self.__file_name}.{self.__line_number})",
            "@SP",
            "A=M",
            "M=-1",
            f"(END.{self.__file_name}.{self.__line_number})",
        ])
        self.__asm.extend(self.INC_SP)

    def _vm_lt(self):
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "D=M"])  # Pop value at top of stack into D
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "D=M-D"])  # Substract D from the value at top of stack, and push the result
        self.__asm.extend([
            f"@TRUE.{self.__file_name}.{self.__line_number}",
            "D;JLT",
            "@SP",
            "A=M",
            "M=0",
            f"@END.{self.__file_name}.{self.__line_number}",
            "0;JMP",
            f"(TRUE.{self.__file_name}.{self.__line_number})",
            "@SP",
            "A=M",
            "M=-1",
            f"(END.{self.__file_name}.{self.__line_number})",
        ])
        self.__asm.extend(self.INC_SP)

    def _vm_and(self):
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "D=M"])  # Pop value at top of stack into D
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "M=D&M"])  # And D with the value at the top of the stack, and push the result
        self.__asm.extend(self.INC_SP)

    def _vm_or(self):
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "D=M"])  # Pop value at top of stack into D
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "M=D|M"])  # Or D with the value at the top of the stack, and push the result
        self.__asm.extend(self.INC_SP)

    def _vm_not(self):
        self.__asm.extend(self.DEC_SP)
        self.__asm.extend(["A=M", "M=!M"])  # Not the top of the stack, and push the result
        self.__asm.extend(self.INC_SP)


class ASM_Optimiser_State(Enum):
    Null = 0
    INC_DEC = 1
    DEC_INC = 2
    EITHER = 3


class ASM_Optimiser:

    SP_INC_DEC = ["@SP", "M=M+1", "@SP", "M=M-1"]
    SP_DEC_INC = ["@SP", "M=M-1", "@SP", "M=M+1"]

    def __init__(self, orig_asm: List[str]):
        self.__orig_asm = orig_asm
        self.__new_asm = []
        self.__buffer = []
        self.__comment_buff = []
        self.__non_comment_buffer_length = 0
        self.__last_check = len(self.__orig_asm) - 3
        self.__state = ASM_Optimiser_State.Null

        self.__optimise()

    @property
    def asm(self) -> List[str]:
        return self.__new_asm

    def __handle_Null(self, current_line):
        if current_line == "@SP":
            self.__state = ASM_Optimiser_State.EITHER
            self.__buffer.append(current_line)
            self.__non_comment_buffer_length += 1
        else:
            self.__new_asm.append(current_line)

    def __handle_INC_or_DEC(self, current_line: str, lookup: List[str]):
        if self.__non_comment_buffer_length == 2 and current_line == "@SP":
            self.__buffer.append(current_line)
            self.__non_comment_buffer_length += 1
        elif self.__non_comment_buffer_length == 3 and current_line == lookup[3]:
            self.__new_asm.extend(self.__comment_buff)
            self.__new_asm.append("@SP")
            self.__buffer = []
            self.__comment_buff = []
            self.__non_comment_buffer_length = 0
            self.__state = ASM_Optimiser_State.Null
        else:
            self.__new_asm.extend(self.__buffer)
            self.__new_asm.append(current_line)
            self.__buffer = []
            self.__comment_buff = []
            self.__non_comment_buffer_length = 0
            self.__state = ASM_Optimiser_State.Null

    def __handle_EITHER(self, current_line: str):
        if current_line == self.SP_INC_DEC[1]:
            self.__state = ASM_Optimiser_State.INC_DEC
            self.__buffer.append(current_line)
            self.__non_comment_buffer_length += 1
        elif current_line == self.SP_DEC_INC[1]:
            self.__state = ASM_Optimiser_State.DEC_INC
            self.__buffer.append(current_line)
            self.__non_comment_buffer_length += 1
        else:
            self.__state = ASM_Optimiser_State.Null
            self.__new_asm.extend(self.__buffer)
            self.__new_asm.append(current_line)
            self.__buffer = []
            self.__comment_buff = []
            self.__non_comment_buffer_length = 0

    def __optimise(self):
        for i in range(self.__last_check):
            current_line = self.__orig_asm[i]
            if self.__state != ASM_Optimiser_State.Null and current_line.startswith("//"):
                self.__buffer.append(current_line)
                self.__comment_buff.append(current_line)
                continue
            if self.__state == ASM_Optimiser_State.EITHER:
                self.__handle_EITHER(current_line)
            elif self.__state == ASM_Optimiser_State.INC_DEC:
                self.__handle_INC_or_DEC(current_line, self.SP_INC_DEC)
            elif self.__state == ASM_Optimiser_State.DEC_INC:
                self.__handle_INC_or_DEC(current_line, self.SP_DEC_INC)
            else:
                self.__handle_Null(current_line)
        self.__new_asm.extend(self.__buffer)
        self.__new_asm.extend(self.__orig_asm[-3:])


def remove_unwanted_whitespace(lines: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    new_lines = []
    for line in lines:
        ln = line[0]
        text = line[1]
        text = text.strip()
        new_lines.append((ln, text))
    return new_lines


def remove_comments(lines: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    new_lines = []
    for line in lines:
        ln = line[0]
        text = line[1]
        comment_idx = text.find("//")
        if comment_idx >= 0:
            text = text[:comment_idx]
        new_lines.append((ln, text))
    return new_lines


def remove_empty(lines: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    new_lines = []
    for line in lines:
        ln = line[0]
        text = line[1]
        if text:
            new_lines.append((ln, text))
    return new_lines


def parse_vm(lines: List[str],
             file_name: str,
             add_comments: bool) -> List[VMLine]:
    lines = list(enumerate(lines))
    lines = remove_unwanted_whitespace(lines)
    lines = remove_comments(lines)
    lines = remove_empty(lines)
    lines = [VMLine(x, y, file_name, add_comments) for x, y in lines]
    return lines


def convert_vm_to_asm(vm_lines: List[VMLine]) -> List[str]:
    asm_lines = []
    for line in vm_lines:
        asm_lines.extend(line.get_asm())
    
    asm_lines.append("(INFINITE_LOOP)")
    asm_lines.append("@INFINITE_LOOP")
    asm_lines.append("0;JMP")
    return asm_lines


def remove_redundant_sp_moves(asm_lines: List[str]) -> List[str]:
    optimiser = ASM_Optimiser(asm_lines)
    return optimiser.asm


def remove_redundant_a_instructions(asm_lines: List[str]) -> List[str]:
    new_asm = []
    for i in range(0, len(asm_lines)-1):
        if asm_lines[i].startswith('@') and asm_lines[i + 1].startswith('@'):
            continue
        new_asm.append(asm_lines[i])
    return new_asm


def parse_file(input_file_path: str,
               output_file_path: str,
               add_comments: bool):
    with open(input_file_path, "r") as f:
        lines = f.readlines()
    # Get file name for static variable naming
    file_name = input_file_path.rsplit('.', 1)[0]
    file_name = path.split(file_name)[-1]
    lines = parse_vm(lines, file_name, add_comments)
    lines = convert_vm_to_asm(lines)
    lines = remove_redundant_sp_moves(lines)
    lines = remove_redundant_a_instructions(lines)
    with open(output_file_path, "w+") as f:
        for line in lines:
            f.write(str(line))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="vm_file_path", type=str, help="Filepath of .vm program to be converted to .asm")
    parser.add_argument("--add_comments", action="store_true", help="Adds comments to generated assembly")
    args = parser.parse_args()
    file_path_without_extension = args.vm_file_path.rsplit('.', 1)[0]
    output_file_path = file_path_without_extension + ".asm"
    parse_file(input_file_path=args.vm_file_path,
               output_file_path=output_file_path,
               add_comments=args.add_comments)
