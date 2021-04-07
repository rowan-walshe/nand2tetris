import argparse
import re

from typing import List, Tuple

JUMP_LOOKUP = {"NULL": 0,
               "JGT": 1,
               "JEQ": 2,
               "JGE": 3,
               "JLT": 4,
               "JNE": 5,
               "JLE": 6,
               "JMP": 7}

DEST_LOOKUP = {"NULL": 0,
               "M": 1 << 3,
               "D": 2 << 3,
               "MD": 3 << 3,
               "A": 4 << 3,
               "AM": 5 << 3,
               "AD": 6 << 3,
               "AMD": 7 << 3}

COMP_LOOKUP = {"0": 0b101010 << 6,
               "1": 0b111111 << 6,
               "-1": 0b111010 << 6,
               "D": 0b001100 << 6,
               "A": 0b110000 << 6,
               "!D": 0b001101 << 6,
               "!A": 0b110001 << 6,
               "-D": 0b001111 << 6,
               "-A": 0b110011 << 6,
               "D+1": 0b011111 << 6,
               "A+1": 0b110111 << 6,
               "D-1": 0b001110 << 6,
               "A-1": 0b110010 << 6,
               "D+A": 0b000010 << 6,
               "D-A": 0b010011 << 6,
               "A-D": 0b000111 << 6,
               "D&A": 0b000000 << 6,
               "D|A": 0b010101 << 6}

BUILT_IN_SYMBOLS = {"R0": 0,
                    "R1": 1,
                    "R2": 2,
                    "R3": 3,
                    "R4": 4,
                    "R5": 5,
                    "R6": 6,
                    "R7": 7,
                    "R8": 8,
                    "R9": 9,
                    "R10": 10,
                    "R11": 11,
                    "R12": 12,
                    "R13": 13,
                    "R14": 14,
                    "R15": 15,
                    "SCREEN":16384,
                    "KBD": 24576,
                    "SP": 0,
                    "LCL": 1,
                    "ARG": 2,
                    "THIS": 3,
                    "THAT": 4}

A = 1 << 12
MAX_RAM = 2 ** 14
MAX_ADDRESS = 2 ** 15
C_INST = 7 << 13


def is_int(n: str) -> bool:
    try:
        int(n)
        return True
    except ValueError:
        return False


class Line:

    def __init__(self,
                original_line_number: int,
                original_input: str):
        self.__line_number : int = original_line_number
        self.__original_input : str = original_input
        self.__binary_value   : int = 0
        self.__line_to_parse = original_input

        self.__parse_input_line()
    
    def __parse_input_line(self):
        if self.__line_to_parse.startswith('@'):
            self.__parse_a_instruction()
        else:
            self.__parse_c_instruction()
    
    def __parse_a_instruction(self):
        self.__line_to_parse = self.__line_to_parse[1:]
        if is_int(self.__line_to_parse):
            self.__binary_value = int(self.__line_to_parse) % MAX_ADDRESS
        else:
            exception = f"Line {self.__line_number}: Invalid memory address\n{self.__original_input}"
            raise Exception(exception)

    def __parse_c_instruction(self):
        self.__binary_value |= C_INST
        self.__parse_jmp()
        self.__parse_dest()
        self.__parse_comp()
    
    def __parse_jmp(self):
        jmp_split = self.__line_to_parse.split(";")
        if len(jmp_split) > 2:
            exception = f"Line {self.__line_number}: Multiple ';' found, but only expected one\n"
            exception += self.__original_input
            arrows = ' ' * len(self.__original_input)
            indices = [idx for idx, c in enumerate(self.__original_input) if c == ';']
            for idx in indices:
                arrows = arrows[:idx] + '^' + arrows[idx+1:]
            exception += f"\n{arrows}"
            raise Exception(exception)
        elif len(jmp_split) == 2:
            jmp = jmp_split[1].upper()
            if jmp in JUMP_LOOKUP:
                self.__binary_value |= JUMP_LOOKUP[jmp]
            else:
                exception = f"Line {self.__line_number}: Jump instruction '{jmp_split[1]}' not recognised\n"
                exception += self.__original_input
                # TODO make arrows
                # arrows = ' ' * len(self.__original_input)
                raise Exception(exception)
        self.__line_to_parse = jmp_split[0]

    def __parse_dest(self):
        dest_split = self.__line_to_parse.split("=")
        dest_split.reverse()
        if len(dest_split) > 2:
            exception = f"Line {self.__line_number}: Multiple '=' found, but only expected one\n"
            exception += self.__original_input
            arrows = ' ' * len(self.__original_input)
            indices = [idx for idx, c in enumerate(self.__original_input) if c == '=']
            for idx in indices:
                arrows = arrows[:idx] + '^' + arrows[idx+1:]
            exception += f"\n{arrows}"
            raise Exception(exception)
        elif len(dest_split) == 2:
            dest = dest_split[1].upper()
            if dest in DEST_LOOKUP:
                self.__binary_value |= DEST_LOOKUP[dest]
            else:
                exception = f"Line {self.__line_number}: Jump instruction '{dest_split[1]}' not recognised\n"
                exception += self.__original_input
                # TODO make arrows
                # arrows = ' ' * len(self.__original_input)
                raise Exception(exception)
        self.__line_to_parse = dest_split[0]
    
    def __parse_comp(self):
        comp = self.__line_to_parse.upper()
        if 'M' in comp:
            self.__binary_value |= A
            comp = comp.replace('M', 'A')
        if comp in COMP_LOOKUP:
            self.__binary_value |= COMP_LOOKUP[comp]
        else:
            exception = f"Line {self.__line_number}: Comp instruction '{self.__line_to_parse}' not recognised\n"
            exception += self.__original_input
            # TODO make arrows
            # arrows = ' ' * len(self.__original_input)
            raise Exception(exception)
        self.__line_to_parse = ''


    @property
    def line_number(self) -> int:
        return self.__line_number

    @property
    def original_input(self) -> str:
        return self.__original_input
    
    @property
    def binary_value(self) -> int:
        return self.__binary_value
    
    def __str__(self):
        return "{0:b}".format(self.__binary_value).zfill(16)[-16:]


def remove_whitespace(lines: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    new_lines = []
    for line in lines:
        ln = line[0]
        text = line[1]
        text = re.sub(r"\s+", '', text)
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


def convert_symbols(lines: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    new_lines = []
    symbols = {}
    symbols_ln = {}
    next_variable_address = 16
    # Find Symbols
    i = 0
    for line in lines:
        ln = line[0]
        text = line[1]
        if text.startswith('(') and text.endswith(')'):
            symbol = text[1:-1]
            if symbol in BUILT_IN_SYMBOLS:
                raise Exception(f"Found duplicate symbol '{text}' on line {ln}. This duplicates a builtin and is not allowed")
            elif symbol in symbols:
                raise Exception(f"Found duplicate symbol '{text}' on line {ln}. This duplicates a previously defined on line {symbols_ln[symbol]}")
            symbols[symbol] = i
            symbols_ln[symbol] = ln
        else:
            new_lines.append((ln, text))
            i += 1
    
    # Replace @Symbol with @address
    for i in range(len(new_lines)):
        ln = new_lines[i][0]
        text = new_lines[i][1]
        if text.startswith('@'):
            symbol = text[1:]
            if symbol in BUILT_IN_SYMBOLS:
                text = f"@{BUILT_IN_SYMBOLS[symbol]}"
                new_lines[i] = (ln, text)
            elif symbol in symbols:
                text = f"@{symbols[symbol]}"
                new_lines[i] = (ln, text)
            elif not is_int(symbol):
                if next_variable_address == MAX_RAM:
                    raise Exception(f"Out of Memory. Tried to define variable '{text}' on line {ln}, however all 2**14 address have been used.")
                else:
                    symbols[symbol] = next_variable_address
                    text = f"@{next_variable_address}"
                    new_lines[i] = (ln, text)
                    next_variable_address += 1

    return new_lines


def parse_asm(input_file_path: str,
              output_file_path: str):
    with open(input_file_path, "r") as f:
        lines = list(enumerate(f.readlines()))
    lines = remove_whitespace(lines)
    lines = remove_comments(lines)
    lines = remove_empty(lines)
    lines = convert_symbols(lines)
    lines = [Line(x,y) for x,y in lines]
    with open(output_file_path, "w+") as f:
        for line in lines:
            f.write(str(line))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="asm_file_path", type=str, help="Filepath of .asm program to be converted to .hack")
    args = parser.parse_args()
    output_file_path = args.asm_file_path.rsplit('.', 1)[0] + ".hack"
    parse_asm(args.asm_file_path, output_file_path)
