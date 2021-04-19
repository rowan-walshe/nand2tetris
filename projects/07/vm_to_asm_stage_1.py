import argparse

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

    def __init__(self,
                 original_line_number: int,
                 original_input: str):
        self.__line_number: int = original_line_number
        self.__original_input: str = original_input
        self.__asm: Tuple = None
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
        if self.__asm is None:
            raise VM_Error(f"Failed to convert VM command: '{self.__original_input}'")
        return self.__asm

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
        asm = [None, "D=M", "@SP", "M=D"]
        if parts[1] == "local":
            pass
        elif parts[1] == "argument":
            pass
        elif parts[1] == "this":
            pass
        elif parts[1] == "that":
            pass
        elif parts[1] == "constant":
            asm[0] = f"@{parts[2]}"
            asm[1] = "D=A"
        elif parts[1] == "static":
            pass
        elif parts[1] == "pointer":
            pass
        elif parts[1] == "temp":
            pass
        if None in asm:
            raise VM_Error(f"Push from memory segment '{parts[1]}' not yet implemented")
        self.__asm = tuple(asm)

    def _vm_pop(self):
        parts = self.__validate_push_pop()
        pass

    def _vm_add(self):
        self.__asm = ("Add not yet implemented",)

    def _vm_sub(self):
        pass

    def _vm_neg(self):
        pass

    def _vm_eq(self):
        pass

    def _vm_gt(self):
        pass

    def _vm_lt(self):
        pass

    def _vm_and(self):
        pass

    def _vm_or(self):
        pass

    def _vm_not(self):
        pass


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


def parse_vm(lines: List[str]) -> List[VMLine]:
    lines = list(enumerate(lines))
    lines = remove_unwanted_whitespace(lines)
    lines = remove_comments(lines)
    lines = remove_empty(lines)
    lines = [VMLine(x, y) for x, y in lines]
    return lines


def convert_vm_to_asm(vm_lines: List[VMLine]) -> List[str]:
    asm_lines = []
    for line in vm_lines:
        asm_lines.extend(line.get_asm())
    return asm_lines


def parse_file(input_file_path: str,
               output_file_path: str):
    with open(input_file_path, "r") as f:
        lines = f.readlines()
    lines = parse_vm(lines)
    lines = convert_vm_to_asm(lines)
    with open(output_file_path, "w+") as f:
        for line in lines:
            f.write(str(line))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="vm_file_path", type=str, help="Filepath of .vm program to be converted to .hack")
    args = parser.parse_args()
    output_file_path = args.vm_file_path.rsplit('.', 1)[0] + ".hack"
    parse_file(args.vm_file_path, output_file_path)
