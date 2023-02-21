#!/bin/env python3

from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from os import getcwd
from os.path import splitext
from typing import *
from shlex import join
from subprocess import call
from sys import argv, stdout, stderr

def fprintf(stream: IO, *args, **kwargs):
    print(*args, file=stream, **kwargs)

def compiler_base_info(title: str, info: str, stream: IO):
    print(f"[{title.upper()}] {info}", file=stream)

def compiler_info(title: str, info: str):
    compiler_base_info(title, info, stdout)

def compiler_error_info(info: str):
    compiler_base_info("error", info, stderr)

def run_cmd_with_log(cmd: List[str]):
    compiler_info("cmd", join(cmd))
    call(cmd, cwd=getcwd())

####### COMPILER

class OpType(Enum):
    PUSH_INT=auto()
    INTRINSIC=auto()
    IF=auto()
    ELSE=auto()
    WHILE=auto()
    DO=auto()
    END=auto()
    NOP=auto()

class Intrinsic(Enum):
    PLUS=auto()
    MINUS=auto()
    EQUAL=auto()
    NEQUAL=auto()
    PRINT=auto()
    EXIT=auto()
    DUP=auto()
    DIVMOD=auto()
    SWAP=auto()
    DROP=auto()

OpOperand=Union[int, Intrinsic]

@dataclass
class Op:
    typ: OpType
    operand: Optional[OpOperand] = None

Program=List[Op]

def generate_nasm_linux_x86_64(program: Program, stream: IO):
    assert len(OpType) == 8, "Not all operation types were handled in generate_nasm_linux_x86_64()"
    fprintf(stream, "BITS 64")
    fprintf(stream, "segment .text")
    fprintf(stream, "print:")
    fprintf(stream, "    mov     r9, -3689348814741910323")
    fprintf(stream, "    sub     rsp, 40")
    fprintf(stream, "    mov     BYTE [rsp+31], 10")
    fprintf(stream, "    lea     rcx, [rsp+30]")
    fprintf(stream, ".L2:")
    fprintf(stream, "    mov     rax, rdi")
    fprintf(stream, "    lea     r8, [rsp+32]")
    fprintf(stream, "    mul     r9")
    fprintf(stream, "    mov     rax, rdi")
    fprintf(stream, "    sub     r8, rcx")
    fprintf(stream, "    shr     rdx, 3")
    fprintf(stream, "    lea     rsi, [rdx+rdx*4]")
    fprintf(stream, "    add     rsi, rsi")
    fprintf(stream, "    sub     rax, rsi")
    fprintf(stream, "    add     eax, 48")
    fprintf(stream, "    mov     BYTE [rcx], al")
    fprintf(stream, "    mov     rax, rdi")
    fprintf(stream, "    mov     rdi, rdx")
    fprintf(stream, "    mov     rdx, rcx")
    fprintf(stream, "    sub     rcx, 1")
    fprintf(stream, "    cmp     rax, 9")
    fprintf(stream, "    ja      .L2")
    fprintf(stream, "    lea     rax, [rsp+32]")
    fprintf(stream, "    mov     edi, 1")
    fprintf(stream, "    sub     rdx, rax")
    fprintf(stream, "    xor     eax, eax")
    fprintf(stream, "    lea     rsi, [rsp+32+rdx]")
    fprintf(stream, "    mov     rdx, r8")
    fprintf(stream, "    mov     rax, 1")
    fprintf(stream, "    syscall")
    fprintf(stream, "    add     rsp, 40")
    fprintf(stream, "    ret")
    fprintf(stream, "global _start")
    fprintf(stream, "_start:")
    for ip, op in enumerate(program):
        comment = str(op.typ)
        if op.typ == OpType.INTRINSIC:
            comment += f" {str(op.operand)}"
        fprintf(stream, f";; -- {comment} --")
        fprintf(stream, f"addr_{ip}:")
        if op.typ == OpType.PUSH_INT:
            assert isinstance(op.operand, int), "This could be a bug in the parser"
            fprintf(stream, "mov rax, %d" % op.operand)
            fprintf(stream, "push rax")
        elif op.typ == OpType.IF:
            assert isinstance(op.operand, int), "This could be a bug in the parser"
            fprintf(stream, "pop rax")
            fprintf(stream, "test rax, rax")
            fprintf(stream, "jz addr_%d" % op.operand)
        elif op.typ == OpType.ELSE:
            assert isinstance(op.operand, int), "This could be a bug in the parser"
            fprintf(stream, "jmp addr_%d" % op.operand)
        elif op.typ == OpType.END:
            if op.operand:
                assert isinstance(op.operand, int), "This could be a bug in the parser"
                fprintf(stream, "jmp addr_%d" % op.operand)
        elif op.typ == OpType.WHILE:
            pass
        elif op.typ == OpType.DO:
            assert isinstance(op.operand, int), "This could be a bug in the parser"
            fprintf(stream, "pop rax")
            fprintf(stream, "test rax, rax")
            fprintf(stream, "jz addr_%d" % op.operand)
        elif op.typ == OpType.INTRINSIC:
            assert len(Intrinsic) == 10, "Not all intrinsics were handled in generate_nasm_linux_x86_64()"
            if op.operand == Intrinsic.PLUS:
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rbx")
                fprintf(stream, "add rax, rbx")
                fprintf(stream, "push rax")
            elif op.operand == Intrinsic.MINUS:
                fprintf(stream, "pop rbx")
                fprintf(stream, "pop rax")
                fprintf(stream, "sub rax, rbx")
                fprintf(stream, "push rax")
            elif op.operand == Intrinsic.EQUAL:
                fprintf(stream, "mov rcx, 0")
                fprintf(stream, "mov rdx, 1")
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rbx")
                fprintf(stream, "cmp rax, rbx")
                fprintf(stream, "cmove rcx, rdx")
                fprintf(stream, "push rcx")
            elif op.operand == Intrinsic.PRINT:
                fprintf(stream, "pop rdi")
                fprintf(stream, "call print")
            elif op.operand == Intrinsic.EXIT:
                fprintf(stream, "mov rax, 60")
                fprintf(stream, "pop rdi")
                fprintf(stream, "syscall")
            elif op.operand == Intrinsic.DUP:
                fprintf(stream, "pop rax")
                fprintf(stream, "push rax")
                fprintf(stream, "push rax")
            elif op.operand == Intrinsic.NEQUAL:
                fprintf(stream, "mov rcx, 0")
                fprintf(stream, "mov rdx, 1")
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rbx")
                fprintf(stream, "cmp rax, rbx")
                fprintf(stream, "cmovne rcx, rdx")
                fprintf(stream, "push rcx")
            elif op.operand == Intrinsic.DIVMOD:
                fprintf(stream, "xor rdx, rdx")
                fprintf(stream, "pop rbx")
                fprintf(stream, "pop rax")
                fprintf(stream, "div rbx")
                fprintf(stream, "push rax")
                fprintf(stream, "push rdx")
            elif op.operand == Intrinsic.SWAP:
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rbx")
                fprintf(stream, "push rax")
                fprintf(stream, "push rbx")
            elif op.operand == Intrinsic.DROP:
                fprintf(stream, "pop rax")
            else:
                raise Exception('Unreachable')
        elif op.typ == OpType.NOP:
            pass
        else:
            raise Exception('Unreachable')

####### LEXER

class TokenType(Enum):
    INT=auto()
    WORD=auto()
    KEYWORD=auto()

class Keyword(Enum):
    IF=auto()
    ELSE=auto()
    WHILE=auto()
    DO=auto()
    MACRO=auto()
    END=auto()

assert len(Keyword) == 6, "Not all keyword types were handled in NAME_TO_KEYWORD_TABLE"
NAME_TO_KEYWORD_TABLE: Dict[str, Keyword] = {
    'if': Keyword.IF,
    'else': Keyword.ELSE,
    'while': Keyword.WHILE,
    'do': Keyword.DO,
    'macro': Keyword.MACRO,
    'end': Keyword.END,
}

TokenValue=Union[int, str, Keyword]
TokenLoc=Tuple[str, int, int]

@dataclass
class Token:
    typ: TokenType
    value: TokenValue
    loc: TokenLoc

# TODO: Find a way to compress that

def compiler_error_base(token: Token, error: str):
    fprintf(stderr, "%s:%d:%d: ERROR: %s" % (token.loc + (error, )))

def compiler_error_unknown_word(token: Token):
    compiler_error_base(token, f"unknown word: `{token.value}`")

def compiler_error_unclosed_block(loc: TokenLoc):
    fprintf(stderr, "%s:%d:%d: ERROR: unclosed block" % loc)

def compiler_error_end_cant_close(token: Token):
    compiler_error_base(token, "`end` can only close `if`, `if-else` and `while-do` blocks")

def compiler_error_else_cant_close(token: Token):
    compiler_error_base(token, "`else` can only close `if` blocks")

def compiler_error_else_no_if(token: Token):
    compiler_error_base(token, "`else` is not preceeded by `if`")

def compiler_error_do_cant_close(token: Token):
    compiler_error_base(token, "`do` can only close `while` blocks")

def compiler_error_do_no_while(token: Token):
    compiler_error_base(token, "`do` is not preceeded by `while`")

def compiler_error_end_no_block(token: Token):
    compiler_error_base(token, "`end` has no block to close")

def compiler_error_macro_unfinished(token: Token):
    compiler_error_base(token, "unfinished macro definition")

def compiler_error_name_not_word(token: Token):
    compiler_error_base(token, "macro name must be a word")

def compiler_error_macro_redefinition(token: Token, loc: TokenLoc):
    compiler_error_base(token, f"redefinition of macro `{token.value}`")
    print("%s:%d:%d: NOTE: original definition located here" % loc, file=stderr)

def advance_loc(char: str, r: int, c: int) -> Tuple[int, int]:
    c += 1
    if char == '\n':
        r += 1
        c = 0
    return (r, c)

def lex_string(string: str, file_loc_name: str) -> List[Token]:
    string += ' '
    tokens: List[Token] = []
    char = 0
    c = 0
    r = 0
    word = ""
    while char < len(string):
        if string[char].isspace():
            if word != '':
                loc = (file_loc_name, r+1, c-len(word)+1)
                try: t = Token(TokenType.INT, int(word), loc)
                except:
                    if word in NAME_TO_KEYWORD_TABLE:
                        t = Token(TokenType.KEYWORD, NAME_TO_KEYWORD_TABLE[word], loc)
                    else:
                        t = Token(TokenType.WORD, word, loc)
                tokens.append(t)
            word = ""
        else:
            word += string[char]
        r, c = advance_loc(string[char], r, c)
        char += 1
    return tokens

def lex_file(file: str) -> List[Token]:
    with open(file, "r") as f:
        return lex_string(f.read(), file)

####### PARSER

assert len(Intrinsic) == 10, "Not all intrinsics were handled in INTRINSICS_TABLE"
INTRINSICS_TABLE: Dict[str, Intrinsic] = {
    'print': Intrinsic.PRINT,
    '+': Intrinsic.PLUS,
    '-': Intrinsic.MINUS,
    '=': Intrinsic.EQUAL,
    '!=': Intrinsic.NEQUAL,
    'divmod': Intrinsic.DIVMOD,
    'exit': Intrinsic.EXIT,
    'dup': Intrinsic.DUP,
    'swap': Intrinsic.SWAP,
    'drop': Intrinsic.DROP,
}

@dataclass
class Macro:
    name: str
    tokens: List[Token]
    loc: TokenLoc

def parse_tokens_into_program(tokens: List[Token]) -> Program:
    rtokens = list(reversed(tokens))
    macros: Dict[str, Macro] = {}
    program: Program = []
    block_stack: List[Tuple[int, TokenLoc]] = []
    while len(rtokens) > 0:
        token = rtokens.pop()
        assert len(TokenType) == 3, "Not all token types were handled in parse_tokens_into_program()"
        if token.typ == TokenType.INT:
            assert isinstance(token.value, int), "This could be a bug in the lexer"
            program.append(Op(typ=OpType.PUSH_INT, operand=token.value))
        elif token.typ == TokenType.WORD:
            assert isinstance(token.value, str), "This could be a bug in the lexer"
            if token.value in INTRINSICS_TABLE:
                program.append(Op(typ=OpType.INTRINSIC, operand=INTRINSICS_TABLE[token.value]))
            elif token.value in macros:
                macro = macros[token.value]
                rtokens += list(reversed(macro.tokens))
            else:
                compiler_error_unknown_word(token)
                exit(1)
        elif token.typ == TokenType.KEYWORD:
            assert len(Keyword) == 6, "Not all keyword types were handled in parse_tokens_into_program()"
            if token.value == Keyword.IF:
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.IF))
            elif token.value == Keyword.ELSE:
                if len(block_stack) < 1:
                    compiler_error_else_no_if(token)
                    exit(1)
                if_ip, _ = block_stack.pop()
                if program[if_ip].typ != OpType.IF:
                    compiler_error_else_cant_close(token)
                    exit(1)
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.ELSE))
                program[if_ip].operand = len(program)
            elif token.value == Keyword.WHILE:
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.WHILE))
            elif token.value == Keyword.DO:
                if len(block_stack) < 1:
                    compiler_error_do_no_while(token)
                    exit(1)
                while_ip, _ = block_stack.pop()
                if program[while_ip].typ != OpType.WHILE:
                    compiler_error_do_cant_close(token)
                    exit(1)
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.DO, operand=while_ip))
            elif token.value == Keyword.MACRO:
                if len(rtokens) < 1:
                    compiler_error_macro_unfinished(token)
                    exit(1)
                macro_name_token = rtokens.pop()
                if macro_name_token.typ != TokenType.WORD:
                    compiler_error_name_not_word(macro_name_token)
                    exit(1)
                macro_loc = token.loc
                macro_name = macro_name_token.value
                macro_tokens: List[Token] = []
                nesting_depth: int = 0
                while True:
                    if len(rtokens) < 1:
                        compiler_error_macro_unfinished(token)
                        exit(1)
                    ntoken = rtokens.pop()
                    if ntoken.typ == TokenType.KEYWORD and ntoken.value == Keyword.END and nesting_depth == 0:
                        break
                    if ntoken.typ == TokenType.KEYWORD:
                        assert len(Keyword) == 6, "Not all keyword types were handled while parsing macro body"
                        if ntoken.value in [Keyword.IF, Keyword.WHILE, Keyword.MACRO]:
                            nesting_depth += 1
                        elif ntoken.value == Keyword.END:
                            nesting_depth -= 1
                    macro_tokens.append(ntoken)
                if macro_name in macros:
                    compiler_error_macro_redefinition(macro_name_token, macros[macro_name].loc)
                    exit(1)
                macros[macro_name] = Macro(name=macro_name, tokens=macro_tokens, loc=macro_loc)
            elif token.value == Keyword.END:
                if len(block_stack) < 1:
                    compiler_error_end_no_block(token)
                    exit(1)
                block_ip, _ = block_stack.pop()
                if program[block_ip].typ in [OpType.IF, OpType.ELSE]:
                    program.append(Op(typ=OpType.END))
                    program[block_ip].operand = len(program)
                elif program[block_ip].typ == OpType.DO:
                    program.append(Op(typ=OpType.END, operand=program[block_ip].operand))
                    program[block_ip].operand = len(program)
                else:
                    compiler_error_end_cant_close(token)
                    exit(1)
            else:
                raise Exception('unreachable')
        else:
            raise Exception('unreachable')
    if len(block_stack) != 0:
        _, loc = block_stack.pop()
        compiler_error_unclosed_block(loc)
        exit(1)
    program.append(Op(typ=OpType.NOP))
    return program

####### MAIN

def usage(stream: IO, myname: str):
    fprintf(stream, f"USAGE: {myname} <SUBCOMMAND>")
    fprintf(stream, "  SUBCOMMANDS:")
    fprintf(stream, "    help                   Prints this help and exits with 0 exit code")
    fprintf(stream, "    com [OPTIONS] <file>   Compile <file>")
    fprintf(stream, "      OPTIONS:")
    fprintf(stream, "        -r          Run the compiled executable after successful compilation")
    fprintf(stream, "        -o <file>   Change the output executable name to <file>")

if __name__ == '__main__':
    args = copy(argv)
    myname = args.pop(0)

    if len(args) < 1:
        compiler_error_info("no subcommand provided")
        usage(stderr, myname)
        exit(1)

    subcommand = args.pop(0)
    if subcommand == "com":
        if len(args) < 1:
            compiler_error_info("no input file was provided")
            usage(stderr, myname)
            exit(1)

        filename = ""
        outputfilename = ""
        next_arg = args.pop(0)
        run = False
        if next_arg.startswith('-'):
            while next_arg[0] == '-': next_arg = next_arg[1:]
            if next_arg == "r":
                run = True
            elif next_arg == "o":
                if len(args) < 1:
                    compiler_error_info("no output file name was provided")
                    usage(stderr, myname)
                    exit(1)
                outputfilename = args.pop(0)
            else:
                compiler_error_info(f"unknown flag: `{next_arg}`")
                usage(stderr, myname)
                exit(1)

            if len(args) < 1:
                compiler_error_info("no input file was provided")
                usage(stderr, myname)
                exit(1)
            filename = args.pop(0)
        else:
            filename = next_arg

        basefilename = splitext(filename)[0]
        if outputfilename != "":
            basefilename = outputfilename
        output = f"{basefilename}.asm"
        o_filename = f"{basefilename}.o"

        program = parse_tokens_into_program(lex_file(filename))
        file = open(output, "w")
        compiler_info("info", f"Generating `{output}`")
        generate_nasm_linux_x86_64(program, file)
        file.close()

        run_cmd_with_log(["nasm", "-felf64", output])
        run_cmd_with_log(["ld", "-o", basefilename, o_filename])
        if run:
            run_cmd_with_log([f"./{basefilename}"])
    elif subcommand == "help":
        usage(stdout, myname)
        exit(0)
    else:
        compiler_error_info(f"unknown subcommand: `{subcommand}`")
        exit(1)
