#!/bin/env python3

from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from os import getcwd, getenv
from os.path import splitext, isfile
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
    PUSH_STR=auto()
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
    SYSCALL0=auto()
    SYSCALL1=auto()
    SYSCALL2=auto()
    SYSCALL3=auto()
    SYSCALL4=auto()
    SYSCALL5=auto()
    SYSCALL6=auto()

OpOperand=Union[int, Intrinsic, str]

@dataclass
class Op:
    typ: OpType
    operand: Optional[OpOperand] = None

Program=List[Op]

def generate_nasm_linux_x86_64(program: Program, stream: IO):
    assert len(OpType) == 9, "Not all operation types were handled in generate_nasm_linux_x86_64()"
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
    strs: List[str] = []
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
        elif op.typ == OpType.PUSH_STR:
            assert isinstance(op.operand, str), "This could be a bug in the parser"
            fprintf(stream, f"push {len(op.operand)}")
            fprintf(stream, f"push str_{len(strs)}")
            strs.append(op.operand)
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
            assert len(Intrinsic) == 17, "Not all intrinsics were handled in generate_nasm_linux_x86_64()"
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
            elif op.operand == Intrinsic.SYSCALL0:
                fprintf(stream, "pop rax")
                fprintf(stream, "syscall")
                fprintf(stream, "push rax")
            elif op.operand == Intrinsic.SYSCALL1:
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rdi")
                fprintf(stream, "syscall")
                fprintf(stream, "push rax")
            elif op.operand == Intrinsic.SYSCALL2:
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rdi")
                fprintf(stream, "pop rsi")
                fprintf(stream, "syscall")
                fprintf(stream, "push rax")
            elif op.operand == Intrinsic.SYSCALL3:
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rdi")
                fprintf(stream, "pop rsi")
                fprintf(stream, "pop rdx")
                fprintf(stream, "syscall")
                fprintf(stream, "push rax")
            elif op.operand == Intrinsic.SYSCALL4:
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rdi")
                fprintf(stream, "pop rsi")
                fprintf(stream, "pop rdx")
                fprintf(stream, "pop r10")
                fprintf(stream, "syscall")
                fprintf(stream, "push rax")
            elif op.operand == Intrinsic.SYSCALL5:
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rdi")
                fprintf(stream, "pop rsi")
                fprintf(stream, "pop rdx")
                fprintf(stream, "pop r10")
                fprintf(stream, "pop r8")
                fprintf(stream, "syscall")
                fprintf(stream, "push rax")
            elif op.operand == Intrinsic.SYSCALL6:
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rdi")
                fprintf(stream, "pop rsi")
                fprintf(stream, "pop rdx")
                fprintf(stream, "pop r10")
                fprintf(stream, "pop r8")
                fprintf(stream, "pop r9")
                fprintf(stream, "syscall")
                fprintf(stream, "push rax")
            else:
                raise Exception('Unreachable')
        elif op.typ == OpType.NOP:
            pass
        else:
            raise Exception('Unreachable')
    for i, s in enumerate(strs):
        stream.write(f"str_{i}: db ")
        for c in s:
            stream.write(f"{hex(ord(c))},")
        stream.write("0x00\n")

def generate_c_linux_x86_64(program: Program, stream: IO):
    assert len(OpType) == 9, "Not all operation types were handled in generate_c_linux_x86_64()"
    fprintf(stream, "#include <stdio.h>")
    fprintf(stream, "#include <stdint.h>")
    fprintf(stream, "#include <stdlib.h>")
    fprintf(stream, "#include <unistd.h>")
    fprintf(stream, "")
    fprintf(stream, "#define STACK_CAPACITY 640000")
    fprintf(stream, "static int64_t stack[STACK_CAPACITY] = {0};")
    fprintf(stream, "size_t stack_count = 0;")
    fprintf(stream, "")
    fprintf(stream, "void push(int64_t value) {")
    fprintf(stream, "    stack[stack_count++] = value;")
    fprintf(stream, "}")
    fprintf(stream, "")
    fprintf(stream, "int64_t pop() {")
    fprintf(stream, "    return stack[--stack_count];")
    fprintf(stream, "}")
    fprintf(stream, "")
    fprintf(stream, "int main(int argc, const int64_t** argv) {")
    fprintf(stream, "    (void) argc;")
    fprintf(stream, "    (void) argv;")
    for ip, op in enumerate(program):
        comment = str(op.typ)
        if op.typ == OpType.INTRINSIC:
            comment += f" {str(op.operand)}"
        fprintf(stream, f"    // -- {comment} --")
        if op.typ == OpType.PUSH_INT:
            fprintf(stream, f"    push({op.operand});")
        elif op.typ == OpType.PUSH_STR:
            raise NotImplementedError
        elif op.typ == OpType.INTRINSIC:
            assert len(Intrinsic) == 17, "Not all intrinsics were handled in generate_c_linux_x86_64()"
            if op.operand == Intrinsic.PLUS:
                fprintf(stream, "    {")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        int b = pop();")
                fprintf(stream, "        push(a + b);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.MINUS:
                fprintf(stream, "    {")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        int b = pop();")
                fprintf(stream, "        push(b - a);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.EQUAL:
                fprintf(stream, "    {")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        int b = pop();")
                fprintf(stream, "        push(a == b);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.NEQUAL:
                fprintf(stream, "    {")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        int b = pop();")
                fprintf(stream, "        push(a != b);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.PRINT:
                fprintf(stream, "    printf(\"%ld\\n\", pop());")
            elif op.operand == Intrinsic.EXIT:
                fprintf(stream, "    exit(pop());")
            elif op.operand == Intrinsic.DUP:
                fprintf(stream, "    {")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        push(a);")
                fprintf(stream, "        push(a);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.DIVMOD:
                fprintf(stream, "    {")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        int b = pop();")
                fprintf(stream, "        push(a / b);")
                fprintf(stream, "        push(a % b);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SWAP:
                fprintf(stream, "    {")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        int b = pop();")
                fprintf(stream, "        push(a);")
                fprintf(stream, "        push(b);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.DROP:
                fprintf(stream, "    pop();")
            elif op.operand == Intrinsic.SYSCALL0:
                fprintf(stream, "    {")
                fprintf(stream, "        int sys = pop();")
                fprintf(stream, "        syscall(sys);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL1:
                fprintf(stream, "    {")
                fprintf(stream, "        int sys = pop();")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        syscall(sys, a);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL2:
                fprintf(stream, "    {")
                fprintf(stream, "        int sys = pop();")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        int b = pop();")
                fprintf(stream, "        syscall(sys, a, b);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL3:
                fprintf(stream, "    {")
                fprintf(stream, "        int sys = pop();")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        int b = pop();")
                fprintf(stream, "        int c = pop();")
                fprintf(stream, "        syscall(sys, a, b, c);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL4:
                fprintf(stream, "    {")
                fprintf(stream, "        int sys = pop();")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        int b = pop();")
                fprintf(stream, "        int c = pop();")
                fprintf(stream, "        int d = pop();")
                fprintf(stream, "        syscall(sys, a, b, c, d);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL5:
                fprintf(stream, "    {")
                fprintf(stream, "        int sys = pop();")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        int b = pop();")
                fprintf(stream, "        int c = pop();")
                fprintf(stream, "        int d = pop();")
                fprintf(stream, "        int e = pop();")
                fprintf(stream, "        syscall(sys, a, b, c, d, e);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL6:
                fprintf(stream, "    {")
                fprintf(stream, "        int sys = pop();")
                fprintf(stream, "        int a = pop();")
                fprintf(stream, "        int b = pop();")
                fprintf(stream, "        int c = pop();")
                fprintf(stream, "        int d = pop();")
                fprintf(stream, "        int e = pop();")
                fprintf(stream, "        int f = pop();")
                fprintf(stream, "        syscall(sys, a, b, c, d, e, f);")
                fprintf(stream, "    }")
        elif op.typ == OpType.IF:
            raise NotImplementedError
        elif op.typ == OpType.ELSE:
            raise NotImplementedError
        elif op.typ == OpType.WHILE:
            raise NotImplementedError
        elif op.typ == OpType.DO:
            raise NotImplementedError
        elif op.typ == OpType.END:
            raise NotImplementedError
        elif op.typ == OpType.NOP:
            pass
    fprintf(stream, "}")

####### LEXER

class TokenType(Enum):
    INT=auto()
    WORD=auto()
    KEYWORD=auto()
    STR=auto()

class Keyword(Enum):
    IF=auto()
    ELSE=auto()
    WHILE=auto()
    DO=auto()
    MACRO=auto()
    END=auto()
    INCLUDE=auto()

assert len(Keyword) == 7, "Not all keyword types were handled in NAME_TO_KEYWORD_TABLE"
NAME_TO_KEYWORD_TABLE: Dict[str, Keyword] = {
    'if': Keyword.IF,
    'else': Keyword.ELSE,
    'while': Keyword.WHILE,
    'do': Keyword.DO,
    'macro': Keyword.MACRO,
    'end': Keyword.END,
    'include': Keyword.INCLUDE,
}

TokenValue=Union[int, str, Keyword]
TokenLoc=Tuple[str, int, int]

@dataclass
class Token:
    typ: TokenType
    value: TokenValue
    loc: TokenLoc

# TODO: Find a way to compress that

def compiler_error(loc: TokenLoc, error: str):
    fprintf(stderr, "%s:%d:%d: ERROR: %s" % (loc + (error, )))

def compiler_note(loc: TokenLoc, note: str):
    fprintf(stderr, "%s:%d:%d: NOTE: %s" % (loc + (note, )))

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
        elif string[char] == '"':
            loc = (file_loc_name, r+1, c+1)
            r, c = advance_loc(string[char], r, c)
            char += 1
            lit = ""
            while string[char] != '"':
                lit += string[char]
                r, c = advance_loc(string[char], r, c)
                char += 1
                if char >= len(string):
                    compiler_error(loc, "unclosed string")
                    exit(1)
            tokens.append(Token(TokenType.STR, bytes(lit, 'utf-8').decode('unicode_escape'), loc))
        else:
            word += string[char]
        r, c = advance_loc(string[char], r, c)
        char += 1
    return tokens

def lex_file(file: str) -> List[Token]:
    with open(file, "r") as f:
        return lex_string(f.read(), file)

####### PARSER

HOME = getenv("HOME")
INCLUDE_SEARCH_PATHS: List[str] = ["./", "./std/", f"{HOME}/.local/include/north/", "/usr/local/include/north/"]

assert len(Intrinsic) == 17, "Not all intrinsics were handled in INTRINSICS_TABLE"
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
    'syscall0': Intrinsic.SYSCALL0,
    'syscall1': Intrinsic.SYSCALL1,
    'syscall2': Intrinsic.SYSCALL2,
    'syscall3': Intrinsic.SYSCALL3,
    'syscall4': Intrinsic.SYSCALL4,
    'syscall5': Intrinsic.SYSCALL5,
    'syscall6': Intrinsic.SYSCALL6,
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
        assert len(TokenType) == 4, "Not all token types were handled in parse_tokens_into_program()"
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
                compiler_error(token.loc, f"unknown word: `{token.value}`")
                exit(1)
        elif token.typ == TokenType.STR:
            assert isinstance(token.value, str), "This could be a bug in the lexer"
            program.append(Op(typ=OpType.PUSH_STR, operand=token.value))
        elif token.typ == TokenType.KEYWORD:
            assert len(Keyword) == 7, "Not all keyword types were handled in parse_tokens_into_program()"
            if token.value == Keyword.IF:
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.IF))
            elif token.value == Keyword.ELSE:
                if len(block_stack) < 1:
                    compiler_error(token.loc, "`else` is not preceeded by `if`")
                    exit(1)
                if_ip, _ = block_stack.pop()
                if program[if_ip].typ != OpType.IF:
                    compiler_error(token.loc, "`else` can only close `if` blocks")
                    exit(1)
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.ELSE))
                program[if_ip].operand = len(program)
            elif token.value == Keyword.WHILE:
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.WHILE))
            elif token.value == Keyword.DO:
                if len(block_stack) < 1:
                    compiler_error(token.loc, "`do` is not preceeded by `while`")
                    exit(1)
                while_ip, _ = block_stack.pop()
                if program[while_ip].typ != OpType.WHILE:
                    compiler_error(token.loc, "`do` can only close `while` blocks")
                    exit(1)
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.DO, operand=while_ip))
            elif token.value == Keyword.MACRO:
                if len(rtokens) < 1:
                    compiler_error(token.loc, "unfinished macro definition")
                    exit(1)
                macro_name_token = rtokens.pop()
                if macro_name_token.typ != TokenType.WORD:
                    compiler_error(macro_name_token.loc, "expected name to be a word")
                    exit(1)
                macro_loc = token.loc
                assert isinstance(macro_name_token.value, str), "This could be a bug in the lexer"
                macro_name = macro_name_token.value
                macro_tokens: List[Token] = []
                nesting_depth: int = 0
                while True:
                    if len(rtokens) < 1:
                        compiler_error(token.loc, "unfinished macro definition")
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
                    compiler_error(macro_name_token.loc, "redefinition of already existing macro")
                    compiler_note(macros[macro_name].loc, "original definition located here")
                    exit(1)
                if macro_name in INTRINSICS_TABLE:
                    compiler_error(macro_name_token.loc, "redefinition of built-in intrinsic")
                    exit(1)
                macros[macro_name] = Macro(name=macro_name, tokens=macro_tokens, loc=macro_loc)
            elif token.value == Keyword.INCLUDE:
                if len(rtokens) < 1:
                    compiler_error(token.loc, "no file path provided for inclusion")
                    exit(1)
                path_token = rtokens.pop()
                if path_token.typ != TokenType.STR:
                    compiler_error(token.loc, "file path for `include` must be a string")
                    exit(1)
                assert isinstance(path_token.value, str), "This could be a bug in the lexer"
                file_path = path_token.value
                include_path = file_path
                for p in INCLUDE_SEARCH_PATHS:
                    path = p + file_path
                    if isfile(path):
                        include_path = path
                        break
                rtokens += list(reversed(lex_file(include_path)))
            elif token.value == Keyword.END:
                if len(block_stack) < 1:
                    compiler_error(token.loc, "`end` has no block to close")
                    exit(1)
                block_ip, _ = block_stack.pop()
                if program[block_ip].typ in [OpType.IF, OpType.ELSE]:
                    program.append(Op(typ=OpType.END))
                    program[block_ip].operand = len(program)
                elif program[block_ip].typ == OpType.DO:
                    program.append(Op(typ=OpType.END, operand=program[block_ip].operand))
                    program[block_ip].operand = len(program)
                else:
                    compiler_error(token.loc, "`end` can only close `while-do`, `if` or `if-else` blocks")
                    exit(1)
            else:
                raise Exception('unreachable')
        else:
            raise Exception('unreachable')
    if len(block_stack) != 0:
        _, loc = block_stack.pop()
        compiler_error(loc, "unclosed block")
        exit(1)
    program += [Op(typ=OpType.PUSH_INT, operand=0), Op(typ=OpType.INTRINSIC, operand=Intrinsic.EXIT)]
    return program

####### MAIN

def usage(stream: IO, myname: str):
    fprintf(stream, f"USAGE: {myname} <SUBCOMMAND>")
    fprintf(stream, "  SUBCOMMANDS:")
    fprintf(stream, "    help                   Prints this help and exits with 0 exit code")
    fprintf(stream, "    com [OPTIONS] <file>   Compile <file>")
    fprintf(stream, "      OPTIONS:")
    fprintf(stream, "        -r                 Run the compiled executable after successful compilation")
    fprintf(stream, "        -o <file>          Change the output executable name to <file>")
    fprintf(stream, "        -target <target>   Change the compilation target to target")
    fprintf(stream, "        TARGETS:")
    fprintf(stream, "          c      Generates C code and then compiles with GCC")
    fprintf(stream, "          nasm   Generates assembly code and then compiles with nasm")

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
        run = False
        target = "nasm"

        next_arg = args.pop(0)
        if not next_arg.startswith('-'):
            filename = next_arg
        else:
            while True:
                while next_arg[0] == '-': next_arg = next_arg[1:]
                if next_arg == "o":
                    if len(args) < 1:
                        compiler_error_info("no output file was provided")
                        usage(stderr, myname)
                        exit(1)
                    outputfilename = args.pop(0)
                elif next_arg == "r":
                    run = True
                elif next_arg == "target":
                    if len(args) < 1:
                        compiler_error_info("no compilation target was provided")
                        usage(stderr, myname)
                        exit(1)
                    target = args.pop(0)
                else:
                    compiler_error_info(f"unknown flag: `{next_arg}`")
                    usage(stderr, myname)
                    exit(1)
                if len(args) < 1:
                    compiler_error_info("no input file was provided")
                    usage(stderr, myname)
                    exit(1)
                next_arg = args.pop(0)
                if not next_arg.startswith('-'):
                    break
            filename = next_arg

        basefilename = splitext(filename)[0]
        if outputfilename != "":
            basefilename = outputfilename
        program = parse_tokens_into_program(lex_file(filename))
        if target == "nasm":
            output = f"{basefilename}.asm"
            o_filename = f"{basefilename}.o"

            file = open(output, "w")
            compiler_info("info", f"Generating `{output}`")
            generate_nasm_linux_x86_64(program, file)
            file.close()

            run_cmd_with_log(["nasm", "-felf64", output])
            run_cmd_with_log(["ld", "-o", basefilename, o_filename])
        elif target == "c":
            output = f"{basefilename}.c"

            file = open(output, "w")
            compiler_info("info", f"Generating `{output}`")
            generate_c_linux_x86_64(program, file)
            file.close()

            run_cmd_with_log(["gcc", "-o", basefilename, output])
        else:
            compiler_error_info(f"unknown compilation target: `{target}`")
            usage(stderr, myname)
            exit(1)

        if run:
            run_cmd_with_log([f"./{basefilename}"])
    elif subcommand == "help":
        usage(stdout, myname)
        exit(0)
    else:
        compiler_error_info(f"unknown subcommand: `{subcommand}`")
        exit(1)
