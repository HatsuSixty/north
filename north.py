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
    MEMORY=auto()
    END=auto()
    INCLUDE=auto()
    CALL=auto()
    CVAR=auto()

assert len(Keyword) == 10, "Not all keyword types were handled in NAME_TO_KEYWORD_TABLE"
NAME_TO_KEYWORD_TABLE: Dict[str, Keyword] = {
    'if': Keyword.IF,
    'else': Keyword.ELSE,
    'while': Keyword.WHILE,
    'do': Keyword.DO,
    'macro': Keyword.MACRO,
    'memory': Keyword.MEMORY,
    'end': Keyword.END,
    'call': Keyword.CALL,
    'cvar': Keyword.CVAR,
    'include': Keyword.INCLUDE,
}

TokenValue=Union[int, str, Keyword]
TokenLoc=Tuple[str, int, int]

@dataclass
class Token:
    typ: TokenType
    value: TokenValue
    loc: TokenLoc

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
    string += ' \n'
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
        elif string[char] == ';':
            while string[char] != '\n':
                r, c = advance_loc(string[char], r, c)
                char += 1
        else:
            word += string[char]
        r, c = advance_loc(string[char], r, c)
        char += 1
    return tokens

def lex_file(file: str) -> List[Token]:
    with open(file, "r") as f:
        return lex_string(f.read(), file)

####### COMPILER

class OpType(Enum):
    PUSH_INT=auto()
    PUSH_STR=auto()
    PUSH_MEM=auto()
    INTRINSIC=auto()
    IF=auto()
    ELSE=auto()
    WHILE=auto()
    DO=auto()
    END=auto()
    CALL0=auto()
    CALL1=auto()
    CALL2=auto()
    CALL3=auto()
    CALL4=auto()
    CALL5=auto()
    CALL6=auto()
    CVAR=auto()

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
    LOAD8=auto()
    LOAD16=auto()
    LOAD32=auto()
    LOAD64=auto()
    STORE8=auto()
    STORE16=auto()
    STORE32=auto()
    STORE64=auto()

@dataclass
class CallOperand:
    func: str
    returns: bool

OpOperand=Union[int, Intrinsic, CallOperand, str]

@dataclass
class Op:
    typ: OpType
    loc: TokenLoc
    operand: Optional[OpOperand] = None

Program=List[Op]

def generate_nasm_linux_x86_64(program: Program, memory_size: int, stream: IO):
    assert len(OpType) == 17, "Not all operation types were handled in generate_nasm_linux_x86_64()"
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
        elif op.typ == OpType.PUSH_MEM:
            fprintf(stream, f"push mem+{op.operand}")
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
            assert len(Intrinsic) == 25, "Not all intrinsics were handled in generate_nasm_linux_x86_64()"
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
            elif op.operand == Intrinsic.LOAD8:
                fprintf(stream, "pop rax")
                fprintf(stream, "xor rbx, rbx")
                fprintf(stream, "mov bl, [rax]")
                fprintf(stream, "push rbx")
            elif op.operand == Intrinsic.LOAD16:
                raise NotImplementedError
            elif op.operand == Intrinsic.LOAD32:
                raise NotImplementedError
            elif op.operand == Intrinsic.LOAD64:
                fprintf(stream, "pop rax")
                fprintf(stream, "xor rbx, rbx")
                fprintf(stream, "mov rbx, [rax]")
                fprintf(stream, "push rbx")
            elif op.operand == Intrinsic.STORE8:
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rbx")
                fprintf(stream, "mov [rbx], al")
            elif op.operand == Intrinsic.STORE16:
                raise NotImplementedError
            elif op.operand == Intrinsic.STORE32:
                raise NotImplementedError
            elif op.operand == Intrinsic.STORE64:
                fprintf(stream, "pop rax")
                fprintf(stream, "pop rbx")
                fprintf(stream, "mov [rbx], rax")
            else:
                raise Exception('Unreachable')
        elif op.typ in [OpType.CALL0,
                        OpType.CALL1,
                        OpType.CALL2,
                        OpType.CALL3,
                        OpType.CALL4,
                        OpType.CALL5,
                        OpType.CALL6]:
            compiler_error(op.loc, "`call` instruction is only available in C compilation target")
            exit(1)
        elif op.typ == OpType.CVAR:
            compiler_error(op.loc, "`cvar` instruction is only available in C compilation target")
            exit(1)
        else:
            raise Exception('Unreachable')
    fprintf(stream, "segment .bss")
    fprintf(stream, f"mem: resb {memory_size}")
    fprintf(stream, "segment .data")
    for i, s in enumerate(strs):
        stream.write(f"str_{i}: db ")
        for c in s:
            stream.write(f"{hex(ord(c))},")
        stream.write("0x00\n")

def generate_c_linux_x86_64(program: Program, memory_size: int, stream: IO):
    assert len(OpType) == 17, "Not all operation types were handled in generate_c_linux_x86_64()"
    fprintf(stream, "#if !(defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER))")
    fprintf(stream, "#  error \"This code is only compilable by GCC\"")
    fprintf(stream, "#endif")
    fprintf(stream, "")
    fprintf(stream, "#include <stdbool.h>")
    fprintf(stream, "#include <stdio.h>")
    fprintf(stream, "#include <stdint.h>")
    fprintf(stream, "#include <stdlib.h>")
    fprintf(stream, "#include <unistd.h>")
    fprintf(stream, "")
    fprintf(stream, "#define STACK_CAPACITY 640000")
    fprintf(stream, "static int64_t stack[STACK_CAPACITY] = {0};")
    fprintf(stream, "size_t stack_count = 0;")
    fprintf(stream, "")
    fprintf(stream, f"static char mem[{memory_size}];")
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
    condition_number = 0
    while_stack: List[int] = []
    for ip, op in enumerate(program):
        comment = str(op.typ)
        if op.typ == OpType.INTRINSIC:
            comment += f" {str(op.operand)}"
        fprintf(stream, f"    // -- {comment} --")
        if op.typ == OpType.PUSH_INT:
            fprintf(stream, f"    push({op.operand});")
        elif op.typ == OpType.PUSH_STR:
            assert isinstance(op.operand, str), "This could be a bug in the parser"
            fprintf(stream, f"    push({len(op.operand)});")
            stream.write("    push((int64_t) ")
            for c in op.operand:
                stream.write("\"\\x%x\"" % ord(c))
            stream.write(");\n")
        elif op.typ == OpType.PUSH_MEM:
            fprintf(stream, f"    push(mem+{op.operand});")
        elif op.typ == OpType.INTRINSIC:
            assert len(Intrinsic) == 25, "Not all intrinsics were handled in generate_c_linux_x86_64()"
            if op.operand == Intrinsic.PLUS:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        push(a + b);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.MINUS:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        push(b - a);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.EQUAL:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        push(a == b);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.NEQUAL:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        push(a != b);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.PRINT:
                fprintf(stream, "    printf(\"%ld\\n\", pop());")
            elif op.operand == Intrinsic.EXIT:
                fprintf(stream, "    exit(pop());")
            elif op.operand == Intrinsic.DUP:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        push(a);")
                fprintf(stream, "        push(a);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.DIVMOD:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        push(a / b);")
                fprintf(stream, "        push(a % b);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SWAP:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        push(a);")
                fprintf(stream, "        push(b);")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.DROP:
                fprintf(stream, "    pop();")
            elif op.operand == Intrinsic.SYSCALL0:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t sys = pop();")
                fprintf(stream, "        push(syscall(sys));")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL1:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t sys = pop();")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        push(syscall(sys, a));")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL2:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t sys = pop();")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        push(syscall(sys, a, b));")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL3:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t sys = pop();")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        int64_t c = pop();")
                fprintf(stream, "        push(syscall(sys, a, b, c));")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL4:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t sys = pop();")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        int64_t c = pop();")
                fprintf(stream, "        int64_t d = pop();")
                fprintf(stream, "        push(syscall(sys, a, b, c, d));")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL5:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t sys = pop();")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        int64_t c = pop();")
                fprintf(stream, "        int64_t d = pop();")
                fprintf(stream, "        int64_t e = pop();")
                fprintf(stream, "        push(syscall(sys, a, b, c, d, e));")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.SYSCALL6:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t sys = pop();")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        int64_t c = pop();")
                fprintf(stream, "        int64_t d = pop();")
                fprintf(stream, "        int64_t e = pop();")
                fprintf(stream, "        int64_t f = pop();")
                fprintf(stream, "        push(syscall(sys, a, b, c, d, e, f));")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.LOAD8:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        push(*((char*) a));")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.LOAD16:
                raise NotImplementedError
            elif op.operand == Intrinsic.LOAD32:
                raise NotImplementedError
            elif op.operand == Intrinsic.LOAD64:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        push(*((int64_t*) a));")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.STORE8:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        *((char*) b) = (char) a;")
                fprintf(stream, "    }")
            elif op.operand == Intrinsic.STORE16:
                raise NotImplementedError
            elif op.operand == Intrinsic.STORE32:
                raise NotImplementedError
            elif op.operand == Intrinsic.STORE64:
                fprintf(stream, "    {")
                fprintf(stream, "        int64_t a = pop();")
                fprintf(stream, "        int64_t b = pop();")
                fprintf(stream, "        *((int64_t*) b) = a;")
                fprintf(stream, "    }")
            else:
                raise Exception('Unreachable')
        elif op.typ == OpType.IF:
            fprintf(stream, "    if (pop()) {")
        elif op.typ == OpType.ELSE:
            fprintf(stream, "    } else {")
        elif op.typ == OpType.WHILE:
            while_stack.append(condition_number)
            fprintf(stream, "    bool condition_%d() {" % condition_number)
            condition_number += 1
        elif op.typ == OpType.DO:
            fprintf(stream, "        return (bool) pop();")
            fprintf(stream, "    }")
            assert len(while_stack) > 0
            fprintf(stream, "    while (condition_%d()) {" % while_stack.pop())
        elif op.typ == OpType.END:
            fprintf(stream, "    }")
        elif op.typ == OpType.CALL0:
            assert isinstance(op.operand, CallOperand), "This could be a bug in the parser"
            func_name = op.operand.func
            returns = op.operand.returns
            fprintf(stream, "    {")
            if returns:
                fprintf(stream, f"        push({func_name}());")
            else:
                fprintf(stream, f"        {func_name}();")
            fprintf(stream, "    }")
        elif op.typ == OpType.CALL1:
            assert isinstance(op.operand, CallOperand), "This could be a bug in the parser"
            func_name = op.operand.func
            returns = op.operand.returns
            fprintf(stream, "    {")
            fprintf(stream, "        int64_t a = pop();")
            if returns:
                fprintf(stream, f"        push({func_name}(a));")
            else:
                fprintf(stream, f"        {func_name}(a);")
            fprintf(stream, "    }")
        elif op.typ == OpType.CALL2:
            assert isinstance(op.operand, CallOperand), "This could be a bug in the parser"
            func_name = op.operand.func
            returns = op.operand.returns
            fprintf(stream, "    {")
            fprintf(stream, "        int64_t a = pop();")
            fprintf(stream, "        int64_t b = pop();")
            if returns:
                fprintf(stream, f"        push({func_name}(a, b));")
            else:
                fprintf(stream, f"        {func_name}(a, b);")
            fprintf(stream, "    }")
        elif op.typ == OpType.CALL3:
            assert isinstance(op.operand, CallOperand), "This could be a bug in the parser"
            func_name = op.operand.func
            returns = op.operand.returns
            fprintf(stream, "    {")
            fprintf(stream, "        int64_t a = pop();")
            fprintf(stream, "        int64_t b = pop();")
            fprintf(stream, "        int64_t c = pop();")
            if returns:
                fprintf(stream, f"        push({func_name}(a, b, c));")
            else:
                fprintf(stream, f"        {func_name}(a, b, c);")
            fprintf(stream, "    }")
        elif op.typ == OpType.CALL4:
            assert isinstance(op.operand, CallOperand), "This could be a bug in the parser"
            func_name = op.operand.func
            returns = op.operand.returns
            fprintf(stream, "    {")
            fprintf(stream, "        int64_t a = pop();")
            fprintf(stream, "        int64_t b = pop();")
            fprintf(stream, "        int64_t c = pop();")
            fprintf(stream, "        int64_t d = pop();")
            if returns:
                fprintf(stream, f"        push({func_name}(a, b, c, d));")
            else:
                fprintf(stream, f"        {func_name}(a, b, c, d);")
            fprintf(stream, "    }")
        elif op.typ == OpType.CALL5:
            assert isinstance(op.operand, CallOperand), "This could be a bug in the parser"
            func_name = op.operand.func
            returns = op.operand.returns
            fprintf(stream, "    {")
            fprintf(stream, "        int64_t a = pop();")
            fprintf(stream, "        int64_t b = pop();")
            fprintf(stream, "        int64_t c = pop();")
            fprintf(stream, "        int64_t d = pop();")
            fprintf(stream, "        int64_t e = pop();")
            if returns:
                fprintf(stream, f"        push({func_name}(a, b, c, d, e));")
            else:
                fprintf(stream, f"        {func_name}(a, b, c, d, e);")
            fprintf(stream, "    }")
        elif op.typ == OpType.CALL6:
            assert isinstance(op.operand, CallOperand), "This could be a bug in the parser"
            func_name = op.operand.func
            returns = op.operand.returns
            fprintf(stream, "    {")
            fprintf(stream, "        int64_t a = pop();")
            fprintf(stream, "        int64_t b = pop();")
            fprintf(stream, "        int64_t c = pop();")
            fprintf(stream, "        int64_t d = pop();")
            fprintf(stream, "        int64_t e = pop();")
            fprintf(stream, "        int64_t f = pop();")
            if returns:
                fprintf(stream, f"        push({func_name}(a, b, c, d, e, f));")
            else:
                fprintf(stream, f"        {func_name}(a, b, c, d, e, f);")
            fprintf(stream, "    }")
        elif op.typ == OpType.CVAR:
            fprintf(stream, f"    push({op.operand});")
        else:
            raise Exception('Unreachable')
    fprintf(stream, "}")

####### PARSER

HOME = getenv("HOME")
INCLUDE_SEARCH_PATHS: List[str] = ["./", "./std/", f"{HOME}/.local/include/north/", "/usr/local/include/north/"]

assert len(Intrinsic) == 25, "Not all intrinsics were handled in INTRINSICS_TABLE"
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
    '!8': Intrinsic.STORE8,
    '!16': Intrinsic.STORE16,
    '!32': Intrinsic.STORE32,
    '!64': Intrinsic.STORE64,
    '@8': Intrinsic.LOAD8,
    '@16': Intrinsic.LOAD16,
    '@32': Intrinsic.LOAD32,
    '@64': Intrinsic.LOAD64,
}

@dataclass
class Macro:
    name: str
    tokens: List[Token]
    loc: TokenLoc

def compile_time_evaluate(tokens: List[Token], macros: Dict[str, Macro]) -> int:
    stack: List[int] = []
    rtokens = list(reversed(tokens))
    while len(rtokens) > 0:
        token = rtokens.pop()
        assert len(TokenType) == 4, "Not all token types were handled in compile_time_evaluate()"
        if token.typ == TokenType.INT:
            assert isinstance(token.value, int), "This could be a bug in the lexer"
            stack.append(token.value)
        elif token.typ == TokenType.WORD:
            assert isinstance(token.value, str), "This could be a bug in the lexer"
            if token.value == '+':
                if len(stack) < 2:
                    compiler_error(token.loc, "not enough arguments for `+` intrinsic in compile time evaluation")
                    exit(1)
                a = stack.pop()
                b = stack.pop()
                stack.append(a + b)
            elif token.value == '-':
                if len(stack) < 2:
                    compiler_error(token.loc, "not enough arguments for `-` intrinsic in compile time evaluation")
                    exit(1)
                a = stack.pop()
                b = stack.pop()
                stack.append(b - a)
            elif token.value == 'divmod':
                if len(stack) < 2:
                    compiler_error(token.loc, "not enough arguments for `+` intrinsic in compile time evaluation")
                    exit(1)
                a = stack.pop()
                b = stack.pop()
                stack.append(int(a / b))
                stack.append(a % b)
            else:
                if token.value in macros:
                    macro = macros[token.value]
                    rtokens += list(reversed(macro.tokens))
                else:
                    compiler_error(token.loc, f"unsupported word in compile time evaluation: `{token.value}`")
                    exit(1)
        else:
            compiler_error(token.loc, "unsupported token type in compile time evaluation")
            exit(1)
    if len(stack) != 1:
        compiler_error(token.loc, "compile time evaluation should produce only one result")
        exit(1)
    return stack[0]

def parse_tokens_into_program(tokens: List[Token]) -> Tuple[int, Program]:
    rtokens = list(reversed(tokens))
    macros: Dict[str, Macro] = {}
    memories: Dict[str, int] = {}
    memories_offset: int = 0
    program: Program = []
    block_stack: List[Tuple[int, TokenLoc]] = []
    while len(rtokens) > 0:
        token = rtokens.pop()
        assert len(TokenType) == 4, "Not all token types were handled in parse_tokens_into_program()"
        if token.typ == TokenType.INT:
            assert isinstance(token.value, int), "This could be a bug in the lexer"
            program.append(Op(typ=OpType.PUSH_INT, operand=token.value, loc=token.loc))
        elif token.typ == TokenType.WORD:
            assert isinstance(token.value, str), "This could be a bug in the lexer"
            if token.value in INTRINSICS_TABLE:
                program.append(Op(typ=OpType.INTRINSIC, operand=INTRINSICS_TABLE[token.value], loc=token.loc))
            elif token.value in macros:
                macro = macros[token.value]
                rtokens += list(reversed(macro.tokens))
            elif token.value in memories:
                memory_offset = memories[token.value]
                program.append(Op(typ=OpType.PUSH_MEM, operand=memory_offset, loc=token.loc))
            else:
                compiler_error(token.loc, f"unknown word: `{token.value}`")
                exit(1)
        elif token.typ == TokenType.STR:
            assert isinstance(token.value, str), "This could be a bug in the lexer"
            program.append(Op(typ=OpType.PUSH_STR, operand=token.value, loc=token.loc))
        elif token.typ == TokenType.KEYWORD:
            assert len(Keyword) == 10, "Not all keyword types were handled in parse_tokens_into_program()"
            if token.value == Keyword.IF:
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.IF, loc=token.loc))
            elif token.value == Keyword.ELSE:
                if len(block_stack) < 1:
                    compiler_error(token.loc, "`else` is not preceeded by `if`")
                    exit(1)
                if_ip, _ = block_stack.pop()
                if program[if_ip].typ != OpType.IF:
                    compiler_error(token.loc, "`else` can only close `if` blocks")
                    exit(1)
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.ELSE, loc=token.loc))
                program[if_ip].operand = len(program)
            elif token.value == Keyword.WHILE:
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.WHILE, loc=token.loc))
            elif token.value == Keyword.DO:
                if len(block_stack) < 1:
                    compiler_error(token.loc, "`do` is not preceeded by `while`")
                    exit(1)
                while_ip, _ = block_stack.pop()
                if program[while_ip].typ != OpType.WHILE:
                    compiler_error(token.loc, "`do` can only close `while` blocks")
                    exit(1)
                block_stack.append((len(program), token.loc))
                program.append(Op(typ=OpType.DO, operand=while_ip, loc=token.loc))
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
                        assert len(Keyword) == 9, "Not all keyword types were handled while parsing macro body"
                        if ntoken.value in [Keyword.IF, Keyword.WHILE, Keyword.MACRO]:
                            nesting_depth += 1
                        elif ntoken.value == Keyword.END:
                            nesting_depth -= 1
                    macro_tokens.append(ntoken)
                if macro_name in macros:
                    compiler_error(macro_name_token.loc, "redefinition of already existing macro")
                    compiler_note(macros[macro_name].loc, "original definition located here")
                    exit(1)
                if macro_name in memories:
                    compiler_error(macro_name_token.loc, "redefinition of already existing memory")
                    exit(1)
                if macro_name in INTRINSICS_TABLE:
                    compiler_error(macro_name_token.loc, "redefinition of built-in intrinsic")
                    exit(1)
                macros[macro_name] = Macro(name=macro_name, tokens=macro_tokens, loc=macro_loc)
            elif token.value == Keyword.MEMORY:
                if len(rtokens) < 1:
                    compiler_error(token.loc, "unfinished memory definition")
                    exit(1)
                name_token = rtokens.pop()
                if name_token.typ != TokenType.WORD:
                    compiler_error(name_token.loc, "memory name must be a word")
                    exit(1)
                assert isinstance(name_token.value, str), "This could be a bug in the lexer"
                mem_name = name_token.value
                mem_tokens: List[Token] = []

                while True:
                    if len(rtokens) < 1:
                        compiler_error(token.loc, "unfinished memory definition")
                        exit(1)
                    ntoken = rtokens.pop()
                    if ntoken.typ == TokenType.KEYWORD and ntoken.value == Keyword.END:
                        break
                    mem_tokens.append(ntoken)

                mem_size = compile_time_evaluate(mem_tokens, macros)

                if mem_name in macros:
                    compiler_error(name_token.loc, "redefinition of already existing macro")
                    exit(1)
                if mem_name in memories:
                    compiler_error(name_token.loc, "redefinition of already existing memory")
                    exit(1)
                if mem_name in INTRINSICS_TABLE:
                    compiler_error(name_token.loc, "redefinition of built-in intrinsic")
                    exit(1)
                memories[mem_name] = memories_offset
                memories_offset += mem_size
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
                    program.append(Op(typ=OpType.END, loc=token.loc))
                    program[block_ip].operand = len(program)
                elif program[block_ip].typ == OpType.DO:
                    program.append(Op(typ=OpType.END, operand=program[block_ip].operand, loc=token.loc))
                    program[block_ip].operand = len(program)
                else:
                    compiler_error(token.loc, "`end` can only close `while-do`, `if` or `if-else` blocks")
                    exit(1)
            elif token.value == Keyword.CALL:
                if len(rtokens) < 1:
                    compiler_error(token.loc, "the arguments count of the function to be called was not provided")
                    exit(1)
                count_token = rtokens.pop()
                if count_token.typ != TokenType.INT:
                    compiler_error(count_token.loc, "the arguments count of the function to be called should be an integer")
                    exit(1)
                assert isinstance(count_token.value, int), "This could be a bug in the lexer"
                args_count = count_token.value

                if len(rtokens) < 1:
                    compiler_error(token.loc, "the name of the function to be called was not provided")
                    exit(1)
                name_token = rtokens.pop()
                if name_token.typ != TokenType.WORD:
                    compiler_error(name_token.loc, "the name of the function to be called should be a word")
                    exit(1)
                assert isinstance(name_token.value, str), "This could be a bug in the lexer"
                func_name = name_token.value

                if len(rtokens) < 1:
                    compiler_error(token.loc, "missing return information of the function to be called")
                    exit(1)
                returns_token = rtokens.pop()
                if returns_token.typ != TokenType.WORD:
                    compiler_error(returns_token.loc, "the return information of the function to be called should be a word")
                    exit(1)
                assert isinstance(returns_token.value, str), "This could be a bug in the lexer"
                returns_string = returns_token.value
                returns: bool
                if returns_string == "true": returns = True
                elif returns_string == "false": returns = False
                else:
                    compiler_error(returns_token.loc, "the return information of the function to be called should be 'true' or 'false'")
                    exit(1)

                ARGC_TO_CALL_INST: Dict[int, OpType] = {
                    0: OpType.CALL0,
                    1: OpType.CALL1,
                    2: OpType.CALL2,
                    3: OpType.CALL3,
                    4: OpType.CALL4,
                    5: OpType.CALL5,
                    6: OpType.CALL6,
                }

                program.append(Op(typ=ARGC_TO_CALL_INST[args_count],
                                  operand=CallOperand(func=func_name, returns=bool(returns)),
                                  loc=token.loc))
            elif token.value == Keyword.CVAR:
                if len(rtokens) < 1:
                    compiler_error(token.loc, "the variable name was not provided")
                    exit(1)
                name_token = rtokens.pop()
                if name_token.typ != TokenType.WORD:
                    compiler_error(name_token.loc, "the variable name must be a word")
                    exit(1)
                assert isinstance(name_token.value, str), "This could be a bug in the lexer"
                var_name = name_token.value

                program.append(Op(typ=OpType.CVAR, operand=var_name, loc=token.loc))
            else:
                raise Exception('unreachable')
        else:
            raise Exception('unreachable')
    if len(block_stack) != 0:
        _, loc = block_stack.pop()
        compiler_error(loc, "unclosed block")
        exit(1)
    program += [Op(typ=OpType.PUSH_INT, operand=0, loc=('',0,0)),
                Op(typ=OpType.INTRINSIC, operand=Intrinsic.EXIT, loc=('',0,0))]
    return memories_offset, program

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
        mem_size, program = parse_tokens_into_program(lex_file(filename))
        if target == "nasm":
            output = f"{basefilename}.asm"
            o_filename = f"{basefilename}.o"

            file = open(output, "w")
            compiler_info("info", f"Generating `{output}`")
            generate_nasm_linux_x86_64(program, mem_size, file)
            file.close()

            run_cmd_with_log(["nasm", "-felf64", output])
            run_cmd_with_log(["ld", "-o", basefilename, o_filename])
        elif target == "c":
            output = f"{basefilename}.c"

            file = open(output, "w")
            compiler_info("info", f"Generating `{output}`")
            generate_c_linux_x86_64(program, mem_size, file)
            file.close()

            disabled_warnings = ["-Wno-int-conversion"]
            run_cmd_with_log(["gcc", "-std=gnu17"] + disabled_warnings + ["-o", basefilename, output])
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
