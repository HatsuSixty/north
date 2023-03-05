;;; north-mode.el --- Major Mode for editing North source code -*- lexical-binding: t -*-

;; Copyright (c) 2023 Roberto Hermenegildo Dias

;; Author: Roberto Hermenegildo Dias
;; URL: https://github.com/robertosixty1/north

;; Permission is hereby granted, free of charge, to any person obtaining a copy
;; of this software and associated documentation files (the "Software"), to deal
;; in the Software without restriction, including without limitation the rights
;; to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
;; copies of the Software, and to permit persons to whom the Software is
;; furnished to do so, subject to the following conditions:

;; The above copyright notice and this permission notice shall be included in all
;; copies or substantial portions of the Software.

;; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
;; IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
;; FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
;; AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
;; LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
;; OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
;; SOFTWARE.

;;; Commentary:
;;
;; Major Mode for editing North source code.

(require 'rx)

(defconst north-mode-syntax-table
  (with-syntax-table (copy-syntax-table)
    ;; Comments
    (modify-syntax-entry ?\; "<")
    (modify-syntax-entry ?\n ">")
    (syntax-table))
  "Syntax table for `north-mode'.")

;; Intrinsics
(defconst north-intrinsics
  '("print" "+" "-" "<<" ">>" "|" "&" "=" "!=" "divmod" "exit" "dup" "swap" "drop" "over" "syscall0" "syscall1" "syscall2" "syscall3" "syscall4" "syscall5" "syscall6" "!8" "!16" "!32" "!64" "@8" "@16" "@32" "@64" "<"))

;; Keywords are instructions that do something more than just generating ir
(defconst north-keywords
  '("if" "else" "while" "do" "macro" "memory" "end" "call" "cvar" "include"))

(defun north-wrap-word-rx (s)
  (concat "\\<" s "\\>"))

(defconst north-number-rx
  (rx (and
       symbol-start
       (or (and (+ digit) (opt (and (any "eE") (opt (any "-+")) (+ digit))))
           (and "0" (any "xX") (+ hex-digit)))
       (opt (and (any "_" "A-Z" "a-z") (* (any "_" "A-Z" "a-z" "0-9"))))
       symbol-end)))

(defconst north-highlights
  `(
    ;; Keywords
    (,(regexp-opt north-keywords 'words) . 'font-lock-keyword-face)

    ;; Numbers
    (,(north-wrap-word-rx north-number-rx) . 'font-lock-constant-face)

    ;; Intrinsics
    (,(regexp-opt north-intrinsics 'words) 1 'font-lock-builtin-face)
    ))

;;;###autoload
(define-derived-mode north-mode prog-mode "North"
  "Major Mode for editing North source code"
  :syntax-table north-mode-syntax-table
  (setq font-lock-defaults '(north-highlights))
  (setq-local comment-start "; "))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.north\\'" . north-mode))

(provide 'north-mode)

;; north-mode.el ends here
