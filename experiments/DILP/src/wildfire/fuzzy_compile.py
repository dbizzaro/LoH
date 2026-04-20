"""Compile and evaluate logical formulas with dILP-aligned soft semantics.

Semantics used for candidate-rule soft valuations:
- NOT: 1 - x
- AND: product (x * y)
- OR: probabilistic sum (x + y - x*y), folded left-to-right for n-ary OR
"""

import re
import tensorflow as tf

_TOKEN_PATTERN = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*|[~&|()])\s*")


class FormulaCompiler:
    def __init__(self, formula):
        self.formula = formula
        self.rpn = self._to_rpn(self._tokenize(formula))

    @staticmethod
    def _tokenize(formula):
        tokens = []
        i = 0
        while i < len(formula):
            m = _TOKEN_PATTERN.match(formula, i)
            if not m:
                raise ValueError('Invalid token near: %s' % formula[i:i + 20])
            tokens.append(m.group(1))
            i = m.end()
        return tokens

    @staticmethod
    def _precedence(op):
        if op == '~':
            return 3
        if op == '&':
            return 2
        if op == '|':
            return 1
        return 0

    @staticmethod
    def _to_rpn(tokens):
        out = []
        ops = []
        for tok in tokens:
            if tok in ('~', '&', '|'):
                if tok == '~':
                    # unary, right-associative
                    while ops and ops[-1] != '(' and FormulaCompiler._precedence(ops[-1]) > FormulaCompiler._precedence(tok):
                        out.append(ops.pop())
                else:
                    while ops and ops[-1] != '(' and FormulaCompiler._precedence(ops[-1]) >= FormulaCompiler._precedence(tok):
                        out.append(ops.pop())
                ops.append(tok)
            elif tok == '(':
                ops.append(tok)
            elif tok == ')':
                while ops and ops[-1] != '(':
                    out.append(ops.pop())
                if not ops or ops[-1] != '(':
                    raise ValueError('Mismatched parentheses in formula')
                ops.pop()
            else:
                out.append(tok)
        while ops:
            op = ops.pop()
            if op in ('(', ')'):
                raise ValueError('Mismatched parentheses in formula')
            out.append(op)
        return out

    @staticmethod
    def probabilistic_or(a, b):
        c = a + b - a * b
        return tf.clip_by_value(c, 0.0, 1.0)

    def evaluate(self, env):
        stack = []
        for tok in self.rpn:
            if tok == '~':
                if not stack:
                    raise ValueError('Malformed formula: missing operand for NOT')
                a = stack.pop()
                stack.append(1.0 - a)
            elif tok == '&':
                if len(stack) < 2:
                    raise ValueError('Malformed formula: missing operands for AND')
                b = stack.pop()
                a = stack.pop()
                stack.append(tf.clip_by_value(a * b, 0.0, 1.0))
            elif tok == '|':
                if len(stack) < 2:
                    raise ValueError('Malformed formula: missing operands for OR')
                b = stack.pop()
                a = stack.pop()
                stack.append(self.probabilistic_or(a, b))
            else:
                if tok not in env:
                    raise KeyError('Variable %s not in environment' % tok)
                stack.append(tf.cast(env[tok], tf.float32))
        if len(stack) != 1:
            raise ValueError('Malformed formula: unresolved stack')
        return stack[0]
