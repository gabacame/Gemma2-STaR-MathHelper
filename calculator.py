import sympy as sp
import numpy as np

class Calculator:
    def __init__(self):
        pass

    def evaluate_expression(self, expression):
        try:
            result = sp.sympify(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def integrate_expression(self, expression, variable):
        try:
            var = sp.symbols(variable)
            result = sp.integrate(expression, var)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def differentiate_expression(self, expression, variable):
        try:
            var = sp.symbols(variable)
            result = sp.diff(expression, var)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def solve_equation(self, equation, variable):
        try:
            var = sp.symbols(variable)
            result = sp.solve(equation, var)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def matrix_operations(self, operation, *matrices):
        try:
            if operation == 'add':
                result = np.add(*matrices)
            elif operation == 'subtract':
                result = np.subtract(*matrices)
            elif operation == 'multiply':
                result = np.matmul(*matrices)
            else:
                return "Error: Operación no soportada"
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
