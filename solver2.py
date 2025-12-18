import sympy as sp
import numpy as np
import pandas as pd

class HybridSolver:
    def __init__(self, func_str):
        """
        Initialize the solver by parsing the mathematical function.
        Uses SymPy to automatically find the derivative.
        """
        self.x = sp.symbols('x')
        try:
            # Convert string input (e.g., "x^2 - 2") into a math expression
            self.expr = sp.sympify(func_str)
            self.deriv = sp.diff(self.expr, self.x)
            
            # Create fast executable functions for f(x) and f'(x)
            self.f = sp.lambdify(self.x, self.expr, 'numpy')
            self.df = sp.lambdify(self.x, self.deriv, 'numpy')
            
            self.valid = True
            self.error_msg = ""
        except Exception as e:
            self.valid = False
            self.error_msg = f"Invalid Function Syntax: {e}"
            
        # To store the step-by-step history
        self.history = []

    def solve(self, a, b, tol=1e-6, max_iter=100):
        """
        The Main Hybrid Loop.
        Returns: root (float), history_df (DataFrame)
        """
        if not self.valid:
            return None, pd.DataFrame()

        # Check for valid bracket (IVT)
        fa = self.f(a)
        fb = self.f(b)
        
        if fa * fb >= 0:
            return None, "Error: The chosen interval [a, b] does not bracket a root. f(a) and f(b) must have opposite signs."

        current_x = (a + b) / 2
        self.history = [] # Clear previous history

        for i in range(1, max_iter + 1):
            method_used = ""
            decision_note = ""
            
            # --- 1. Calculate Newton Step ---
            try:
                f_val = self.f(current_x)
                df_val = self.df(current_x)
                
                if abs(df_val) < 1e-12: # Check for Zero Division
                    raise ValueError("Zero Derivative")
                
                x_newton = current_x - (f_val / df_val)
                
                # --- 2. Safety Check (The Switching Logic) ---
                # Is the Newton step inside the current bracket?
                if a < x_newton < b:
                    current_x = x_newton
                    method_used = "Newton-Raphson"
                    decision_note = "Accepted (Step is Safe)"
                else:
                    # Switch to Bisection if Newton shoots outside
                    current_x = (a + b) / 2
                    method_used = "Bisection"
                    decision_note = "Switched (Newton Overshot Bounds)"
            
            except Exception:
                # Fallback to Bisection if Newton crashes (e.g. division by zero)
                current_x = (a + b) / 2
                method_used = "Bisection"
                decision_note = "Switched (Newton Singularity)"

            # --- 3. Update Brackets (Standard Bisection Logic) ---
            f_new = self.f(current_x)
            
            if self.f(a) * f_new < 0:
                b = current_x
            else:
                a = current_x
                
            # Calculate Error (Interval Width)
            error = abs(b - a)
            
            # Log the Step
            self.history.append({
                "Iteration": i,
                "Method": method_used,
                "Current Root Estimate": float(current_x),
                "Function Value f(x)": float(f_new),
                "Error Estimate": float(error),
                "Decision Logic": decision_note
            })

            # Check for Convergence
            if error < tol or abs(f_new) < 1e-12:
                break
        
        # Convert history to a clean table
        return current_x, pd.DataFrame(self.history)