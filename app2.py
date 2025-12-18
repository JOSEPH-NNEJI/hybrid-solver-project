import matplotlib
matplotlib.use('Agg')  # Prevents server errors with plots
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from solver import HybridSolver

# --- Page Config ---
st.set_page_config(page_title="Hybrid Root Finder", page_icon="🧮", layout="wide")

st.title("🧮 Hybrid Root-Finding Solver")
st.markdown("""
**Designed for Covenant University Project**
This tool demonstrates the **Adaptive Switching Logic** between the *Newton-Raphson Method* (Fast) and the *Bisection Method* (Reliable).
""")

# --- Sidebar Inputs ---
st.sidebar.header("1. Define Problem")
func_input = st.sidebar.text_input("Enter Function f(x):", value="x**3 - 2*x + 2")
st.sidebar.caption("Use Python syntax: e.g., `x**2` for x², `np.cos(x)` for cos(x)")

st.sidebar.header("2. Set Interval")
col1, col2 = st.sidebar.columns(2)
a_val = col1.number_input("Start (a)", value=-2.0)
b_val = col2.number_input("End (b)", value=1.0)

st.sidebar.header("3. Parameters")
tol = st.sidebar.number_input("Tolerance", value=1e-6, format="%.1e")

# --- Main Logic ---
if st.sidebar.button("Find Root"):
    solver = HybridSolver(func_input)
    
    if not solver.valid:
        st.error(solver.error_msg)
    else:
        # Solve
        root, history = solver.solve(a_val, b_val, tol)
        
        if isinstance(history, str): # Error returned as string
            st.error(history)
        else:
            # --- Result Display ---
            st.success(f"✅ Root Found: **{root:.8f}**")
            
            # --- Tabbed View for Details ---
            tab1, tab2 = st.tabs(["📈 Graphical Analysis", "📝 Step-by-Step Solution"])
            
            with tab1:
                st.subheader("Convergence Visualization")
                
                # Create Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # 1. Plot the Function Curve
                x_vals = np.linspace(a_val - 1, b_val + 1, 400)
                try:
                    y_vals = solver.f(x_vals)
                    ax.plot(x_vals, y_vals, label=f"f(x) = {func_input}", color='blue', alpha=0.6)
                    ax.axhline(0, color='black', linewidth=1) # x-axis
                except:
                    st.warning("Could not plot full curve due to domain errors.")

                # 2. Plot the Iteration Steps
                if not history.empty:
                    # Filter points by method for color coding
                    newton_steps = history[history["Method"] == "Newton-Raphson"]
                    bisection_steps = history[history["Method"] == "Bisection"]
                    
                    ax.scatter(newton_steps["Current Root Estimate"], 
                               [0]*len(newton_steps), 
                               color='green', s=50, label='Newton Step', zorder=3)
                    
                    ax.scatter(bisection_steps["Current Root Estimate"], 
                               [0]*len(bisection_steps), 
                               color='red', marker='x', s=50, label='Bisection Step', zorder=3)
                    
                    # Highlight Final Root
                    ax.scatter(root, 0, color='gold', s=150, edgecolors='black', label='Final Root', zorder=4)

                ax.legend()
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
                
                st.info("🟢 Green Dots = Fast Newton Steps | 🔴 Red Crosses = Safe Bisection Steps")

            with tab2:
                st.subheader("Algorithmic Audit Trail")
                st.markdown("This table shows exactly when and why the algorithm switched methods.")
                
                # Formatting the dataframe for better readability
                st.dataframe(history.style.format({
                    "Current Root Estimate": "{:.6f}",
                    "Function Value f(x)": "{:.2e}",
                    "Error Estimate": "{:.2e}"
                }).applymap(lambda v: 'color: red;' if 'Switched' in str(v) else None, subset=['Decision Logic']))
