import streamlit as st
import numpy as np
import pandas as pd
import time

# Set page configuration
st.set_page_config(
    page_title="FreshDrinks Co. - Production Optimization",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .problem-section {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
    }
    .step-box {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #43A047;
    }
    .tableau-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #FF9800;
        overflow-x: auto;
    }
    .solution-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    .insight-box {
        background-color: #F3E5F5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #9C27B0;
    }
    .stButton button {
        width: 100%;
        font-size: 1.2rem;
        height: 3rem;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        color: white;
        border: none;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #1565C0, #2E7D32);
    }
    .tableau-table {
        font-family: monospace;
        font-size: 0.8rem;
        border-collapse: collapse;
        width: 100%;
    }
    .tableau-table th, .tableau-table td {
        border: 1px solid #ddd;
        padding: 4px;
        text-align: right;
        min-width: 50px;
    }
    .tableau-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    .basic-var {
        background-color: #E8F5E9;
        font-weight: bold;
    }
    .pivot-element {
        background-color: #FFF3E0;
        font-weight: bold;
        border: 2px solid #FF9800 !important;
    }
    .iteration-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2196F3;
    }
    </style>
""", unsafe_allow_html=True)

class FreshDrinksSimplexSolver:
    def __init__(self):
        self.tableau = None
        self.basic_vars = []
        self.var_names = []
        self.M = 10000  # Big M value
        self.optimal_value = 0
        self.solution = {}
        
        # Problem data
        self.products = [
            'Orange Juice (x1)', 'Apple Juice (x2)', 'Mango Juice (x3)', 
            'Lemon Drink (x4)', 'Energy Drink (x5)', 'Sports Drink (x6)',
            'Vitamin Water (x7)', 'Sparkling Water (x8)', 'Iced Tea (x9)', 
            'Cold Coffee (x10)'
        ]
        
        self.product_codes = ['OJ', 'AJ', 'MJ', 'LD', 'ED', 'SD', 'VW', 'SPW', 'IT', 'CC']
        
        # Initialize num_products here
        self.num_products = len(self.products)
        
        self.profit_margins = [20, 18, 25, 15, 30, 22, 17, 10, 12, 16]
        
        # Resource consumption matrix (10 products x 6 resources)
        self.resource_consumption = np.array([
            # Fruit, Sugar, Bottles, Mixing, Labeling, Labor
            [0.5, 0.3, 1, 0.6, 0.3, 0.4],  # OJ
            [0.4, 0.2, 1, 0.5, 0.2, 0.3],  # AJ
            [0.6, 0.4, 1, 0.7, 0.4, 0.5],  # MJ
            [0.3, 0.3, 1, 0.4, 0.3, 0.2],  # LD
            [0.7, 0.5, 1, 0.9, 0.6, 0.6],  # ED
            [0.5, 0.4, 1, 0.6, 0.4, 0.4],  # SD
            [0.4, 0.3, 1, 0.5, 0.2, 0.3],  # VW
            [0.2, 0.1, 1, 0.3, 0.1, 0.2],  # SPW
            [0.3, 0.2, 1, 0.4, 0.1, 0.3],  # IT
            [0.4, 0.3, 1, 0.5, 0.2, 0.3]   # CC
        ])
        
        # Resource limits
        self.resource_limits = [500, 350, 800, 600, 400, 500]
        self.resource_names = ["Fruit Concentrate", "Sugar Syrup", "Bottles", 
                              "Mixing Hours", "Labeling Hours", "Labor Hours"]
        
        # Minimum production requirements
        self.min_production = [40, 30, 20, 25, 15, 10, 12, 8, 10, 12]
        
        # Storage capacity
        self.storage_capacity = 900
        
        # Current production
        self.current_production = [60, 50, 45, 40, 30, 35, 25, 20, 25, 30]
        
        # Calculate current profit
        self.current_profit = sum(self.current_production[i] * self.profit_margins[i]
                                 for i in range(self.num_products))
        
        # Initialize num_constraints
        self.num_constraints = 0
        
        # For tracking iterations
        self.iterations_history = []
        self.pivot_history = []
    
    def check_feasibility(self):
        """Check if the problem is feasible with given constraints"""
        # Check minimum production storage requirement
        min_storage_needed = sum(self.min_production)
        
        if min_storage_needed > self.storage_capacity:
            return False, f"Minimum production requires {min_storage_needed} boxes, but storage capacity is only {self.storage_capacity}"
        
        # Check resource requirements for minimum production
        for i in range(6):
            min_resource_needed = sum(self.min_production[j] * self.resource_consumption[j, i]
                                     for j in range(self.num_products))
            if min_resource_needed > self.resource_limits[i]:
                return False, f"{self.resource_names[i]} requires {min_resource_needed:.1f}, but only {self.resource_limits[i]} available"
        
        return True, "Problem is feasible"
    
    def build_problem(self):
        """Build the FreshDrinks Co. LP problem"""
        # Number of constraints: 6 resource + 1 storage + 10 minimum production
        self.num_constraints = 6 + 1 + self.num_products
        
        # Decision variable names
        self.var_names = [f'x{i+1}' for i in range(self.num_products)]
        
        # Count additional variables needed
        num_slack = 7  # 6 resource + 1 storage constraints (all <=)
        num_surplus = self.num_products   # minimum production constraints (all >=)
        num_artificial = self.num_products  # artificial variables for >= constraints
        
        # Add slack variables
        for i in range(num_slack):
            self.var_names.append(f'S{i+1}')
        
        # Add surplus variables
        for i in range(num_surplus):
            self.var_names.append(f's{i+1}')
        
        # Add artificial variables
        for i in range(num_artificial):
            self.var_names.append(f'A{i+1}')
        
        self.var_names.append('RHS')
        
        # Initialize tableau
        total_vars = self.num_products + num_slack + num_surplus + num_artificial
        self.tableau = np.zeros((self.num_constraints + 1, total_vars + 1))
        
        # Reset basic variables
        self.basic_vars = []
        
        # ===== CONSTRAINT 1-6: RESOURCE CONSTRAINTS (<=) =====
        for i in range(6):  # 6 resources
            # Add coefficients for products
            for j in range(self.num_products):
                self.tableau[i, j] = self.resource_consumption[j, i]
            # Add slack variable (coefficient = 1)
            slack_idx = self.num_products + i
            self.tableau[i, slack_idx] = 1
            # RHS = resource limit
            self.tableau[i, -1] = self.resource_limits[i]
            self.basic_vars.append(f'S{i+1}')
        
        # ===== CONSTRAINT 7: STORAGE CONSTRAINT (<=) =====
        storage_row = 6
        # Each product uses 1 box of storage
        for j in range(self.num_products):
            self.tableau[storage_row, j] = 1
        # Add slack variable
        slack_idx = self.num_products + 6
        self.tableau[storage_row, slack_idx] = 1
        self.tableau[storage_row, -1] = self.storage_capacity
        self.basic_vars.append('S7')
        
        # ===== CONSTRAINT 8-17: MINIMUM PRODUCTION (>=) =====
        for i in range(self.num_products):
            row_idx = 7 + i
            # Coefficient for the product = 1
            self.tableau[row_idx, i] = 1
            # Add surplus variable (coefficient = -1)
            surplus_idx = self.num_products + 7 + i
            self.tableau[row_idx, surplus_idx] = -1
            # Add artificial variable (coefficient = 1)
            artificial_idx = self.num_products + 7 + self.num_products + i
            self.tableau[row_idx, artificial_idx] = 1
            # RHS = minimum production
            self.tableau[row_idx, -1] = self.min_production[i]
            self.basic_vars.append(f'A{i+1}')
        
        # ===== OBJECTIVE FUNCTION =====
        # We're maximizing profit: Z = 20x1 + 18x2 + ... + 16x10
        # In tableau (maximizing -Z): -20, -18, ..., -16
        for i in range(self.num_products):
            self.tableau[-1, i] = -self.profit_margins[i]
        
        # Add Big M coefficients for artificial variables
        artificial_start_idx = self.num_products + 7 + self.num_products
        for i in range(self.num_products):
            self.tableau[-1, artificial_start_idx + i] = self.M
        
        # Eliminate artificial variables from objective row
        for i in range(self.num_products):
            row_idx = 7 + i  # Row for artificial variable Ai
            if abs(self.tableau[row_idx, -1]) > 1e-6:  # Only if RHS is non-zero
                self.tableau[-1] -= self.M * self.tableau[row_idx]
    
    def find_pivot_column(self):
        """Find entering variable (most negative coefficient in objective row)"""
        if self.tableau is None or len(self.tableau[-1]) == 0:
            return -1
        
        obj_row = self.tableau[-1, :-1]
        
        # Skip artificial variables in phase 1
        artificial_start = self.num_products + 7 + self.num_products
        for i in range(artificial_start, len(obj_row)):
            if abs(obj_row[i]) > self.M/2:  # If it's an artificial variable coefficient
                obj_row[i] = float('inf')  # Don't choose artificial variables
        
        min_val = np.min(obj_row)
        
        if min_val >= -1e-10:  # Tolerance for numerical errors
            return -1
        
        return np.argmin(obj_row)
    
    def find_pivot_row(self, pivot_col):
        """Find leaving variable using minimum ratio test"""
        if pivot_col < 0 or pivot_col >= self.tableau.shape[1]:
            return -1
        
        ratios = []
        for i in range(self.num_constraints):
            if self.tableau[i, pivot_col] > 1e-10:  # Positive
                ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                if ratio >= 0:  # Only consider non-negative ratios
                    ratios.append(ratio)
                else:
                    ratios.append(float('inf'))
            else:
                ratios.append(float('inf'))
        
        if all(r == float('inf') for r in ratios):
            return -1
        
        # Find minimum positive ratio
        min_ratio = float('inf')
        min_idx = -1
        for i, ratio in enumerate(ratios):
            if 0 <= ratio < min_ratio:
                min_ratio = ratio
                min_idx = i
        
        return min_idx
    
    def pivot(self, pivot_row, pivot_col):
        """Perform pivot operation"""
        if pivot_row < 0 or pivot_col < 0:
            return
        
        pivot_element = self.tableau[pivot_row, pivot_col]
        
        if abs(pivot_element) < 1e-10:
            return
        
        # Normalize pivot row
        self.tableau[pivot_row] = self.tableau[pivot_row] / pivot_element
        
        # Update other rows
        for i in range(len(self.tableau)):
            if i != pivot_row:
                multiplier = self.tableau[i, pivot_col]
                if abs(multiplier) > 1e-10:
                    self.tableau[i] = self.tableau[i] - multiplier * self.tableau[pivot_row]
        
        # Update basic variable
        if pivot_row < len(self.basic_vars):
            self.basic_vars[pivot_row] = self.var_names[pivot_col]
    
    def get_tableau_html(self, iteration, pivot_row=-1, pivot_col=-1):
        """Generate HTML table for the tableau"""
        html = f'<div style="margin-bottom: 20px;"><h4>Iteration {iteration}</h4>'
        html += f'<table class="tableau-table">'
        
        # Header row
        html += '<tr><th>BV</th>'
        for var in self.var_names[:-1]:  # All variables except RHS
            html += f'<th>{var}</th>'
        html += '<th>RHS</th></tr>'
        
        # Constraint rows
        for i in range(self.num_constraints):
            html += '<tr>'
            # Basic variable
            if i < len(self.basic_vars):
                html += f'<td class="basic-var">{self.basic_vars[i]}</td>'
            else:
                html += '<td></td>'
            
            # Coefficients
            for j in range(len(self.var_names) - 1):
                cell_class = ""
                if i == pivot_row and j == pivot_col:
                    cell_class = "pivot-element"
                value = self.tableau[i, j]
                html += f'<td class="{cell_class}">{value:7.2f}</td>'
            
            # RHS
            html += f'<td>{self.tableau[i, -1]:7.2f}</td></tr>'
        
        # Objective row
        html += '<tr><td>Z</td>'
        for j in range(len(self.var_names) - 1):
            value = self.tableau[-1, j]
            html += f'<td>{value:7.2f}</td>'
        html += f'<td>{self.tableau[-1, -1]:7.2f}</td></tr>'
        
        html += '</table>'
        
        # Add summary
        obj_val = -self.tableau[-1, -1]
        html += f'<p><strong>Current Z = {obj_val:.2f}</strong></p>'
        
        # Check optimality
        if len(self.tableau[-1, :-1]) > 0:
            min_coeff = min(self.tableau[-1, :-1])
            if min_coeff >= -1e-6:
                html += '<p style="color: green;">✓ Optimality reached (all reduced costs ≥ 0)</p>'
            else:
                html += f'<p>Not optimal yet. Most negative reduced cost: {min_coeff:.2f}</p>'
        
        html += '</div>'
        return html
    
    def solve(self):
        """Main solving method"""
        self.iterations_history = []
        self.pivot_history = []
        
        # First check if problem is feasible
        feasible, message = self.check_feasibility()
        if not feasible:
            return False, message
        
        # Build initial tableau
        self.build_problem()
        
        # Record initial tableau
        self.iterations_history.append(self.get_tableau_html(0))
        
        iteration = 0
        max_iterations = 50
        
        for iteration in range(1, max_iterations + 1):
            # Find entering variable
            pivot_col = self.find_pivot_column()
            
            if pivot_col == -1:
                # Check if any artificial variables are still in basis with non-zero value
                artificial_in_basis = False
                for i, basic_var in enumerate(self.basic_vars):
                    if basic_var.startswith('A') and i < len(self.tableau) and abs(self.tableau[i, -1]) > 1e-6:
                        artificial_in_basis = True
                        break
                
                if artificial_in_basis:
                    return False, "No feasible solution found! Artificial variables remain in basis."
                
                break
            
            # Find leaving variable
            pivot_row = self.find_pivot_row(pivot_col)
            
            if pivot_row == -1:
                return False, "The problem is unbounded!"
            
            # Record pivot information
            self.pivot_history.append({
                'iteration': iteration,
                'entering': self.var_names[pivot_col],
                'leaving': self.basic_vars[pivot_row],
                'pivot_element': self.tableau[pivot_row, pivot_col]
            })
            
            # Perform pivot
            self.pivot(pivot_row, pivot_col)
            
            # Record tableau after pivot
            self.iterations_history.append(self.get_tableau_html(iteration, pivot_row, pivot_col))
            
            # Early stopping if we've done many iterations
            if iteration >= max_iterations:
                return False, f"Maximum iterations ({max_iterations}) reached."
        
        # Extract solution
        self.extract_solution()
        return True, "Solution found successfully!"
    
    def extract_solution(self):
        """Extract the optimal solution"""
        self.solution = {}
        
        total_optimal = 0
        
        for i in range(self.num_products):
            product_code = self.product_codes[i]
            product_name = self.products[i]
            min_req = self.min_production[i]
            current = self.current_production[i]
            
            # Find optimal value
            if f'x{i+1}' in self.basic_vars:
                idx = self.basic_vars.index(f'x{i+1}')
                if idx < len(self.tableau):
                    optimal = self.tableau[idx, -1]
                else:
                    optimal = min_req
            else:
                # Non-basic variable, value is 0 (but must meet minimum)
                optimal = min_req
            
            # Ensure optimal meets minimum requirement
            optimal = max(optimal, min_req)
            
            self.solution[product_code] = {
                'name': product_name,
                'optimal': optimal,
                'minimum': min_req,
                'current': current,
                'profit_per_unit': self.profit_margins[i],
                'total_profit': optimal * self.profit_margins[i]
            }
            total_optimal += optimal
        
        # Calculate total profit
        self.optimal_value = sum(data['total_profit'] for data in self.solution.values())
    
    def get_resource_utilization(self):
        """Calculate resource utilization for the optimal solution"""
        utilizations = []
        
        for i in range(6):  # 6 resources
            used = 0
            for j in range(self.num_products):
                product_code = self.product_codes[j]
                if product_code in self.solution:
                    used += self.solution[product_code]['optimal'] * self.resource_consumption[j, i]
            
            available = self.resource_limits[i]
            utilization = (used / available) * 100 if available > 0 else 0
            
            utilizations.append({
                'resource': self.resource_names[i],
                'used': used,
                'available': available,
                'utilization': utilization,
                'status': "Bottleneck" if utilization > 95 else "Underutilized" if utilization < 80 else "Well-utilized"
            })
        
        # Storage utilization
        total_production = sum(data['optimal'] for data in self.solution.values())
        storage_utilization = (total_production / self.storage_capacity) * 100
        
        utilizations.append({
            'resource': "Storage",
            'used': total_production,
            'available': self.storage_capacity,
            'utilization': storage_utilization,
            'status': "Bottleneck" if storage_utilization > 95 else "OK"
        })
        
        return utilizations

def main():
    # Header
    st.markdown('<div class="main-header">🥤 FreshDrinks Co. - Production Optimization Simplex Solver</div>', unsafe_allow_html=True)
    
    # Problem Statement Section
    st.markdown('<div class="problem-section">', unsafe_allow_html=True)
    st.subheader("📋 Linear Programming Problem Formulation")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 **Objective Function (Maximize Profit):**
        $$Z = 20x_1 + 18x_2 + 25x_3 + 15x_4 + 30x_5 + 22x_6 + 17x_7 + 10x_8 + 12x_9 + 16x_{10}$$
        
        ### 📊 **Decision Variables:**
        - $x_1$ to $x_{10}$ = Weekly production of each beverage (boxes)
        
        ### ⚡ **Constraints:**
        **1. Resource Constraints (≤):**
        - Fruit Concentrate: $0.5x_1 + 0.4x_2 + ... + 0.4x_{10} ≤ 500$
        - Sugar Syrup: $0.3x_1 + 0.2x_2 + ... + 0.3x_{10} ≤ 350$
        - Bottles: $x_1 + x_2 + ... + x_{10} ≤ 800$
        - Mixing Hours: $0.6x_1 + 0.5x_2 + ... + 0.5x_{10} ≤ 600$
        - Labeling Hours: $0.3x_1 + 0.2x_2 + ... + 0.2x_{10} ≤ 400$
        - Labor Hours: $0.4x_1 + 0.3x_2 + ... + 0.3x_{10} ≤ 500$
        
        **2. Storage Constraint (≤):**
        - Storage: $x_1 + x_2 + ... + x_{10} ≤ 900$
        
        **3. Minimum Production Requirements (≥):**
        - $x_1 ≥ 40$, $x_2 ≥ 30$, $x_3 ≥ 20$, $x_4 ≥ 25$, $x_5 ≥ 15$
        - $x_6 ≥ 10$, $x_7 ≥ 12$, $x_8 ≥ 8$, $x_9 ≥ 10$, $x_{10} ≥ 12$
        
        **4. Non-negativity:**
        - $x_i ≥ 0$ for all $i = 1,...,10$
        """)
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 4rem;'>📊</div>
            <div style='margin-top: 1rem;'>
                <div style='font-size: 1.2rem; font-weight: bold; color: #1E88E5;'>10 Products</div>
                <div style='font-size: 1.2rem; font-weight: bold; color: #43A047;'>17 Constraints</div>
                <div style='font-size: 1.2rem; font-weight: bold; color: #FF9800;'>27 Variables</div>
                <div style='font-size: 1.2rem; font-weight: bold; color: #9C27B0;'>Big M Method</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create solver instance
    solver = FreshDrinksSimplexSolver()
    
    # Add a separator
    st.markdown("---")
    
    # Solution Button
    st.markdown("<div style='text-align: center; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        solve_button = st.button(
            "🚀 **SOLVE USING SIMPLEX METHOD**", 
            type="primary", 
            help="Click to see step-by-step simplex solution with tableau displays",
            key="solve_button_main",
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Solution Section
    if solve_button:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Feasibility Check
        status_text.text("Step 1: Checking feasibility...")
        progress_bar.progress(10)
        
        feasible, message = solver.check_feasibility()
        if not feasible:
            st.error(f"❌ Problem Infeasible: {message}")
            st.stop()
        
        st.success("✅ Problem is feasible")
        
        # Step 2: Build Tableau
        status_text.text("Step 2: Building initial tableau...")
        progress_bar.progress(30)
        
        solver.build_problem()
        
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("📊 Step 1: Initial Tableau Setup")
        st.markdown("""
        **Variables Added:**
        - **Slack Variables (S₁-S₇):** For ≤ constraints
        - **Surplus Variables (s₁-s₁₀):** For ≥ constraints  
        - **Artificial Variables (A₁-A₁₀):** For Big M method
        
        **Total Variables:** 10 (original) + 7 (slack) + 10 (surplus) + 10 (artificial) = 37 variables
        
        **Big M Value:** M = 10,000
        """)
        
        # Show initial basic variables
        st.markdown("**Initial Basic Variables:**")
        cols = st.columns(4)
        for i, basic_var in enumerate(solver.basic_vars[:8]):
            with cols[i % 4]:
                st.text(f"{basic_var}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 3: Simplex Iterations
        status_text.text("Step 3: Performing simplex iterations...")
        progress_bar.progress(50)
        
        # Create tabs for iterations
        st.markdown('<div class="tableau-box">', unsafe_allow_html=True)
        st.subheader("🔄 Step 2: Simplex Iterations")
        
        success, message = solver.solve()
        
        if not success:
            st.error(f"❌ {message}")
            st.stop()
        
        # Display iteration count
        st.success(f"✅ Solution found in {len(solver.iterations_history)-1} iterations")
        
        # Create tabs for different iteration groups
        iteration_tabs = st.tabs([f"Iteration {i}" for i in range(min(6, len(solver.iterations_history)))])
        
        for i, tab in enumerate(iteration_tabs):
            with tab:
                if i < len(solver.iterations_history):
                    st.markdown(solver.iterations_history[i], unsafe_allow_html=True)
                    
                    # Show pivot information if available
                    if i > 0 and i-1 < len(solver.pivot_history):
                        pivot_info = solver.pivot_history[i-1]
                        st.markdown(f"""
                        **Pivot Information:**
                        - **Iteration:** {pivot_info['iteration']}
                        - **Entering Variable:** {pivot_info['entering']}
                        - **Leaving Variable:** {pivot_info['leaving']}
                        - **Pivot Element:** {pivot_info['pivot_element']:.4f}
                        """)
        
        # If there are more iterations, show them in expanders
        if len(solver.iterations_history) > 6:
            with st.expander(f"Show iterations 6-{len(solver.iterations_history)-1}"):
                for i in range(6, len(solver.iterations_history)):
                    st.markdown(solver.iterations_history[i], unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 4: Optimal Solution
        status_text.text("Step 4: Extracting optimal solution...")
        progress_bar.progress(80)
        
        st.markdown('<div class="solution-box">', unsafe_allow_html=True)
        st.subheader("🎯 Step 3: Optimal Production Plan")
        
        # Create production comparison table
        production_data = []
        total_current = sum(solver.current_production)
        total_optimal = 0
        
        for i, product_code in enumerate(solver.product_codes):
            product_info = solver.solution[product_code]
            current = solver.current_production[i]
            optimal = product_info['optimal']
            change = optimal - current
            
            production_data.append({
                'Product': solver.products[i],
                'Current': current,
                'Optimal': f"{optimal:.2f}",
                'Change': f"{change:+.2f}",
                'Profit/Box': f"₹{solver.profit_margins[i]}",
                'Profit Contribution': f"₹{product_info['total_profit']:.2f}"
            })
            total_optimal += optimal
        
        # Display as dataframe
        df_production = pd.DataFrame(production_data)
        st.dataframe(df_production, use_container_width=True)
        
        # Calculate and display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Production", 
                f"{total_optimal:.0f} boxes",
                f"{total_optimal - total_current:+.0f} boxes"
            )
        
        with col2:
            st.metric(
                "Optimal Profit", 
                f"₹{solver.optimal_value:.2f}",
                f"₹{solver.optimal_value - solver.current_profit:+.2f}"
            )
        
        with col3:
            if solver.current_profit > 0:
                improvement = ((solver.optimal_value - solver.current_profit) / solver.current_profit) * 100
                st.metric(
                    "Improvement", 
                    f"{improvement:.1f}%"
                )
            else:
                st.metric("Improvement", "N/A")
        
        with col4:
            avg_profit_per_box = solver.optimal_value / total_optimal if total_optimal > 0 else 0
            st.metric(
                "Avg. Profit/Box", 
                f"₹{avg_profit_per_box:.2f}"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 5: Sensitivity Analysis
        status_text.text("Step 5: Performing sensitivity analysis...")
        progress_bar.progress(100)
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🔍 Step 4: Sensitivity Analysis & Business Insights")
        
        # Resource utilization analysis
        st.markdown("### 📊 Resource Utilization Analysis")
        utilizations = solver.get_resource_utilization()
        
        # Display resource utilization metrics
        cols = st.columns(4)
        for i, util in enumerate(utilizations[:4]):
            with cols[i]:
                if util['utilization'] > 95:
                    st.error(f"**{util['resource']}**\n{util['utilization']:.1f}% utilized")
                elif util['utilization'] > 80:
                    st.warning(f"**{util['resource']}**\n{util['utilization']:.1f}% utilized")
                else:
                    st.success(f"**{util['resource']}**\n{util['utilization']:.1f}% utilized")
        
        if len(utilizations) > 4:
            cols = st.columns(4)
            for i, util in enumerate(utilizations[4:]):
                with cols[i]:
                    if util['utilization'] > 95:
                        st.error(f"**{util['resource']}**\n{util['utilization']:.1f}% utilized")
                    elif util['utilization'] > 80:
                        st.warning(f"**{util['resource']}**\n{util['utilization']:.1f}% utilized")
                    else:
                        st.success(f"**{util['resource']}**\n{util['utilization']:.1f}% utilized")
        
        # Product profitability analysis
        st.markdown("### 📈 Product Profitability Analysis")
        
        # Sort products by profit contribution
        product_profits = []
        for product_code in solver.product_codes:
            product_info = solver.solution[product_code]
            product_profits.append({
                'Product': product_info['name'],
                'Production': product_info['optimal'],
                'Profit/Box': product_info['profit_per_unit'],
                'Total Profit': product_info['total_profit'],
                'Share of Total': (product_info['total_profit'] / solver.optimal_value * 100) if solver.optimal_value > 0 else 0
            })
        
        # Sort by total profit
        product_profits.sort(key=lambda x: x['Total Profit'], reverse=True)
        
        # Display top products
        cols = st.columns(3)
        for i in range(min(3, len(product_profits))):
            with cols[i]:
                product = product_profits[i]
                st.metric(
                    f"#{i+1}: {product['Product'].split('(')[0].strip()}",
                    f"₹{product['Total Profit']:.2f}",
                    f"{product['Share of Total']:.1f}% of total"
                )
        
        # Managerial recommendations
        st.markdown("### 🎯 Managerial Recommendations")
        
        recommendations = [
            ("🚀 **Immediate Actions**", [
                f"Increase {product_profits[0]['Product'].split('(')[0].strip()} production",
                f"Focus on high-margin products identified above",
                "Maintain minimum production for all products"
            ]),
            ("💰 **Investment Priorities**", [
                "Expand capacity for bottleneck resources (>95% utilization)",
                "Improve efficiency of fully utilized processes",
                "Consider outsourcing low-margin product lines"
            ]),
            ("📊 **Operational Improvements**", [
                "Implement real-time resource monitoring",
                "Optimize production scheduling",
                "Regularly review and update minimum production requirements"
            ])
        ]
        
        for title, items in recommendations:
            with st.expander(title):
                for item in items:
                    st.markdown(f"- {item}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        status_text.text("✅ Solution complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        # Download Section
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        download_cols = st.columns(3)
        
        with download_cols[0]:
            # Create production plan CSV
            production_df = pd.DataFrame(production_data)
            csv_production = production_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Production Plan",
                data=csv_production,
                file_name="freshdrinks_production_plan.csv",
                mime="text/csv"
            )
        
        with download_cols[1]:
            # Create resource utilization CSV
            resource_df = pd.DataFrame(utilizations)
            csv_resources = resource_df.to_csv(index=False)
            
            st.download_button(
                label="📈 Download Resource Analysis",
                data=csv_resources,
                file_name="freshdrinks_resource_analysis.csv",
                mime="text/csv"
            )
        
        with download_cols[2]:
            # Create solution summary
            summary_data = {
                'Metric': ['Optimal Weekly Profit', 'Total Production', 'Number of Iterations',
                          'Number of Products', 'Number of Constraints', 'Solution Method'],
                'Value': [f"₹{solver.optimal_value:.2f}", f"{total_optimal:.0f} boxes",
                         f"{len(solver.iterations_history)-1}", '10', '17', 'Big M Simplex']
            }
            
            summary_df = pd.DataFrame(summary_data)
            csv_summary = summary_df.to_csv(index=False)
            
            st.download_button(
                label="📋 Download Solution Summary",
                data=csv_summary,
                file_name="freshdrinks_solution_summary.csv",
                mime="text/csv"
            )

    else:
        # Instructions when no solution generated
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h2 style='color: #1E88E5;'>🎯 Ready to Optimize Production?</h2>
            <p style='font-size: 1.2rem; color: #666; margin: 1rem 0;'>
                Click the button above to see the complete simplex method solution with:
            </p>
            
            <div style='display: flex; justify-content: center; gap: 2rem; margin: 2rem 0; flex-wrap: wrap;'>
                <div style='text-align: center; padding: 1rem; border-radius: 10px; background: #E3F2FD; width: 150px;'>
                    <div style='font-size: 2.5rem;'>📊</div>
                    <div style='font-weight: bold;'>Full Tableau</div>
                    <div>Display</div>
                </div>
                <div style='text-align: center; padding: 1rem; border-radius: 10px; background: #F1F8E9; width: 150px;'>
                    <div style='font-size: 2.5rem;'>🔄</div>
                    <div style='font-weight: bold;'>Step-by-Step</div>
                    <div>Iterations</div>
                </div>
                <div style='text-align: center; padding: 1rem; border-radius: 10px; background: #FFF3E0; width: 150px;'>
                    <div style='font-size: 2.5rem;'>🎯</div>
                    <div style='font-weight: bold;'>Optimal</div>
                    <div>Solution</div>
                </div>
                <div style='text-align: center; padding: 1rem; border-radius: 10px; background: #F3E5F5; width: 150px;'>
                    <div style='font-size: 2.5rem;'>📈</div>
                    <div style='font-weight: bold;'>Sensitivity</div>
                    <div>Analysis</div>
                </div>
            </div>
            
            <p style='font-size: 1.1rem; color: #666; margin-top: 2rem;'>
                This simulator demonstrates the <strong>Big M Simplex Method</strong> for solving linear programming problems
                with both ≤ and ≥ constraints.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Learning Objectives
        st.markdown("### 🎓 What You'll Learn:")
        
        learning_cols = st.columns(2)
        
        with learning_cols[0]:
            st.markdown("""
            **Simplex Method Steps:**
            1. Problem formulation in standard form
            2. Adding slack, surplus, and artificial variables
            3. Building initial tableau
            4. Identifying entering and leaving variables
            5. Performing pivot operations
            6. Checking optimality conditions
            7. Interpreting the final solution
            """)
        
        with learning_cols[1]:
            st.markdown("""
            **Key Concepts Demonstrated:**
            - Big M method for handling ≥ constraints
            - Two-phase simplex approach
            - Basic feasible solutions
            - Reduced costs and optimality
            - Shadow prices and sensitivity
            - Resource utilization analysis
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;'>
        <p><strong>🥤 FreshDrinks Co. - Production Optimization Simplex Solver</strong></p>
        <p>Operations Research Application | Linear Programming with Big M Method</p>
        <p>📚 Educational Tool for Understanding the Simplex Algorithm</p>
        <div style='margin-top: 1rem; font-size: 0.8rem; color: #999;'>
            <p>Note: This is a simulation tool for educational purposes.</p>
            <p>For actual business implementation, consult with operations research specialists.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
