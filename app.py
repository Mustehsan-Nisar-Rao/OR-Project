import streamlit as st
import numpy as np
import pandas as pd
import time
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="FreshDrinks Simplex Solver",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .section-box {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tableau-box {
        background-color: #FFF3E0;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #FF9800;
        overflow-x: auto;
        font-family: monospace;
    }
    .iteration-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    .insight-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2196F3;
    }
    .stButton button {
        background: linear-gradient(90deg, #1E88E5, #43A047);
        color: white;
        border: none;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #1565C0, #2E7D32);
    }
    .tableau-table {
        width: 100%;
        border-collapse: collapse;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .tableau-table th, .tableau-table td {
        border: 1px solid #ddd;
        padding: 4px 6px;
        text-align: right;
        min-width: 60px;
    }
    .tableau-table th {
        background-color: #f5f5f5;
        font-weight: bold;
        position: sticky;
        top: 0;
    }
    .basic-var {
        background-color: #E8F5E9;
        font-weight: bold;
    }
    .pivot-cell {
        background-color: #FFF3E0;
        font-weight: bold;
        border: 2px solid #FF9800 !important;
    }
    .optimal-cell {
        background-color: #C8E6C9;
        font-weight: bold;
    }
    .slack-var {
        color: #1976D2;
    }
    .surplus-var {
        color: #7B1FA2;
    }
    .artificial-var {
        color: #D32F2F;
    }
    .decision-var {
        color: #388E3C;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
        
        # Store iteration history for Streamlit
        self.iterations_history = []
        self.pivot_history = []
        
        # Problem data
        self.products = [
            'Orange Juice (x1)', 'Apple Juice (x2)', 'Mango Juice (x3)', 
            'Lemon Drink (x4)', 'Energy Drink (x5)', 'Sports Drink (x6)',
            'Vitamin Water (x7)', 'Sparkling Water (x8)', 'Iced Tea (x9)', 
            'Cold Coffee (x10)'
        ]
        
        self.product_codes = ['OJ', 'AJ', 'MJ', 'LD', 'ED', 'SD', 'VW', 'SPW', 'IT', 'CC']
        
        self.num_products = len(self.products)
        self.profit_margins = [20, 18, 25, 15, 30, 22, 17, 10, 12, 16]
        
        # Resource consumption matrix
        self.resource_consumption = np.array([
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
        
        # Solution status
        self.is_optimal = False
        self.is_feasible = True
        self.message = ""
    
    def check_feasibility(self):
        """Check if the problem is feasible with given constraints"""
        # Check minimum production storage requirement
        min_storage_needed = sum(self.min_production)
        if min_storage_needed > self.storage_capacity:
            self.is_feasible = False
            self.message = f"Minimum production requires {min_storage_needed} boxes, but storage capacity is only {self.storage_capacity}"
            return False
        
        # Check resource requirements for minimum production
        for i in range(6):
            min_resource_needed = sum(self.min_production[j] * self.resource_consumption[j, i]
                                     for j in range(self.num_products))
            if min_resource_needed > self.resource_limits[i]:
                self.is_feasible = False
                self.message = f"{self.resource_names[i]} requires {min_resource_needed:.1f}, but only {self.resource_limits[i]} available"
                return False
        
        self.is_feasible = True
        self.message = "Problem is feasible"
        return True
    
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
            for j in range(self.num_products):
                self.tableau[i, j] = self.resource_consumption[j, i]
            slack_idx = self.num_products + i
            self.tableau[i, slack_idx] = 1
            self.tableau[i, -1] = self.resource_limits[i]
            self.basic_vars.append(f'S{i+1}')
        
        # ===== CONSTRAINT 7: STORAGE CONSTRAINT (<=) =====
        storage_row = 6
        for j in range(self.num_products):
            self.tableau[storage_row, j] = 1
        slack_idx = self.num_products + 6
        self.tableau[storage_row, slack_idx] = 1
        self.tableau[storage_row, -1] = self.storage_capacity
        self.basic_vars.append('S7')
        
        # ===== CONSTRAINT 8-17: MINIMUM PRODUCTION (>=) =====
        for i in range(self.num_products):
            row_idx = 7 + i
            self.tableau[row_idx, i] = 1
            surplus_idx = self.num_products + 7 + i
            self.tableau[row_idx, surplus_idx] = -1
            artificial_idx = self.num_products + 7 + self.num_products + i
            self.tableau[row_idx, artificial_idx] = 1
            self.tableau[row_idx, -1] = self.min_production[i]
            self.basic_vars.append(f'A{i+1}')
        
        # ===== OBJECTIVE FUNCTION =====
        for i in range(self.num_products):
            self.tableau[-1, i] = -self.profit_margins[i]
        
        artificial_start_idx = self.num_products + 7 + self.num_products
        for i in range(self.num_products):
            self.tableau[-1, artificial_start_idx + i] = self.M
        
        # Eliminate artificial variables from objective row
        for i in range(self.num_products):
            row_idx = 7 + i
            if abs(self.tableau[row_idx, -1]) > 1e-6:
                self.tableau[-1] -= self.M * self.tableau[row_idx]
        
        # Reset iteration history
        self.iterations_history = []
        self.pivot_history = []
        self.is_optimal = False
        
        # Save initial tableau
        self.save_iteration(0, "Initial Tableau")
    
    def save_iteration(self, iteration, description=""):
        """Save current tableau state for display"""
        self.iterations_history.append({
            'iteration': iteration,
            'tableau': self.tableau.copy(),
            'basic_vars': self.basic_vars.copy(),
            'var_names': self.var_names.copy(),
            'description': description,
            'objective_value': -self.tableau[-1, -1]
        })
    
    def check_optimality(self):
        """Check if current solution is optimal"""
        if self.tableau is None:
            return False
        
        obj_row = self.tableau[-1, :-1]
        
        # Skip artificial variables
        artificial_start = self.num_products + 7 + self.num_products
        for i in range(artificial_start, len(obj_row)):
            if abs(obj_row[i]) > self.M/2:
                obj_row[i] = float('inf')
        
        min_val = np.min(obj_row)
        
        # If all reduced costs are non-negative, we're optimal
        if min_val >= -1e-10:
            # Check if any artificial variables are still in basis with non-zero value
            artificial_in_basis = False
            for i, basic_var in enumerate(self.basic_vars):
                if basic_var.startswith('A') and i < len(self.tableau) and abs(self.tableau[i, -1]) > 1e-6:
                    artificial_in_basis = True
                    break
            
            if artificial_in_basis:
                self.message = "No feasible solution found - artificial variables remain in basis"
                return False
            
            self.is_optimal = True
            self.message = "Optimal solution reached"
            return True
        
        return False
    
    def perform_iteration(self, iteration):
        """Perform one simplex iteration"""
        if self.is_optimal:
            return False, "Already at optimal solution"
        
        # Find entering variable (most negative reduced cost)
        obj_row = self.tableau[-1, :-1]
        
        # Skip artificial variables
        artificial_start = self.num_products + 7 + self.num_products
        for i in range(artificial_start, len(obj_row)):
            if abs(obj_row[i]) > self.M/2:
                obj_row[i] = float('inf')
        
        min_val = np.min(obj_row)
        pivot_col = np.argmin(obj_row)
        
        if min_val >= -1e-10:
            self.check_optimality()
            return False, "Optimal solution reached"
        
        # Find leaving variable using minimum ratio test
        ratios = []
        for i in range(self.num_constraints):
            if self.tableau[i, pivot_col] > 1e-10:
                ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                if ratio >= 0:
                    ratios.append(ratio)
                else:
                    ratios.append(float('inf'))
            else:
                ratios.append(float('inf'))
        
        if all(r == float('inf') for r in ratios):
            return False, "Problem is unbounded"
        
        min_ratio = float('inf')
        pivot_row = -1
        for i, ratio in enumerate(ratios):
            if 0 <= ratio < min_ratio:
                min_ratio = ratio
                pivot_row = i
        
        if pivot_row == -1:
            return False, "No valid pivot row found"
        
        # Store pivot info before performing pivot
        pivot_info = {
            'iteration': iteration,
            'entering': self.var_names[pivot_col],
            'leaving': self.basic_vars[pivot_row],
            'pivot_element': self.tableau[pivot_row, pivot_col],
            'pivot_row': pivot_row,
            'pivot_col': pivot_col,
            'ratio': min_ratio
        }
        
        # Perform pivot operation
        pivot_element = self.tableau[pivot_row, pivot_col]
        
        if abs(pivot_element) < 1e-10:
            return False, "Pivot element is zero"
        
        # Save state before pivot for display
        self.save_iteration(iteration, f"Before pivot: {pivot_info['entering']} enters, {pivot_info['leaving']} leaves")
        
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
        
        # Add pivot info to history
        self.pivot_history.append(pivot_info)
        
        # Check optimality after pivot
        is_optimal = self.check_optimality()
        
        # Save state after pivot
        if is_optimal:
            description = f"Optimal solution reached after pivot"
        else:
            description = f"After pivot: {pivot_info['entering']} in basis, {pivot_info['leaving']} out"
        
        self.save_iteration(iteration + 0.5, description)
        
        return True, f"Iteration {iteration} complete: {pivot_info['entering']} enters, {pivot_info['leaving']} leaves"
    
    def solve_complete(self):
        """Solve the complete problem automatically"""
        self.build_problem()
        
        max_iterations = 50
        for iteration in range(1, max_iterations + 1):
            success, message = self.perform_iteration(iteration)
            if not success:
                break
        
        # Extract solution if optimal
        if self.is_optimal:
            self.extract_solution()
        
        return self.is_optimal, self.message
    
    def get_tableau_html(self, iteration_idx):
        """Generate HTML for tableau display"""
        if iteration_idx >= len(self.iterations_history):
            return ""
        
        iteration_data = self.iterations_history[iteration_idx]
        tableau = iteration_data['tableau']
        basic_vars = iteration_data['basic_vars']
        var_names = iteration_data['var_names']
        description = iteration_data['description']
        
        # Get pivot info for this iteration if available
        pivot_row = -1
        pivot_col = -1
        for pivot_info in self.pivot_history:
            if pivot_info['iteration'] == int(iteration_data['iteration']):
                pivot_row = pivot_info['pivot_row']
                pivot_col = pivot_info['pivot_col']
                break
        
        html = f'<h4>{description}</h4>'
        
        # Show iteration number and objective value
        html += f'<p><strong>Iteration:</strong> {iteration_data["iteration"]} | '
        html += f'<strong>Objective Value (Z):</strong> {iteration_data["objective_value"]:.2f}</p>'
        
        # Show pivot info if available
        if pivot_row != -1:
            pivot_info = next((p for p in self.pivot_history if p['iteration'] == int(iteration_data['iteration'])), None)
            if pivot_info:
                html += f"""
                <div style="background-color: #FFF3E0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>Pivot Operation:</strong> 
                    Entering: <span class="decision-var">{pivot_info['entering']}</span> | 
                    Leaving: <span class="basic-var">{pivot_info['leaving']}</span> | 
                    Pivot Element: {pivot_info['pivot_element']:.4f} |
                    Ratio: {pivot_info['ratio']:.2f}
                </div>
                """
        
        # Create tableau table
        html += '<table class="tableau-table">'
        
        # Header row
        html += '<tr><th>BV</th>'
        for var in var_names[:-1]:
            # Color code variables
            var_class = ""
            if var.startswith('x'):
                var_class = "decision-var"
            elif var.startswith('S'):
                var_class = "slack-var"
            elif var.startswith('s'):
                var_class = "surplus-var"
            elif var.startswith('A'):
                var_class = "artificial-var"
            
            html += f'<th class="{var_class}">{var}</th>'
        html += '<th>RHS</th></tr>'
        
        # Constraint rows
        for i in range(self.num_constraints):
            html += '<tr>'
            # Basic variable
            basic_var = basic_vars[i] if i < len(basic_vars) else ""
            html += f'<td class="basic-var">{basic_var}</td>'
            
            # Coefficients
            for j in range(len(var_names) - 1):
                cell_class = ""
                if i == pivot_row and j == pivot_col:
                    cell_class = "pivot-cell"
                
                value = tableau[i, j]
                html += f'<td class="{cell_class}">{value:8.3f}</td>'
            
            # RHS
            html += f'<td>{tableau[i, -1]:8.2f}</td></tr>'
        
        # Objective row
        html += '<tr><td><strong>Z</strong></td>'
        for j in range(len(var_names) - 1):
            value = tableau[-1, j]
            html += f'<td>{value:8.2f}</td>'
        html += f'<td><strong>{tableau[-1, -1]:8.2f}</strong></td></tr>'
        
        html += '</table>'
        
        # Show optimality status
        if "Optimal" in description:
            html += '<div style="color: green; font-weight: bold; margin-top: 10px;">✓ OPTIMAL SOLUTION REACHED</div>'
        
        return html
    
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
                optimal = min_req
            
            optimal = max(optimal, min_req)
            
            self.solution[product_code] = {
                'name': product_name,
                'optimal': optimal,
                'minimum': min_req,
                'current': current,
                'profit_per_unit': self.profit_margins[i],
                'total_profit': optimal * self.profit_margins[i],
                'change': optimal - current,
                'change_pct': ((optimal - current) / current * 100) if current > 0 else 0
            }
            total_optimal += optimal
        
        # Calculate total profit
        self.optimal_value = sum(data['total_profit'] for data in self.solution.values())
    
    def get_resource_utilization(self):
        """Calculate resource utilization for optimal solution"""
        utilizations = []
        
        for i in range(6):
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
    st.markdown('<div class="main-header">🥤 FreshDrinks Production Optimization Simplex Solver</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'solver' not in st.session_state:
        st.session_state.solver = FreshDrinksSimplexSolver()
        st.session_state.solver.build_problem()
    
    if 'current_iteration' not in st.session_state:
        st.session_state.current_iteration = 0
    
    if 'total_iterations' not in st.session_state:
        st.session_state.total_iterations = 1
    
    solver = st.session_state.solver
    
    # Problem Overview
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📋 Problem Overview")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        **Objective:** Maximize weekly profit from 10 beverage products
        
        **Constraints:**
        - 6 Resource constraints (Fruit, Sugar, Bottles, Mixing, Labeling, Labor)
        - 1 Storage capacity constraint
        - 10 Minimum production requirements
        - All variables non-negative
        
        **Method:** Big M Simplex Method
        """)
    
    with col2:
        st.markdown("""
        **Key Metrics:**
        """)
        
        metrics_cols = st.columns(2)
        with metrics_cols[0]:
            st.metric("Current Profit", f"${solver.current_profit:.2f}")
            st.metric("Total Products", "10")
            st.metric("Current Production", f"{sum(solver.current_production):.0f} boxes")
        
        with metrics_cols[1]:
            st.metric("Constraints", "17")
            st.metric("Variables", "37")
            st.metric("Big M Value", f"{solver.M}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Control Buttons
    st.markdown("---")
    st.markdown("### 🎮 Simplex Method Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Run Complete Solution", use_container_width=True):
            with st.spinner("Solving with Simplex Method..."):
                # Solve the complete problem
                success, message = solver.solve_complete()
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
                
                # Update session state
                st.session_state.total_iterations = len(solver.iterations_history)
                st.session_state.current_iteration = st.session_state.total_iterations - 1
                st.rerun()
    
    with col2:
        if st.button("🔧 Build/Rebuild Initial Tableau", use_container_width=True):
            solver.build_problem()
            st.session_state.current_iteration = 0
            st.session_state.total_iterations = 1
            st.success("✅ Initial tableau built successfully!")
            st.rerun()
    
    with col3:
        if st.button("⏭️ Perform Next Iteration", use_container_width=True):
            if not solver.is_optimal:
                # Perform next iteration
                iteration_num = len(solver.pivot_history) + 1
                success, message = solver.perform_iteration(iteration_num)
                
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.warning(f"⚠️ {message}")
                
                # Update session state
                st.session_state.total_iterations = len(solver.iterations_history)
                st.session_state.current_iteration = st.session_state.total_iterations - 1
                st.rerun()
            else:
                st.info("Already at optimal solution!")
    
    with col4:
        if st.button("⏮️ Previous Iteration", use_container_width=True):
            if st.session_state.current_iteration > 0:
                st.session_state.current_iteration -= 1
                st.rerun()
    
    # Display current iteration info
    st.markdown(f"**Current Iteration:** {st.session_state.current_iteration} / {st.session_state.total_iterations - 1}")
    
    # Status information
    if hasattr(solver, 'is_optimal') and solver.is_optimal:
        st.success("✅ **Optimal Solution Reached!**")
    elif hasattr(solver, 'message') and solver.message:
        st.info(f"**Status:** {solver.message}")
    
    # Tableau Display
    if hasattr(solver, 'iterations_history') and len(solver.iterations_history) > 0:
        st.markdown("---")
        st.markdown("### 📊 Simplex Tableau Display")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Current Tableau", "All Iterations", "Pivot History"])
        
        with tab1:
            st.markdown('<div class="tableau-box">', unsafe_allow_html=True)
            
            # Show iteration selector
            selected_iteration = st.slider(
                "Select iteration to view:",
                min_value=0,
                max_value=len(solver.iterations_history)-1,
                value=st.session_state.current_iteration,
                key="iteration_slider"
            )
            
            if selected_iteration != st.session_state.current_iteration:
                st.session_state.current_iteration = selected_iteration
                st.rerun()
            
            # Display selected tableau
            tableau_html = solver.get_tableau_html(st.session_state.current_iteration)
            st.markdown(tableau_html, unsafe_allow_html=True)
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀ Previous", key="prev_tableau"):
                    if st.session_state.current_iteration > 0:
                        st.session_state.current_iteration -= 1
                        st.rerun()
            
            with col2:
                st.markdown(f"**Viewing: Iteration {st.session_state.current_iteration}**")
            
            with col3:
                if st.button("Next ▶", key="next_tableau"):
                    if st.session_state.current_iteration < len(solver.iterations_history) - 1:
                        st.session_state.current_iteration += 1
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### All Iterations")
            
            # Show all iterations in expanders
            for i in range(len(solver.iterations_history)):
                iteration_data = solver.iterations_history[i]
                with st.expander(f"Iteration {iteration_data['iteration']}: {iteration_data['description']}", 
                               expanded=(i == st.session_state.current_iteration)):
                    tableau_html = solver.get_tableau_html(i)
                    st.markdown(tableau_html, unsafe_allow_html=True)
                    
                    # Button to view this iteration
                    if st.button(f"View This Iteration", key=f"view_iter_{i}"):
                        st.session_state.current_iteration = i
                        st.rerun()
        
        with tab3:
            st.markdown("### Pivot Operation History")
            
            if hasattr(solver, 'pivot_history') and solver.pivot_history:
                # Create pivot history table
                pivot_data = []
                for pivot in solver.pivot_history:
                    pivot_data.append({
                        'Iteration': pivot['iteration'],
                        'Entering Variable': pivot['entering'],
                        'Leaving Variable': pivot['leaving'],
                        'Pivot Element': f"{pivot['pivot_element']:.4f}",
                        'Ratio': f"{pivot['ratio']:.2f}"
                    })
                
                pivot_df = pd.DataFrame(pivot_data)
                st.dataframe(pivot_df, use_container_width=True)
                
                st.markdown("**Pivot Operations Summary:**")
                for pivot in solver.pivot_history:
                    st.markdown(f"""
                    - **Iteration {pivot['iteration']}:** 
                      *Entering:* {pivot['entering']}, 
                      *Leaving:* {pivot['leaving']}, 
                      *Pivot Element:* {pivot['pivot_element']:.4f}, 
                      *Ratio:* {pivot['ratio']:.2f}
                    """)
            else:
                st.info("No pivot operations performed yet.")
    
    # Optimal Solution Display
    if hasattr(solver, 'is_optimal') and solver.is_optimal and hasattr(solver, 'solution') and solver.solution:
        st.markdown("---")
        st.markdown("### 🎯 Optimal Production Plan")
        
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        
        # Production comparison
        st.markdown("#### Production Quantities (Boxes/Week)")
        
        production_data = []
        for product_code in solver.product_codes:
            if product_code in solver.solution:
                data = solver.solution[product_code]
                production_data.append({
                    'Product': data['name'],
                    'Current': data['current'],
                    'Optimal': data['optimal'],
                    'Change': data['change'],
                    'Change %': f"{data['change_pct']:.1f}%",
                    'Profit/Box': f"${data['profit_per_unit']}",
                    'Total Profit': f"${data['total_profit']:.2f}"
                })
        
        df_production = pd.DataFrame(production_data)
        st.dataframe(df_production, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_current = sum(solver.current_production)
            total_optimal = sum(data['optimal'] for data in solver.solution.values())
            st.metric("Total Production", f"{total_optimal:.0f} boxes", 
                     f"{total_optimal - total_current:+.0f} boxes")
        
        with col2:
            st.metric("Optimal Profit", f"${solver.optimal_value:.2f}",
                     f"${solver.optimal_value - solver.current_profit:+.2f}")
        
        with col3:
            improvement_pct = ((solver.optimal_value - solver.current_profit) / solver.current_profit * 100) if solver.current_profit > 0 else 0
            st.metric("Improvement", f"{improvement_pct:.1f}%")
        
        with col4:
            avg_profit = solver.optimal_value / total_optimal if total_optimal > 0 else 0
            st.metric("Avg. Profit/Box", f"${avg_profit:.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Variable Legend
    with st.expander("📚 Variable Legend"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Decision Variables:**")
            st.markdown("- x₁ to x₁₀: Production quantities")
            st.markdown("- **Color:** Green")
        
        with col2:
            st.markdown("**Slack Variables:**")
            st.markdown("- S₁ to S₇: For ≤ constraints")
            st.markdown("- **Color:** Blue")
        
        with col3:
            st.markdown("**Surplus Variables:**")
            st.markdown("- s₁ to s₁₀: For ≥ constraints")
            st.markdown("- **Color:** Purple")
        
        with col4:
            st.markdown("**Artificial Variables:**")
            st.markdown("- A₁ to A₁₀: For Big M method")
            st.markdown("- **Color:** Red")
        
        st.markdown("""
        **Tableau Features:**
        - **Basic Variables (BV):** Variables in the basis (highlighted)
        - **Pivot Cell:** Orange border indicates pivot element
        - **RHS:** Right-hand side values
        - **Z:** Objective function row
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;'>
        <p><strong>🥤 FreshDrinks Production Optimization - Simplex Method Solver</strong></p>
        <p>Operations Research Application | Big M Simplex Method</p>
        <p>Educational Tool for Linear Programming Visualization</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
