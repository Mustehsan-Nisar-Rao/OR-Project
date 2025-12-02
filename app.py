import streamlit as st
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="FreshDrinks Simplex Solver",
    page_icon="🥤",
    layout="wide"
)

# Custom CSS for simple styling
st.markdown("""
    <style>
    .tableau-table {
        width: 100%;
        border-collapse: collapse;
        font-family: monospace;
        font-size: 0.85rem;
        margin: 10px 0;
    }
    .tableau-table th, .tableau-table td {
        border: 1px solid #ddd;
        padding: 4px 6px;
        text-align: right;
        min-width: 50px;
    }
    .tableau-table th {
        background-color: #f5f5f5;
        font-weight: bold;
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
            'OJ', 'AJ', 'MJ', 'LD', 'ED',
            'SD', 'VW', 'SPW', 'IT', 'CC'
        ]
        
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
        self.min_production = [40, 30, 20, 25, 15, 10, 12, 8, 10, 12]
        self.storage_capacity = 900
        self.current_production = [60, 50, 45, 40, 30, 35, 25, 20, 25, 30]
        self.current_profit = sum(self.current_production[i] * self.profit_margins[i]
                                 for i in range(self.num_products))
        self.num_constraints = 0
        
        # For tracking
        self.iterations = []
        self.pivots = []
    
    def build_problem(self):
        """Build the initial tableau"""
        self.num_constraints = 6 + 1 + self.num_products
        self.var_names = [f'x{i+1}' for i in range(self.num_products)]
        
        # Add slack, surplus, and artificial variables
        for i in range(7):
            self.var_names.append(f'S{i+1}')
        for i in range(self.num_products):
            self.var_names.append(f's{i+1}')
        for i in range(self.num_products):
            self.var_names.append(f'A{i+1}')
        self.var_names.append('RHS')
        
        total_vars = self.num_products + 7 + self.num_products + self.num_products
        self.tableau = np.zeros((self.num_constraints + 1, total_vars + 1))
        self.basic_vars = []
        
        # Resource constraints
        for i in range(6):
            for j in range(self.num_products):
                self.tableau[i, j] = self.resource_consumption[j, i]
            slack_idx = self.num_products + i
            self.tableau[i, slack_idx] = 1
            self.tableau[i, -1] = self.resource_limits[i]
            self.basic_vars.append(f'S{i+1}')
        
        # Storage constraint
        storage_row = 6
        for j in range(self.num_products):
            self.tableau[storage_row, j] = 1
        slack_idx = self.num_products + 6
        self.tableau[storage_row, slack_idx] = 1
        self.tableau[storage_row, -1] = self.storage_capacity
        self.basic_vars.append('S7')
        
        # Minimum production constraints
        for i in range(self.num_products):
            row_idx = 7 + i
            self.tableau[row_idx, i] = 1
            surplus_idx = self.num_products + 7 + i
            self.tableau[row_idx, surplus_idx] = -1
            artificial_idx = self.num_products + 7 + self.num_products + i
            self.tableau[row_idx, artificial_idx] = 1
            self.tableau[row_idx, -1] = self.min_production[i]
            self.basic_vars.append(f'A{i+1}')
        
        # Objective function
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
    
    def find_pivot_column(self):
        """Find entering variable"""
        obj_row = self.tableau[-1, :-1]
        artificial_start = self.num_products + 7 + self.num_products
        
        for i in range(artificial_start, len(obj_row)):
            if abs(obj_row[i]) > self.M/2:
                obj_row[i] = float('inf')
        
        min_val = np.min(obj_row)
        if min_val >= -1e-10:
            return -1
        
        return np.argmin(obj_row)
    
    def find_pivot_row(self, pivot_col):
        """Find leaving variable"""
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
            return -1
        
        min_ratio = float('inf')
        min_idx = -1
        for i, ratio in enumerate(ratios):
            if 0 <= ratio < min_ratio:
                min_ratio = ratio
                min_idx = i
        
        return min_idx
    
    def pivot(self, pivot_row, pivot_col):
        """Perform pivot operation"""
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row] = self.tableau[pivot_row] / pivot_element
        
        for i in range(len(self.tableau)):
            if i != pivot_row:
                multiplier = self.tableau[i, pivot_col]
                if abs(multiplier) > 1e-10:
                    self.tableau[i] = self.tableau[i] - multiplier * self.tableau[pivot_row]
        
        if pivot_row < len(self.basic_vars):
            self.basic_vars[pivot_row] = self.var_names[pivot_col]
    
    def solve(self):
        """Solve using simplex method"""
        self.iterations = []
        self.pivots = []
        
        # Save initial tableau
        self.save_iteration(0)
        
        iteration = 0
        max_iterations = 50
        
        while iteration < max_iterations:
            iteration += 1
            
            # Find entering variable
            pivot_col = self.find_pivot_column()
            if pivot_col == -1:
                break
            
            # Find leaving variable
            pivot_row = self.find_pivot_row(pivot_col)
            if pivot_row == -1:
                break
            
            # Save pivot info
            pivot_info = {
                'iteration': iteration,
                'entering': self.var_names[pivot_col],
                'leaving': self.basic_vars[pivot_row],
                'pivot_element': self.tableau[pivot_row, pivot_col],
                'pivot_row': pivot_row,
                'pivot_col': pivot_col
            }
            self.pivots.append(pivot_info)
            
            # Perform pivot
            self.pivot(pivot_row, pivot_col)
            
            # Save iteration
            self.save_iteration(iteration)
        
        # Extract solution
        self.extract_solution()
        return iteration
    
    def save_iteration(self, iteration):
        """Save current tableau state"""
        self.iterations.append({
            'iteration': iteration,
            'tableau': self.tableau.copy(),
            'basic_vars': self.basic_vars.copy(),
            'var_names': self.var_names.copy(),
            'objective_value': -self.tableau[-1, -1]
        })
    
    def extract_solution(self):
        """Extract optimal solution"""
        self.solution = {}
        for i in range(self.num_products):
            product = self.products[i]
            min_req = self.min_production[i]
            current = self.current_production[i]
            
            # Find optimal value
            if f'x{i+1}' in self.basic_vars:
                idx = self.basic_vars.index(f'x{i+1}')
                optimal = self.tableau[idx, -1]
            else:
                optimal = min_req
            
            optimal = max(optimal, min_req)
            self.solution[product] = optimal
        
        # Calculate optimal profit
        self.optimal_value = sum(self.solution[product] * self.profit_margins[i] 
                               for i, product in enumerate(self.products))
    
    def get_tableau_html(self, iteration_idx):
        """Generate HTML for tableau display"""
        if iteration_idx >= len(self.iterations):
            return ""
        
        iteration = self.iterations[iteration_idx]
        tableau = iteration['tableau']
        basic_vars = iteration['basic_vars']
        var_names = iteration['var_names']
        
        # Get pivot info if available
        pivot_row = -1
        pivot_col = -1
        if iteration_idx > 0:
            for pivot in self.pivots:
                if pivot['iteration'] == iteration_idx:
                    pivot_row = pivot['pivot_row']
                    pivot_col = pivot['pivot_col']
                    break
        
        html = f'<h3>Iteration {iteration_idx}</h3>'
        
        # Show pivot info if available
        if iteration_idx > 0 and iteration_idx <= len(self.pivots):
            pivot = self.pivots[iteration_idx-1]
            html += f'<p><strong>Entering:</strong> {pivot["entering"]} | '
            html += f'<strong>Leaving:</strong> {pivot["leaving"]} | '
            html += f'<strong>Pivot Element:</strong> {pivot["pivot_element"]:.4f}</p>'
        
        # Create tableau table
        html += '<table class="tableau-table">'
        
        # Header row
        html += '<tr><th>BV</th>'
        for var in var_names[:-1]:
            html += f'<th>{var}</th>'
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
                html += f'<td class="{cell_class}">{value:7.2f}</td>'
            
            # RHS
            html += f'<td>{tableau[i, -1]:7.2f}</td></tr>'
        
        # Objective row
        html += '<tr><td><strong>Z</strong></td>'
        for j in range(len(var_names) - 1):
            value = tableau[-1, j]
            html += f'<td>{value:7.2f}</td>'
        html += f'<td><strong>{tableau[-1, -1]:7.2f}</strong></td></tr>'
        
        html += '</table>'
        
        # Show objective value
        obj_val = iteration['objective_value']
        html += f'<p><strong>Z = {obj_val:.2f}</strong></p>'
        
        return html

def main():
    st.title("🥤 FreshDrinks Simplex Solver")
    st.write("Solving production optimization using Big M Simplex Method")
    
    # Initialize solver
    solver = FreshDrinksSimplexSolver()
    
    # Build problem
    solver.build_problem()
    
    # Solve button
    if st.button("Run Simplex Algorithm", type="primary"):
        with st.spinner("Solving..."):
            iterations = solver.solve()
            st.success(f"Solved in {iterations} iterations!")
    
    # Show all iterations
    if solver.iterations:
        st.header("📊 Simplex Tableau Iterations")
        
        # Create tabs for iterations
        tabs = st.tabs([f"Iteration {i}" for i in range(len(solver.iterations))])
        
        for i, tab in enumerate(tabs):
            with tab:
                st.markdown(solver.get_tableau_html(i), unsafe_allow_html=True)
        
        # Show pivot history
        st.header("🔄 Pivot Operations")
        if solver.pivots:
            for pivot in solver.pivots:
                st.write(f"**Iteration {pivot['iteration']}:** {pivot['entering']} enters, {pivot['leaving']} leaves (pivot: {pivot['pivot_element']:.4f})")
        
        # Show solution
        st.header("🎯 Optimal Solution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Production Quantities")
            for i, product in enumerate(solver.products):
                optimal = solver.solution[product]
                current = solver.current_production[i]
                change = optimal - current
                
                st.write(f"**{product}:**")
                st.write(f"- Current: {current} boxes")
                st.write(f"- Optimal: **{optimal:.1f}** boxes")
                st.write(f"- Change: {change:+.1f} boxes")
                st.write("---")
        
        with col2:
            st.subheader("Summary")
            
            total_current = sum(solver.current_production)
            total_optimal = sum(solver.solution.values())
            
            st.metric("Total Production", 
                     f"{total_optimal:.0f} boxes", 
                     f"{total_optimal - total_current:+.0f} boxes")
            
            st.metric("Weekly Profit", 
                     f"${solver.optimal_value:.2f}", 
                     f"${solver.optimal_value - solver.current_profit:+.2f}")
            
            if solver.current_profit > 0:
                improvement = ((solver.optimal_value - solver.current_profit) / solver.current_profit) * 100
                st.metric("Improvement", f"{improvement:.1f}%")
            
            # Show which resources are bottlenecks
            st.subheader("Resource Utilization")
            
            resource_names = ["Fruit", "Sugar", "Bottles", "Mixing", "Labeling", "Labor"]
            for i in range(6):
                used = sum(solver.solution[solver.products[j]] * solver.resource_consumption[j, i] 
                          for j in range(solver.num_products))
                available = solver.resource_limits[i]
                utilization = (used / available) * 100
                
                if utilization > 95:
                    st.error(f"{resource_names[i]}: {utilization:.1f}% used")
                elif utilization > 80:
                    st.warning(f"{resource_names[i]}: {utilization:.1f}% used")
                else:
                    st.success(f"{resource_names[i]}: {utilization:.1f}% used")
            
            # Storage utilization
            storage_util = (total_optimal / solver.storage_capacity) * 100
            st.write(f"**Storage:** {storage_util:.1f}% used")
    
    else:
        st.info("Click 'Run Simplex Algorithm' to solve the problem and see iterations.")

if __name__ == "__main__":
    main()
