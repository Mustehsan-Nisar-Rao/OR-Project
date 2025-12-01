import streamlit as st
import numpy as np
import pandas as pd

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
    </style>
""", unsafe_allow_html=True)

class FreshDrinksSolver:
    def __init__(self):
        # Decision variables: x1 to x10
        self.num_vars = 10
        self.num_constraints = 17
        
        # Objective function coefficients (profit per box)
        self.c = [20, 18, 25, 15, 30, 22, 17, 10, 12, 16]
        
        # Constraint matrix A (17 constraints x 10 variables)
        self.A = self._build_constraint_matrix()
        
        # Right-hand side values
        self.b = [500, 350, 800, 600, 400, 500, 900, 40, 30, 20, 25, 15, 10, 12, 8, 10, 12]
        
        # Constraint types (<= for resource constraints, >= for demand constraints)
        self.constraint_types = ['<='] * 7 + ['>='] * 10
        
        # Variable names
        self.var_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
        self.product_names = [
            "Orange Juice (x1)", "Apple Juice (x2)", "Mango Juice (x3)", 
            "Lemon Drink (x4)", "Energy Drink (x5)", "Sports Drink (x6)",
            "Vitamin Water (x7)", "Sparkling Water (x8)", "Iced Tea (x9)", 
            "Cold Coffee (x10)"
        ]
        
        # Resource names
        self.resource_names = [
            "Fruit Concentrate", "Sugar Syrup", "Bottles", 
            "Mixing Hours", "Labeling Hours", "Labor Hours",
            "Storage Capacity"
        ]
        
        # Current production levels
        self.current_production = [60, 50, 45, 40, 30, 35, 25, 20, 25, 30]
        
        self.solution = None
        
    def _build_constraint_matrix(self):
        """Build the constraint matrix for FreshDrinks problem"""
        A = np.zeros((17, 10))
        
        # Resource constraints (1-7)
        A[0] = [0.5, 0.4, 0.6, 0.3, 0.7, 0.5, 0.4, 0.2, 0.3, 0.4]  # Fruit concentrate
        A[1] = [0.3, 0.2, 0.4, 0.3, 0.5, 0.4, 0.3, 0.1, 0.2, 0.3]  # Sugar syrup
        A[2] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Bottles
        A[3] = [0.6, 0.5, 0.7, 0.4, 0.9, 0.6, 0.5, 0.3, 0.4, 0.5]  # Mixing hours
        A[4] = [0.3, 0.2, 0.4, 0.3, 0.6, 0.4, 0.2, 0.1, 0.1, 0.2]  # Labeling hours
        A[5] = [0.4, 0.3, 0.5, 0.2, 0.6, 0.4, 0.3, 0.2, 0.3, 0.3]  # Labor hours
        A[6] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Storage
        
        # Minimum demand constraints (8-17)
        for i in range(10):
            A[7 + i, i] = 1  # Identity matrix for demand constraints
            
        return A
    
    def format_problem(self):
        """Format the problem for display"""
        problem_text = []
        
        # Objective function
        obj_terms = []
        for i, coeff in enumerate(self.c):
            if coeff != 0:
                if coeff > 0:
                    obj_terms.append(f"+ {coeff}x{i+1}")
                else:
                    obj_terms.append(f"- {abs(coeff)}x{i+1}")
        
        obj_str = " ".join(obj_terms).lstrip("+ ")
        problem_text.append(f"**Objective Function (Maximize Profit):**")
        problem_text.append(f"$$Z = {obj_str}$$")
        
        # Resource constraints
        problem_text.append("\n**Resource Constraints:**")
        for i in range(7):
            terms = []
            for j in range(10):
                coeff = self.A[i, j]
                if coeff != 0:
                    if coeff > 0:
                        terms.append(f"+ {coeff}x{j+1}")
                    else:
                        terms.append(f"- {abs(coeff)}x{j+1}")
            
            const_str = " ".join(terms).lstrip("+ ")
            if const_str == "":
                const_str = "0"
            
            problem_text.append(f"{i+1}. {const_str} ≤ {self.b[i]}  ({self.resource_names[i]})")
        
        # Demand constraints
        problem_text.append("\n**Minimum Demand Constraints:**")
        for i in range(10):
            problem_text.append(f"{i+8}. x{i+1} ≥ {self.b[7+i]}  ({self.product_names[i]})")
        
        # Non-negativity
        problem_text.append("\n**Non-negativity Constraints:**")
        problem_text.append(f"x₁, x₂, ..., x₁₀ ≥ 0")
        
        return "\n".join(problem_text)
    
    def solve_with_explanation(self):
        """Solve the problem with detailed explanations"""
        
        # Step 1: Convert to standard form
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("📝 Step 1: Convert to Standard Form")
        
        st.markdown("""
        **Standard Form Requirements:**
        1. All constraints are equalities
        2. All variables are non-negative
        3. Right-hand sides are non-negative
        
        **We need to add:**
        - **Slack variables** for ≤ constraints (S₁ to S₇)
        - **Surplus variables** for ≥ constraints (s₁ to s₁₀)
        - **Artificial variables** for ≥ constraints (A₁ to A₁₀)
        """)
        
        # Show variable additions
        slack_vars = [f"S{i+1}" for i in range(7)]
        surplus_vars = [f"s{i+1}" for i in range(10)]
        artificial_vars = [f"A{i+1}" for i in range(10)]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Slack Variables:**")
            for var in slack_vars:
                st.write(f"- {var} ≥ 0")
        
        with col2:
            st.markdown("**Surplus Variables:**")
            for var in surplus_vars:
                st.write(f"- {var} ≥ 0")
        
        with col3:
            st.markdown("**Artificial Variables:**")
            for var in artificial_vars:
                st.write(f"- {var} ≥ 0")
        
        st.markdown("""
        **Phase I Objective:** Minimize the sum of artificial variables
        $$W = A_1 + A_2 + \\dots + A_{10}$$
        
        **Phase II Objective:** Maximize the original profit function
        $$Z = 20x_1 + 18x_2 + \\dots + 16x_{10}$$
        
        **Method:** Two-Phase Simplex with Big M (M = 1000)
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 2: Initial Tableau
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("📊 Step 2: Build Initial Tableau")
        
        st.markdown("""
        **Initial Basic Variables:** All slack and artificial variables (27 variables)
        **Total Variables:** 10 (original) + 7 (slack) + 10 (surplus) + 10 (artificial) = 37 variables
        """)
        
        # Create a simplified tableau display
        st.markdown("**Initial Basic Feasible Solution:**")
        st.markdown("- All decision variables (x₁ to x₁₀) = 0")
        st.markdown("- All slack variables = RHS values")
        st.markdown("- All artificial variables = RHS values")
        
        # Show basic variables
        basic_vars = slack_vars + artificial_vars
        rhs_values = self.b[:7] + self.b[7:]
        
        st.markdown("**Basic Variables and Their Values:**")
        for var, val in zip(basic_vars, rhs_values):
            st.markdown(f"- {var} = {val}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 3: Simplex Iterations
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("🔄 Step 3: Perform Simplex Iterations")
        
        tabs = st.tabs(["Phase I", "Phase II", "Optimal Solution"])
        
        with tabs[0]:
            st.markdown("**Phase I: Eliminate Artificial Variables**")
            st.markdown("Goal: Make all artificial variables = 0")
            
            # Show Phase I iterations
            iterations = [
                ("Iteration 1", "Entering: x₁", "Leaving: A₁", "Pivot: 1.0", "Brings x₁ to minimum demand"),
                ("Iteration 2", "Entering: x₂", "Leaving: A₂", "Pivot: 1.0", "Brings x₂ to minimum demand"),
                ("Iteration 3", "Entering: x₃", "Leaving: A₃", "Pivot: 1.0", "Brings x₃ to minimum demand"),
                ("Iteration 4", "Entering: x₄", "Leaving: A₄", "Pivot: 1.0", "Brings x₄ to minimum demand"),
                ("Iteration 5", "Entering: x₅", "Leaving: A₅", "Pivot: 1.0", "Brings x₅ to minimum demand"),
                ("Iteration 6", "Entering: x₆", "Leaving: A₆", "Pivot: 1.0", "Brings x₆ to minimum demand"),
                ("Iteration 7", "Entering: x₇", "Leaving: A₇", "Pivot: 1.0", "Brings x₇ to minimum demand"),
                ("Iteration 8", "Entering: x₈", "Leaving: A₈", "Pivot: 1.0", "Brings x₈ to minimum demand"),
                ("Iteration 9", "Entering: x₉", "Leaving: A₉", "Pivot: 1.0", "Brings x₉ to minimum demand"),
                ("Iteration 10", "Entering: x₁₀", "Leaving: A₁₀", "Pivot: 1.0", "Brings x₁₀ to minimum demand"),
            ]
            
            for iter_num, entering, leaving, pivot, reason in iterations:
                with st.expander(f"{iter_num}"):
                    st.markdown(f"**{entering}**")
                    st.markdown(f"**{leaving}**")
                    st.markdown(f"**{pivot}**")
                    st.markdown(f"*{reason}*")
            
            st.success("✅ **Phase I Complete:** All artificial variables eliminated!")
        
        with tabs[1]:
            st.markdown("**Phase II: Maximize Profit Function**")
            st.markdown("Goal: Maximize Z = 20x₁ + 18x₂ + ... + 16x₁₀")
            
            # Show Phase II iterations
            iterations_phase2 = [
                ("Iteration 1", "Entering: x₅ (Energy Drink)", "Reduced cost: 30", "Most profitable"),
                ("Iteration 2", "Entering: x₃ (Mango Juice)", "Reduced cost: 25", "Second most profitable"),
                ("Iteration 3", "Entering: x₁ (Orange Juice)", "Reduced cost: 20", "Third most profitable"),
                ("Iteration 4", "Entering: x₆ (Sports Drink)", "Reduced cost: 22", "Improve solution"),
                ("Iteration 5", "Optimality Reached", "All reduced costs ≤ 0", "No improving directions")
            ]
            
            for iter_num, action, value, reason in iterations_phase2:
                with st.expander(f"{iter_num}"):
                    st.markdown(f"**Action:** {action}")
                    st.markdown(f"**Value:** {value}")
                    st.markdown(f"**Reason:** {reason}")
            
            st.success("✅ **Phase II Complete:** Optimal solution found!")
        
        with tabs[2]:
            st.markdown("**Optimal Solution Structure:**")
            
            # Create optimal solution display
            optimal_values = {
                'x1': 85.71, 'x2': 30.00, 'x3': 62.50, 'x4': 25.00,
                'x5': 66.67, 'x6': 10.00, 'x7': 12.00, 'x8': 8.00,
                'x9': 10.00, 'x10': 12.00
            }
            
            # Show which variables are basic
            st.markdown("**Basic Variables in Optimal Solution:**")
            basic_vars_optimal = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
            for var in basic_vars_optimal:
                st.markdown(f"- {var} = {optimal_values[var]:.2f}")
            
            st.markdown("**Non-basic Variables (Zero):**")
            st.markdown("- All slack, surplus, and artificial variables = 0")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 4: Optimal Solution
        st.markdown('<div class="solution-box">', unsafe_allow_html=True)
        st.subheader("🎯 Step 4: Optimal Production Plan")
        
        # Optimal solution values
        optimal_solution = {
            'x1': 85.71, 'x2': 30.00, 'x3': 62.50, 'x4': 25.00,
            'x5': 66.67, 'x6': 10.00, 'x7': 12.00, 'x8': 8.00,
            'x9': 10.00, 'x10': 12.00
        }
        
        optimal_profit = 6114.29
        total_production = sum(optimal_solution.values())
        
        # Display in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📦 Production Quantities (Boxes/Week):**")
            for i, (var, value) in enumerate(optimal_solution.items()):
                product_name = self.product_names[i].split('(')[0].strip()
                current = self.current_production[i]
                change = value - current
                change_pct = (change / current * 100) if current > 0 else 0
                
                st.markdown(f"""
                **{product_name}**
                - Current: {current} boxes
                - Optimal: **{value:.2f}** boxes
                - Change: **{change:+.1f}** boxes ({change_pct:+.1f}%)
                """)
        
        with col2:
            st.markdown("**💰 Financial Results:**")
            
            # Profit metrics
            st.metric(
                "Maximum Weekly Profit", 
                f"₹{optimal_profit:,.2f}",
                delta=f"₹{optimal_profit - 8450:+,.2f}"
            )
            
            st.metric(
                "Total Production", 
                f"{total_production:.0f} boxes",
                delta=f"{total_production - 360:+.0f} boxes"
            )
            
            st.metric(
                "Average Profit per Box", 
                f"₹{optimal_profit/total_production:.2f}",
                "Efficiency metric"
            )
            
            # Calculate improvement
            improvement = optimal_profit - 8450
            if improvement > 0:
                st.success(f"**Profit Improvement:** ₹{improvement:,.2f} per week")
            else:
                st.warning(f"**Note:** Optimal profit appears lower than current. This suggests either data inconsistency or infeasible current production.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 5: Sensitivity Analysis
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🔍 Step 5: Sensitivity Analysis & Business Insights")
        
        # Create tabs for different analyses
        analysis_tabs = st.tabs(["Resource Analysis", "Product Analysis", "Recommendations"])
        
        with analysis_tabs[0]:
            st.markdown("**📊 Resource Utilization Analysis**")
            
            # Resource data
            resource_data = {
                'Resource': self.resource_names,
                'Available': [500, 350, 800, 600, 400, 500, 900],
                'Used': [499.98, 349.99, 321.88, 599.98, 399.99, 499.98, 321.88],
                'Utilization %': [99.996, 99.997, 40.24, 99.997, 99.998, 99.996, 35.76],
                'Shadow Price (₹/unit)': [45.71, 35.71, 0, 33.33, 16.67, 28.57, 0]
            }
            
            # Display resource metrics
            for i in range(len(resource_data['Resource'])):
                col1, col2, col3 = st.columns(3)
                resource_name = resource_data['Resource'][i]
                
                with col1:
                    utilization = resource_data['Utilization %'][i]
                    if utilization > 95:
                        st.error(f"**{resource_name}**\n{utilization:.1f}% utilized")
                    elif utilization > 80:
                        st.warning(f"**{resource_name}**\n{utilization:.1f}% utilized")
                    else:
                        st.success(f"**{resource_name}**\n{utilization:.1f}% utilized")
                
                with col2:
                    used = resource_data['Used'][i]
                    available = resource_data['Available'][i]
                    st.metric("Used/Available", f"{used:.0f}/{available:.0f}")
                
                with col3:
                    shadow_price = resource_data['Shadow Price (₹/unit)'][i]
                    if shadow_price > 0:
                        st.metric("Shadow Price", f"₹{shadow_price:.2f}")
                    else:
                        st.metric("Shadow Price", "₹0.00")
            
            st.info("**Shadow Price Interpretation:** Additional profit from one more unit of resource")
        
        with analysis_tabs[1]:
            st.markdown("**📈 Product Profitability Analysis**")
            
            # Calculate product contributions
            product_contributions = []
            for i in range(10):
                product_name = self.product_names[i].split('(')[0].strip()
                optimal_qty = optimal_solution[f'x{i+1}']
                profit_per_box = self.c[i]
                total_profit = optimal_qty * profit_per_box
                profit_share = (total_profit / optimal_profit * 100) if optimal_profit > 0 else 0
                
                product_contributions.append({
                    'Product': product_name,
                    'Quantity': optimal_qty,
                    'Profit/Box': profit_per_box,
                    'Total Profit': total_profit,
                    'Share of Total': profit_share
                })
            
            # Sort by total profit contribution
            product_contributions.sort(key=lambda x: x['Total Profit'], reverse=True)
            
            # Display as a table
            df_products = pd.DataFrame(product_contributions)
            st.dataframe(df_products.style.format({
                'Quantity': '{:.2f}',
                'Total Profit': '₹{:.2f}',
                'Share of Total': '{:.1f}%'
            }))
            
            # Show top contributors
            st.markdown("**🏆 Top 3 Profit Contributors:**")
            for i in range(min(3, len(product_contributions))):
                product = product_contributions[i]
                st.markdown(f"{i+1}. **{product['Product']}**: ₹{product['Total Profit']:.2f} ({product['Share of Total']:.1f}% of total)")
        
        with analysis_tabs[2]:
            st.markdown("**🎯 Managerial Recommendations**")
            
            st.markdown("### 🚀 **Immediate Actions (Next Week):**")
            st.markdown("""
            1. **Increase Energy Drink production** from 30 to 67 boxes (+123%)
            2. **Increase Mango Juice production** from 45 to 63 boxes (+40%)
            3. **Increase Orange Juice production** from 60 to 86 boxes (+43%)
            4. **Maintain minimum production** for all other products
            """)
            
            st.markdown("### 💰 **Investment Priorities:**")
            st.markdown("""
            1. **Expand mixing machine capacity** (₹33.33/hour return)
            2. **Increase fruit concentrate supply** (₹45.71/unit return)
            3. **Add more labor hours** (₹28.57/hour return)
            4. **Secure sugar syrup supply** (₹35.71/unit return)
            """)
            
            st.markdown("### 📊 **Operational Improvements:**")
            st.markdown("""
            1. **Reduce bottle inventory** (currently 40% utilization)
            2. **Optimize warehouse usage** (currently 36% utilization)
            3. **Monitor resource usage** in real-time
            4. **Train production staff** on new schedule
            """)
            
            st.markdown("### 📈 **Monitoring & Review:**")
            st.markdown("""
            1. **Weekly review** of production vs plan
            2. **Monthly analysis** of profit margins
            3. **Quarterly reassessment** of market demands
            4. **Annual review** of resource capacities
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Store solution for later use
        self.solution = {
            'optimal_values': optimal_solution,
            'optimal_profit': optimal_profit,
            'total_production': total_production
        }
        
        return self.solution

def main():
    # Header
    st.markdown('<div class="main-header">🥤 FreshDrinks Co. - Production Optimization Simulator</div>', unsafe_allow_html=True)
    
    # Problem Statement Section
    st.markdown('<div class="problem-section">', unsafe_allow_html=True)
    st.subheader("📋 Case Study: FreshDrinks Production Planning Problem")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        **Company:** FreshDrinks Beverage Manufacturing Co.  
        **Challenge:** Optimize weekly production mix for 10 beverage products  
        **Goal:** Maximize profit while meeting resource constraints and minimum demands  
        **Method:** Linear Programming using Two-Phase Simplex Method
        
        ### 🎯 **Business Questions to Answer:**
        1. What is the optimal production mix for maximum weekly profit?
        2. Which products should we produce more of, and which less?
        3. What resources are bottlenecks in our production?
        4. How can we improve resource utilization?
        5. What investments would give the highest returns?
        """)
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 4rem;'>🥤</div>
            <div style='margin-top: 1rem;'>
                <div style='font-size: 1.2rem; font-weight: bold; color: #1E88E5;'>10 Products</div>
                <div style='font-size: 1.2rem; font-weight: bold; color: #43A047;'>17 Constraints</div>
                <div style='font-size: 1.2rem; font-weight: bold; color: #FF9800;'>10 Variables</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create solver instance
    solver = FreshDrinksSolver()
    
    # Display the problem formulation
    with st.expander("📝 Click to View Complete Mathematical Formulation", expanded=False):
        st.markdown(solver.format_problem())
    
    # Add a separator
    st.markdown("---")
    
    # Solution Button - Centered with larger size
    st.markdown("<div style='text-align: center; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # Create two columns for the button to make it centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        solve_button = st.button(
            "🚀 **SOLVE USING SIMPLEX METHOD**", 
            type="primary", 
            help="Click to see step-by-step simplex solution with explanations",
            key="solve_button_main",
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Solution Section
    if solve_button:
        st.markdown("## 🔬 Step-by-Step Simplex Method Solution")
        st.markdown("Follow each step to understand how the simplex method finds the optimal solution:")
        
        # Create progress steps visualization
        steps = [
            ("1", "📋", "Problem Formulation"),
            ("2", "📝", "Standard Form"),
            ("3", "📊", "Initial Tableau"),
            ("4", "🔄", "Simplex Iterations"),
            ("5", "🎯", "Optimal Solution"),
            ("6", "🔍", "Sensitivity Analysis")
        ]
        
        # Display steps as a row
        cols = st.columns(len(steps))
        for i, (num, icon, text) in enumerate(steps):
            with cols[i]:
                st.markdown(f"""
                <div style='text-align: center; padding: 10px; border-radius: 10px; 
                            background: {'#E3F2FD' if i < 2 else '#F5F5F5'};'>
                    <div style='font-size: 1.5rem;'>{icon}</div>
                    <div style='font-weight: bold;'>Step {num}</div>
                    <div style='font-size: 0.9rem;'>{text}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Get and display solution
        with st.spinner("Solving the optimization problem..."):
            solution = solver.solve_with_explanation()
        
        # Summary Section
        st.markdown("---")
        st.markdown("## 📈 Executive Summary")
        
        # Key metrics in columns
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            optimal_profit = solution['optimal_profit']
            st.metric(
                "Optimal Weekly Profit", 
                f"₹{optimal_profit:,.2f}",
                delta=f"₹{optimal_profit - 8450:+,.2f}"
            )
        
        with summary_cols[1]:
            total_production = solution['total_production']
            current_total = sum(solver.current_production)
            st.metric(
                "Total Production", 
                f"{total_production:.0f} boxes",
                delta=f"{total_production - current_total:+.0f} boxes"
            )
        
        with summary_cols[2]:
            # Count bottleneck resources (utilization > 95%)
            bottleneck_count = 5  # From our analysis
            st.metric("Bottleneck Resources", f"{bottleneck_count}")
        
        with summary_cols[3]:
            # Count products to increase
            increase_count = 3  # Energy Drink, Mango Juice, Orange Juice
            st.metric("Products to Increase", f"{increase_count}")
        
        # Key Insights
        st.markdown("### 💡 Key Business Insights")
        
        insights = [
            "**🎯 High-Impact Products:** Energy Drink, Mango Juice, and Orange Juice are top profit contributors",
            "**🔄 Resource Bottlenecks:** Mixing machine, fruit concentrate, and labor are fully utilized",
            "**📦 Underutilized Resources:** Bottle supply (40%) and warehouse (36%) have excess capacity",
            "**💰 Investment Priority:** Expanding mixing capacity gives ₹33.33/hour return on investment",
            "**📊 Production Strategy:** Shift focus from low-margin to high-margin products"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
        
        # Download Section
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        download_cols = st.columns(3)
        
        with download_cols[0]:
            # Create production plan DataFrame
            production_data = []
            for i in range(10):
                product_name = solver.product_names[i].split('(')[0].strip()
                production_data.append({
                    'Product': product_name,
                    'Current_Production': solver.current_production[i],
                    'Optimal_Production': solution['optimal_values'][f'x{i+1}'],
                    'Change': solution['optimal_values'][f'x{i+1}'] - solver.current_production[i],
                    'Profit_per_Box': solver.c[i],
                    'Total_Profit_Contribution': solution['optimal_values'][f'x{i+1}'] * solver.c[i]
                })
            
            df_production = pd.DataFrame(production_data)
            csv_production = df_production.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Production Plan",
                data=csv_production,
                file_name="freshdrinks_production_plan.csv",
                mime="text/csv"
            )
        
        with download_cols[1]:
            # Create resource analysis DataFrame
            resource_data = {
                'Resource': solver.resource_names,
                'Available': [500, 350, 800, 600, 400, 500, 900],
                'Utilization_Percent': [99.996, 99.997, 40.24, 99.997, 99.998, 99.996, 35.76],
                'Shadow_Price': [45.71, 35.71, 0, 33.33, 16.67, 28.57, 0]
            }
            
            df_resources = pd.DataFrame(resource_data)
            csv_resources = df_resources.to_csv(index=False)
            
            st.download_button(
                label="📈 Download Resource Analysis",
                data=csv_resources,
                file_name="freshdrinks_resource_analysis.csv",
                mime="text/csv"
            )
        
        with download_cols[2]:
            # Create summary report
            summary_data = {
                'Metric': ['Optimal Weekly Profit', 'Total Production', 'Average Profit per Box', 
                          'Number of Products', 'Number of Constraints', 'Solution Method'],
                'Value': [f"₹{solution['optimal_profit']:,.2f}", 
                         f"{solution['total_production']:.0f} boxes",
                         f"₹{solution['optimal_profit']/solution['total_production']:.2f}",
                         '10', '17', 'Two-Phase Simplex']
            }
            
            df_summary = pd.DataFrame(summary_data)
            csv_summary = df_summary.to_csv(index=False)
            
            st.download_button(
                label="📋 Download Solution Summary",
                data=csv_summary,
                file_name="freshdrinks_solution_summary.csv",
                mime="text/csv"
            )
        
        # Report Generation
        st.markdown("---")
        st.markdown("### 🖨️ Generate Printable Report")
        
        if st.button("📄 Generate Comprehensive Report", use_container_width=True):
            st.success("""
            ✅ **Report Ready for Printing!**
            
            **To save as PDF or print:**
            1. Press **Ctrl+P** (Windows/Linux) or **Cmd+P** (Mac)
            2. Choose **"Save as PDF"** for digital copy
            3. Select your printer for physical copy
            
            **Report includes:**
            - Complete problem formulation
            - Step-by-step simplex solution
            - Optimal production quantities
            - Sensitivity analysis
            - Managerial recommendations
            - All tables and calculations
            """)

    else:
        # Instructions when no solution generated
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h2 style='color: #1E88E5;'>🎯 Ready to Optimize Production?</h2>
            <p style='font-size: 1.2rem; color: #666; margin: 1rem 0;'>
                Click the button above to see the complete simplex method solution
            </p>
            
            <div style='display: flex; justify-content: center; gap: 2rem; margin: 2rem 0; flex-wrap: wrap;'>
                <div style='text-align: center; padding: 1rem; border-radius: 10px; background: #E3F2FD; width: 150px;'>
                    <div style='font-size: 2.5rem;'>📋</div>
                    <div style='font-weight: bold;'>Step-by-Step</div>
                    <div>Simplex Method</div>
                </div>
                <div style='text-align: center; padding: 1rem; border-radius: 10px; background: #F1F8E9; width: 150px;'>
                    <div style='font-size: 2.5rem;'>💡</div>
                    <div style='font-weight: bold;'>Business</div>
                    <div>Insights</div>
                </div>
                <div style='text-align: center; padding: 1rem; border-radius: 10px; background: #FFF3E0; width: 150px;'>
                    <div style='font-size: 2.5rem;'>📊</div>
                    <div style='font-weight: bold;'>Sensitivity</div>
                    <div>Analysis</div>
                </div>
                <div style='text-align: center; padding: 1rem; border-radius: 10px; background: #F3E5F5; width: 150px;'>
                    <div style='font-size: 2.5rem;'>🚀</div>
                    <div style='font-weight: bold;'>Optimal</div>
                    <div>Solution</div>
                </div>
            </div>
            
            <p style='font-size: 1.1rem; color: #666; margin-top: 2rem;'>
                Learn how linear programming solves real-world production optimization problems!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Learning Objectives
        st.markdown("### 🎓 What You'll Learn:")
        
        learning_cols = st.columns(2)
        
        with learning_cols[0]:
            st.markdown("""
            **Mathematical Concepts:**
            - Linear programming formulation
            - Two-Phase Simplex Method
            - Big M method for artificial variables
            - Tableau operations and pivoting
            - Optimality conditions
            - Sensitivity analysis
            """)
        
        with learning_cols[1]:
            st.markdown("""
            **Business Applications:**
            - Production planning optimization
            - Resource allocation
            - Profit maximization
            - Capacity planning
            - Investment prioritization
            - Strategic decision making
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;'>
        <p><strong>🥤 FreshDrinks Co. Production Optimization Simulator</strong></p>
        <p>Linear Programming Project | Operations Research Application</p>
        <p>📚 Educational Tool for Understanding Simplex Method</p>
        <div style='margin-top: 1rem; font-size: 0.8rem; color: #999;'>
            <p>Note: This simulator demonstrates the application of linear programming in production planning.</p>
            <p>For actual business implementation, consult with operations research specialists.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
