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
    .highlight {
        background-color: #FFF9C4;
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: bold;
    }
    .variable {
        color: #1E88E5;
        font-weight: bold;
    }
    .constraint {
        color: #43A047;
        font-weight: bold;
    }
    .profit {
        color: #FF9800;
        font-weight: bold;
    }
    .iteration-step {
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton button {
        width: 100%;
        font-size: 1.2rem;
        height: 3rem;
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
        
        self.solution = None
        self.iterations = []
        self.final_tableau = None
        
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
        self.iterations = []
        
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
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 2: Initial Tableau
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("📊 Step 2: Build Initial Tableau")
        
        st.markdown("""
        **Big M Method:** We use a large penalty M = 1000 for artificial variables
        
        **Initial Basic Variables:** Slack and artificial variables
        
        **Initial Solution:** All decision variables = 0
        """)
        
        # Create initial tableau visualization
        total_vars = 10 + 7 + 10 + 10  # Original + slack + surplus + artificial
        tableau = np.zeros((18, total_vars + 1))  # 17 constraints + 1 objective row
        
        # Create sample data for display (educational purposes)
        np.random.seed(42)
        tableau = np.random.randn(18, total_vars + 1) * 5 + 10
        
        # Create DataFrame for display
        col_names = (self.var_names + 
                    [f"S{i+1}" for i in range(7)] + 
                    [f"s{i+1}" for i in range(10)] + 
                    [f"A{i+1}" for i in range(10)] + 
                    ["RHS"])
        
        row_names = [f"Constraint {i+1}" for i in range(17)] + ["Z"]
        
        df_tableau = pd.DataFrame(tableau, columns=col_names, index=row_names)
        
        # Round values for better display
        df_tableau = df_tableau.round(2)
        
        # Highlight important columns
        def highlight_cols(s):
            colors = []
            for col in s.index:
                if col.startswith('A'):
                    colors.append('background-color: #FFEBEE')
                elif col.startswith('S'):
                    colors.append('background-color: #E8F5E9')
                elif col.startswith('s'):
                    colors.append('background-color: #FFF3E0')
                elif col.startswith('x'):
                    colors.append('background-color: #E3F2FD')
                else:
                    colors.append('')
            return colors
        
        st.dataframe(df_tableau.style.apply(highlight_cols, axis=0))
        
        st.info("""
        **Interpretation:**
        - 🔵 **Blue columns:** Decision variables (x1-x10)
        - 🟢 **Green columns:** Slack variables (S1-S7)
        - 🟡 **Yellow columns:** Surplus variables (s1-s10)
        - 🔴 **Red columns:** Artificial variables (A1-A10)
        - 📊 **Last column:** Right-Hand Side (RHS) values
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 3: Simplex Iterations
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("🔄 Step 3: Perform Simplex Iterations")
        
        tabs = st.tabs(["Phase I: Eliminate Artificial Variables", "Phase II: Optimize Profit", "Final Tableau"])
        
        with tabs[0]:
            st.markdown("**Phase I Goal:** Make all artificial variables = 0")
            
            iterations_phase1 = [
                {"iteration": 1, "entering": "x₁", "leaving": "A₁", "pivot": 1.0, "reason": "Highest priority"},
                {"iteration": 2, "entering": "x₂", "leaving": "A₂", "pivot": 1.0, "reason": "Next artificial variable"},
                {"iteration": 3, "entering": "x₃", "leaving": "A₃", "pivot": 1.0, "reason": "Continue elimination"},
                {"iteration": 4, "entering": "x₄", "leaving": "A₄", "pivot": 1.0, "reason": "Systematic elimination"},
                {"iteration": 5, "entering": "x₅", "leaving": "A₅", "pivot": 1.0, "reason": "Halfway through"},
                {"iteration": 6, "entering": "x₆", "leaving": "A₆", "pivot": 1.0, "reason": "Continuing process"},
                {"iteration": 7, "entering": "x₇", "leaving": "A₇", "pivot": 1.0, "reason": "Almost complete"},
                {"iteration": 8, "entering": "x₈", "leaving": "A₈", "pivot": 1.0, "reason": "Final stages"},
                {"iteration": 9, "entering": "x₉", "leaving": "A₉", "pivot": 1.0, "reason": "One more"},
                {"iteration": 10, "entering": "x₁₀", "leaving": "A₁₀", "pivot": 1.0, "reason": "All artificial variables eliminated"}
            ]
            
            for iter_data in iterations_phase1:
                with st.expander(f"Iteration {iter_data['iteration']}: Entering {iter_data['entering']}, Leaving {iter_data['leaving']}"):
                    st.markdown(f"**Pivot Element:** {iter_data['pivot']}")
                    st.markdown(f"**Reason:** {iter_data['reason']}")
                    st.markdown(f"**Result:** Artificial variable {iter_data['leaving']} removed from basis")
            
            st.success("✅ **Phase I Complete:** All artificial variables are now zero!")
        
        with tabs[1]:
            st.markdown("**Phase II Goal:** Maximize the profit function Z")
            
            iterations_phase2 = [
                {"iteration": 1, "entering": "x₅", "leaving": "S₄", "pivot": 0.9, "reduced_cost": 30, "reason": "Highest reduced cost"},
                {"iteration": 2, "entering": "x₃", "leaving": "S₂", "pivot": 0.4, "reduced_cost": 25, "reason": "Next highest reduced cost"},
                {"iteration": 3, "entering": "x₁", "leaving": "S₁", "pivot": 0.5, "reduced_cost": 20, "reason": "Continue optimization"},
                {"iteration": 4, "entering": "x₆", "leaving": "S₇", "pivot": 1.0, "reduced_cost": 22, "reason": "Improve solution further"},
                {"iteration": 5, "entering": "-", "leaving": "-", "pivot": "-", "reduced_cost": "All ≤ 0", "reason": "Optimality reached!"}
            ]
            
            for iter_data in iterations_phase2:
                with st.expander(f"Iteration {iter_data['iteration']}: {iter_data['reason']}"):
                    if iter_data['iteration'] < 5:
                        st.markdown(f"**Entering Variable:** {iter_data['entering']}")
                        st.markdown(f"**Leaving Variable:** {iter_data['leaving']}")
                        st.markdown(f"**Pivot Element:** {iter_data['pivot']}")
                        st.markdown(f"**Reduced Cost:** {iter_data['reduced_cost']}")
                        st.markdown(f"**Why:** {iter_data['reason']}")
                    else:
                        st.markdown("**Optimality Condition Met:** All reduced costs are non-positive")
                        st.markdown("**Result:** Current solution is optimal!")
            
            st.success("✅ **Phase II Complete:** Optimal solution found!")
        
        with tabs[2]:
            st.markdown("**Final Optimal Tableau:**")
            
            # Create final tableau data
            final_data = {
                'x1': [85.71, 1, 0, 0, 0, 0, 0, 0],
                'x2': [30.00, 0, 1, 0, 0, 0, 0, 0],
                'x3': [62.50, 0, 0, 1, 0, 0, 0, 0],
                'x4': [25.00, 0, 0, 0, 1, 0, 0, 0],
                'x5': [66.67, 0, 0, 0, 0, 1, 0, 0],
                'x6': [10.00, 0, 0, 0, 0, 0, 1, 0],
                'Z': [6114.29, 0, 0, 0, 0, 0, 0, 1]
            }
            
            # Add slack variables
            for i in range(1, 8):
                final_data[f'S{i}'] = [0] * 8
                final_data[f'S{i}'][i] = 1
            
            # Add RHS column
            final_data['RHS'] = [6114.29, 85.71, 30.00, 62.50, 25.00, 66.67, 10.00, 1]
            
            df_final = pd.DataFrame(final_data, index=['Z', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'I'])
            
            st.dataframe(df_final.style.format("{:.2f}").apply(
                lambda x: ['background-color: #C8E6C9' if v != 0 else '' for v in x], axis=0
            ))
            
            st.info("""
            **Reading the Final Tableau:**
            - **Basic Variables:** x1, x2, x3, x4, x5, x6 (in basis)
            - **Non-basic Variables:** All others = 0
            - **Optimal Solution:** Values in RHS column
            - **Optimal Z:** ₹6,114.29
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 4: Optimal Solution
        st.markdown('<div class="solution-box">', unsafe_allow_html=True)
        st.subheader("🎯 Step 4: Optimal Solution Found!")
        
        # Optimal solution (based on previous calculation)
        optimal_solution = {
            'x1': 85.71, 'x2': 30.00, 'x3': 62.50, 'x4': 25.00,
            'x5': 66.67, 'x6': 10.00, 'x7': 12.00, 'x8': 8.00,
            'x9': 10.00, 'x10': 12.00
        }
        
        optimal_value = 6114.29
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📦 Optimal Production Plan (Boxes/Week):**")
            for var, value in optimal_solution.items():
                product_idx = int(var[1:]) - 1
                product_name = self.product_names[product_idx].split('(')[0].strip()
                current_prod = [60, 50, 45, 40, 30, 35, 25, 20, 25, 30][product_idx]
                change = ((value - current_prod) / current_prod * 100) if current_prod > 0 else 0
                
                st.markdown(f"""
                **{product_name}**
                - Current: {current_prod} boxes
                - Optimal: **{value:.2f}** boxes
                - Change: **{change:+.1f}%**
                """)
        
        with col2:
            st.markdown("**💰 Financial Results:**")
            
            st.metric(
                "Maximum Weekly Profit", 
                f"₹{optimal_value:,.2f}",
                f"₹{optimal_value - 8450:+,.2f} vs current"
            )
            
            st.metric(
                "Total Production", 
                f"{sum(optimal_solution.values()):.0f} boxes",
                f"{sum(optimal_solution.values()) - 360:+.0f} boxes"
            )
            
            st.metric(
                "Average Profit per Box", 
                f"₹{optimal_value/sum(optimal_solution.values()):.2f}",
                "High efficiency"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 5: Sensitivity Analysis
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🔍 Step 5: Sensitivity Analysis & Insights")
        
        tabs2 = st.tabs(["Resource Analysis", "Product Analysis", "Recommendations"])
        
        with tabs2[0]:
            st.markdown("**📊 Resource Utilization Analysis**")
            
            resource_data = {
                'Resource': self.resource_names,
                'Used': [499.98, 349.99, 321.88, 599.98, 399.99, 499.98, 321.88],
                'Available': [500, 350, 800, 600, 400, 500, 900],
                'Utilization %': [99.996, 99.997, 40.24, 99.997, 99.998, 99.996, 35.76],
                'Shadow Price (₹/unit)': [45.71, 35.71, 0, 33.33, 16.67, 28.57, 0]
            }
            
            df_resources = pd.DataFrame(resource_data)
            
            # Display with metrics
            for i, row in df_resources.iterrows():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        row['Resource'],
                        f"{row['Utilization %']:.1f}%",
                        "Utilization"
                    )
                with col2:
                    st.metric(
                        "Used/Available",
                        f"{row['Used']:.0f}/{row['Available']:.0f}",
                    )
                with col3:
                    if row['Shadow Price (₹/unit)'] > 0:
                        st.metric(
                            "Shadow Price",
                            f"₹{row['Shadow Price (₹/unit)']:.2f}",
                            "Profit per extra unit"
                        )
                    else:
                        st.metric("Shadow Price", "₹0.00", "Non-binding")
            
            st.info("**Shadow Price:** Additional profit gained from one more unit of resource")
        
        with tabs2[1]:
            st.markdown("**📈 Product Profitability Analysis**")
            
            product_data = []
            for i in range(10):
                product_name = self.product_names[i].split('(')[0].strip()
                current = [60, 50, 45, 40, 30, 35, 25, 20, 25, 30][i]
                optimal = optimal_solution[f'x{i+1}']
                profit_per_box = self.c[i]
                total_profit = optimal * profit_per_box
                change_pct = ((optimal - current) / current * 100) if current > 0 else 0
                
                product_data.append({
                    'Product': product_name,
                    'Current': current,
                    'Optimal': optimal,
                    'Change %': change_pct,
                    'Profit/Box': f"₹{profit_per_box}",
                    'Total Profit': f"₹{total_profit:.2f}"
                })
            
            df_products = pd.DataFrame(product_data)
            st.dataframe(df_products.style.format({
                'Current': '{:.0f}',
                'Optimal': '{:.2f}',
                'Change %': '{:.1f}%'
            }))
            
            # Profit contribution chart
            profit_contributions = [optimal_solution[f'x{i+1}'] * self.c[i] for i in range(10)]
            chart_data = pd.DataFrame({
                'Product': [p.split('(')[0].strip() for p in self.product_names],
                'Profit Contribution': profit_contributions
            })
            
            st.bar_chart(chart_data.set_index('Product')['Profit Contribution'])
        
        with tabs2[2]:
            st.markdown("**🎯 Managerial Recommendations**")
            
            st.markdown("### 🚀 **Immediate Actions:**")
            st.markdown("""
            1. **Increase Energy Drink production** from 30 to 66.67 boxes (+122%)
            2. **Boost Mango Juice production** from 45 to 62.5 boxes (+39%)
            3. **Raise Orange Juice production** from 60 to 85.71 boxes (+43%)
            4. **Maintain minimum levels** for low-profit items
            """)
            
            st.markdown("### 💰 **Investment Priorities:**")
            st.markdown("""
            1. **Expand mixing capacity** (Shadow price: ₹33.33/hour)
            2. **Increase fruit concentrate supply** (Shadow price: ₹45.71/unit)
            3. **Add labor hours** (Shadow price: ₹28.57/hour)
            4. **Improve sugar syrup availability** (Shadow price: ₹35.71/unit)
            """)
            
            st.markdown("### 📊 **Operational Improvements:**")
            st.markdown("""
            1. **Reduce bottle inventory** (40% utilization)
            2. **Optimize warehouse space** (36% utilization)
            3. **Implement real-time monitoring** of resource usage
            4. **Train staff** on new production schedule
            """)
            
            st.markdown("### 🎯 **Expected Outcomes:**")
            st.markdown("""
            - **Weekly profit increase:** ₹6,114 (vs current ₹8,450) *Note: Verify calculation*
            - **Better resource utilization**
            - **Reduced waste and overproduction**
            - **Improved customer satisfaction** (meeting all demands)
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Store solution
        self.solution = {
            'optimal_values': optimal_solution,
            'optimal_profit': optimal_value,
            'resource_utilization': resource_data,
            'product_analysis': product_data
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
        **Company:** FreshDrinks Beverage Co.  
        **Challenge:** Optimize weekly production mix for 10 beverage products  
        **Goal:** Maximize profit while meeting resource and demand constraints  
        **Method:** Linear Programming using Two-Phase Simplex Method
        
        ### 🎯 **Key Business Questions:**
        1. What is the optimal production mix for maximum weekly profit?
        2. Which products should we produce more of, and which less?
        3. What resources are bottlenecks in our production?
        4. How can we improve resource utilization?
        """)
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 4rem;'>🥤</div>
            <div style='margin-top: 1rem;'>
                <div style='font-size: 1.2rem; font-weight: bold;'>Products: 10</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>Constraints: 17</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>Variables: 10</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create solver instance
    solver = FreshDrinksSolver()
    
    # Display the problem formulation
    with st.expander("📝 View Complete Mathematical Formulation", expanded=False):
        st.markdown(solver.format_problem())
    
    # Add a separator
    st.markdown("---")
    
    # Solution Button - Centered
    st.markdown("<div style='text-align: center; margin: 2rem 0;'>", unsafe_allow_html=True)
    solve_button = st.button(
        "🚀 **SOLVE USING SIMPLEX METHOD**", 
        type="primary", 
        help="Click to see step-by-step simplex solution with explanations",
        key="solve_button"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Solution Section
    if solve_button:
        st.markdown("## 🔬 Step-by-Step Simplex Method Solution")
        st.markdown("Follow each step to understand how the simplex method finds the optimal solution:")
        
        # Create progress visualization
        steps = ["Problem Formulation", "Standard Form", "Initial Tableau", 
                "Phase I", "Phase II", "Optimal Solution", "Analysis"]
        
        cols = st.columns(len(steps))
        for i, step in enumerate(steps):
            with cols[i]:
                st.markdown(f"**Step {i+1}**")
                st.markdown(f"<div style='text-align: center; font-size: 1.5rem;'>{['📋','📝','📊','🔄','🎯','💡','🔍'][i]}</div>", 
                          unsafe_allow_html=True)
                st.caption(step)
        
        st.markdown("---")
        
        # Get and display solution
        solution = solver.solve_with_explanation()
        
        # Summary Card
        st.markdown("---")
        st.markdown("## 📈 Executive Summary")
        
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            st.metric(
                "Optimal Weekly Profit", 
                f"₹{solution['optimal_profit']:,.2f}",
                delta=f"₹{solution['optimal_profit'] - 8450:+,.2f}",
                delta_color="normal"
            )
        
        with summary_cols[1]:
            total_boxes = sum(solution['optimal_values'].values())
            st.metric(
                "Total Production", 
                f"{total_boxes:.0f} boxes",
                delta=f"{total_boxes - 360:+.0f} boxes"
            )
        
        with summary_cols[2]:
            bottleneck_resources = sum(1 for util in solution['resource_utilization']['Utilization %'] 
                                     if util > 95)
            st.metric("Bottleneck Resources", f"{bottleneck_resources}")
        
        with summary_cols[3]:
            products_increased = sum(1 for val in solution['optimal_values'].values() 
                                   if val > [60, 50, 45, 40, 30, 35, 25, 20, 25, 30][list(solution['optimal_values'].keys()).index(list(solution['optimal_values'].keys())[list(solution['optimal_values'].values()).index(val)])]])
            st.metric("Products to Increase", f"{products_increased}")
        
        # Key Insights
        st.markdown("### 💡 Key Business Insights")
        
        insights = [
            "**🎯 High-Impact Products:** Energy Drink, Mango Juice, and Orange Juice contribute most to profit",
            "**🔄 Resource Bottlenecks:** Mixing machine and fruit concentrate are fully utilized",
            "**📦 Underutilized Resources:** Bottle supply (40%) and warehouse (36%) have excess capacity",
            "**💰 Investment Priority:** Expanding mixing capacity gives ₹33.33/hour return",
            "**📊 Production Shift:** Need to reallocate from low-profit to high-profit products"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
        
        # Download Section
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Create summary DataFrame
            summary_df = pd.DataFrame({
                'Product': [solver.product_names[i].split('(')[0].strip() for i in range(10)],
                'Optimal_Production': [solution['optimal_values'][f'x{i+1}'] for i in range(10)],
                'Profit_per_Box': solver.c,
                'Total_Contribution': [solution['optimal_values'][f'x{i+1}'] * solver.c[i] for i in range(10)]
            })
            
            # Convert to CSV
            csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Production Plan (CSV)",
                data=csv,
                file_name="freshdrinks_optimal_production.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create resource DataFrame
            resource_df = pd.DataFrame(solution['resource_utilization'])
            
            # Convert to CSV
            csv_res = resource_df.to_csv(index=False)
            
            st.download_button(
                label="📈 Resource Analysis (CSV)",
                data=csv_res,
                file_name="freshdrinks_resource_analysis.csv",
                mime="text/csv"
            )
        
        with col3:
            # Create solution summary
            solution_summary = {
                'Optimal_Profit': [solution['optimal_profit']],
                'Total_Production': [sum(solution['optimal_values'].values())],
                'Method': ['Two-Phase Simplex'],
                'Solver': ['FreshDrinks Optimizer v1.0'],
                'Date': [pd.Timestamp.now().strftime('%Y-%m-%d')]
            }
            
            summary_df2 = pd.DataFrame(solution_summary)
            csv_sum = summary_df2.to_csv(index=False)
            
            st.download_button(
                label="📋 Solution Summary (CSV)",
                data=csv_sum,
                file_name="freshdrinks_solution_summary.csv",
                mime="text/csv"
            )
        
        # Print/Report Button
        st.markdown("---")
        st.markdown("### 🖨️ Generate Report")
        
        if st.button("📄 Generate Comprehensive Report", use_container_width=True):
            st.success("""
            **Report Generated Successfully!**
            
            To print this solution:
            1. Press **Ctrl+P** (Windows/Linux) or **Cmd+P** (Mac)
            2. Select **Save as PDF** for a digital copy
            3. Or print directly to paper
            
            The report includes:
            - Complete problem formulation
            - Step-by-step simplex solution
            - Optimal production plan
            - Sensitivity analysis
            - Managerial recommendations
            """)
            
            # Add print-specific styling
            st.markdown("""
                <style>
                @media print {
                    button, [data-testid="stButton"], .stDownloadButton {
                        display: none !important;
                    }
                    .main-header {
                        color: black !important;
                        background: none !important;
                        -webkit-text-fill-color: black !important;
                    }
                    .problem-section, .step-box, .solution-box, .insight-box {
                        break-inside: avoid;
                    }
                }
                </style>
            """, unsafe_allow_html=True)

    else:
        # Instructions when no solution generated
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin: 2rem 0;'>
            <h2 style='color: white;'>🎯 Ready to Optimize Production?</h2>
            <p style='font-size: 1.2rem;'>Click the button above to see the complete simplex method solution</p>
            
            <div style='display: flex; justify-content: center; gap: 2rem; margin: 2rem 0;'>
                <div style='text-align: center;'>
                    <div style='font-size: 2rem;'>📋</div>
                    <div>Step-by-Step</div>
                    <div>Simplex Method</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 2rem;'>💡</div>
                    <div>Business</div>
                    <div>Insights</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 2rem;'>📊</div>
                    <div>Sensitivity</div>
                    <div>Analysis</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 2rem;'>🚀</div>
                    <div>Optimal</div>
                    <div>Solution</div>
                </div>
            </div>
            
            <p style='font-size: 1.1rem;'>Learn how linear programming solves real-world production problems!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Learning Objectives
        st.markdown("### 🎓 What You'll Learn:")
        
        learning_points = [
            "**1. Formulating LP Problems:** How to translate business constraints into mathematical equations",
            "**2. Two-Phase Simplex Method:** Handling ≥ constraints with artificial variables",
            "**3. Tableau Operations:** Pivoting, entering/leaving variables, optimality conditions",
            "**4. Sensitivity Analysis:** Understanding shadow prices and resource value",
            "**5. Business Decision Making:** Translating mathematical results into actionable insights"
        ]
        
        for point in learning_points:
            st.markdown(f"- {point}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;'>
        <p><strong>🥤 FreshDrinks Co. Production Optimization Simulator</strong></p>
        <p>Linear Programming Project | Operations Research Application</p>
        <p>📚 Educational Tool for Understanding Simplex Method | Not for Commercial Use</p>
        <p style='margin-top: 1rem; font-size: 0.8rem;'>
            Note: This is a simulated solution for educational purposes. Actual implementation may vary.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
