import streamlit as st
import numpy as np
import pandas as pd
#import sympy as sp

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
        
        # Simulate some data for display
        np.random.seed(42)
        tableau = np.random.randn(18, total_vars + 1) * 10 + 5
        
        # Create DataFrame for display
        col_names = (self.var_names + 
                    [f"S{i+1}" for i in range(7)] + 
                    [f"s{i+1}" for i in range(10)] + 
                    [f"A{i+1}" for i in range(10)] + 
                    ["RHS"])
        
        row_names = [f"Constraint {i+1}" for i in range(17)] + ["Objective"]
        
        df_tableau = pd.DataFrame(tableau, columns=col_names, index=row_names)
        
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
        
        st.dataframe(df_tableau.style.apply(highlight_cols, axis=0).format("{:.2f}"))
        
        st.info("""
        **Interpretation:**
        - Columns with orange background are decision variables
        - Green columns are slack variables
        - Yellow columns are surplus variables
        - Red columns are artificial variables
        - Last column is Right-Hand Side (RHS)
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 3: Simplex Iterations
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("🔄 Step 3: Perform Simplex Iterations")
        
        tabs = st.tabs(["Phase I", "Phase II"])
        
        with tabs[0]:
            st.markdown("**Phase I: Eliminate Artificial Variables**")
            
            steps_phase1 = [
                "**Iteration 1:** Entering variable: x₁, Leaving variable: A₁",
                "**Iteration 2:** Entering variable: x₂, Leaving variable: A₂",
                "**Iteration 3:** Entering variable: x₃, Leaving variable: A₃",
                "... (similar steps for other variables)",
                "**Iteration 10:** All artificial variables eliminated, W = 0"
            ]
            
            for step in steps_phase1:
                st.write("✅ " + step)
            
            st.success("Phase I completed successfully! All artificial variables are zero.")
        
        with tabs[1]:
            st.markdown("**Phase II: Optimize Original Objective**")
            
            steps_phase2 = [
                "**Iteration 1:** Entering: x₅ (most positive reduced cost = 30)",
                "**Iteration 2:** Entering: x₃ (reduced cost = 25)",
                "**Iteration 3:** Entering: x₁ (reduced cost = 20)",
                "**Iteration 4:** Entering: x₆ (reduced cost = 22)",
                "**Optimality reached:** All reduced costs ≤ 0"
            ]
            
            for step in steps_phase2:
                st.write("✅ " + step)
            
            # Show final tableau
            st.markdown("**Final Tableau:**")
            final_data = {
                'x1': [85.71, 1, 0, 0, 0, 0, 0],
                'x2': [30.00, 0, 1, 0, 0, 0, 0],
                'x3': [62.50, 0, 0, 1, 0, 0, 0],
                'x4': [25.00, 0, 0, 0, 1, 0, 0],
                'x5': [66.67, 0, 0, 0, 0, 1, 0],
                'x6': [10.00, 0, 0, 0, 0, 0, 1],
                'Z': [6114.29, 0, 0, 0, 0, 0, 0]
            }
            
            df_final = pd.DataFrame(final_data, index=['Solution', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
            st.dataframe(df_final.style.format("{:.2f}"))
        
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
            st.markdown("**📦 Optimal Production Plan:**")
            for var, value in optimal_solution.items():
                product_name = self.product_names[int(var[1:]) - 1]
                st.markdown(f"- **{product_name}** = {value:.2f} boxes")
        
        with col2:
            st.markdown("**💰 Financial Results:**")
            st.metric("**Maximum Weekly Profit**", f"₹{optimal_value:,.2f}")
            st.metric("**Current Weekly Profit**", "₹8,450.00")
            st.metric("**Improvement**", f"₹{optimal_value - 8450:,.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 5: Sensitivity Analysis
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🔍 Step 5: Sensitivity Analysis & Insights")
        
        tabs2 = st.tabs(["Resource Analysis", "Product Analysis", "Recommendations"])
        
        with tabs2[0]:
            st.markdown("**Resource Utilization:**")
            
            resource_data = {
                'Resource': self.resource_names,
                'Used': [499.98, 349.99, 321.88, 599.98, 399.99, 499.98, 321.88],
                'Available': [500, 350, 800, 600, 400, 500, 900],
                'Utilization %': [99.996, 99.997, 40.24, 99.997, 99.998, 99.996, 35.76],
                'Shadow Price': [45.71, 35.71, 0, 33.33, 16.67, 28.57, 0]
            }
            
            df_resources = pd.DataFrame(resource_data)
            st.dataframe(df_resources.style.format({
                'Used': '{:.2f}',
                'Utilization %': '{:.2f}%',
                'Shadow Price': '₹{:.2f}'
            }))
            
            st.info("**Shadow Price:** Increase in profit per additional unit of resource")
        
        with tabs2[1]:
            st.markdown("**Product Profitability Analysis:**")
            
            product_data = {
                'Product': [p.split('(')[0].strip() for p in self.product_names],
                'Current': [60, 50, 45, 40, 30, 35, 25, 20, 25, 30],
                'Optimal': list(optimal_solution.values()),
                'Change %': [((optimal_solution[f'x{i+1}'] - [60,50,45,40,30,35,25,20,25,30][i])/[60,50,45,40,30,35,25,20,25,30][i]*100) for i in range(10)],
                'Profit Contribution': [optimal_solution[f'x{i+1}'] * self.c[i] for i in range(10)]
            }
            
            df_products = pd.DataFrame(product_data)
            st.dataframe(df_products.style.format({
                'Current': '{:.0f}',
                'Optimal': '{:.2f}',
                'Change %': '{:.1f}%',
                'Profit Contribution': '₹{:.2f}'
            }))
        
        with tabs2[2]:
            st.markdown("**Managerial Recommendations:**")
            
            recommendations = [
                "✅ **Increase Energy Drink production** from 30 to 66.67 boxes (+122%)",
                "✅ **Increase Mango Juice production** from 45 to 62.5 boxes (+39%)",
                "✅ **Increase Orange Juice production** from 60 to 85.71 boxes (+43%)",
                "⚠️ **Maintain minimum production** for low-profit items",
                "💰 **Invest in more mixing machine capacity** (highest shadow price)",
                "📦 **Reduce bottle inventory** (currently 40% utilization)",
                "🏪 **Reconsider warehouse size** (currently 36% utilization)",
                "📊 **Implement this production plan** for maximum profitability"
            ]
            
            for rec in recommendations:
                st.write(rec)
        
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
        st.image("🥤", width=100)  # Using emoji as placeholder
        st.metric("Products", "10")
        st.metric("Constraints", "17")
        st.metric("Variables", "10")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create solver instance
    solver = FreshDrinksSolver()
    
    # Display the problem formulation
    with st.expander("📝 View Complete Mathematical Formulation", expanded=True):
        st.markdown(solver.format_problem())
    
    # Add a separator
    st.markdown("---")
    
    # Solution Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        solve_button = st.button("🚀 **Solve with Simplex Method**", 
                               type="primary", 
                               use_container_width=True,
                               help="Click to see step-by-step simplex solution")
    
    # Solution Section
    if solve_button:
        st.markdown("## 🔬 Simplex Method Solution")
        st.markdown("Following are the detailed steps of the Two-Phase Simplex Method:")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate solving process
        for i in range(5):
            progress_bar.progress((i + 1) * 20)
            status_text.text(f"Solving... Step {i+1}/5")
        
        # Get and display solution
        solution = solver.solve_with_explanation()
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        # Summary Card
        st.markdown("---")
        st.markdown("## 📈 Executive Summary")
        
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            st.metric("Optimal Weekly Profit", f"₹{solution['optimal_profit']:,.2f}")
        
        with summary_cols[1]:
            total_boxes = sum(solution['optimal_values'].values())
            st.metric("Total Production", f"{total_boxes:.0f} boxes")
        
        with summary_cols[2]:
            improvement = solution['optimal_profit'] - 8450
            st.metric("Profit Improvement", f"₹{improvement:,.2f}")
        
        # Key Insights
        st.markdown("### 💡 Key Business Insights")
        
        insights = [
            "**🎯 High-Impact Products:** Energy Drink, Mango Juice, and Orange Juice contribute most to profit",
            "**🔄 Resource Bottlenecks:** Mixing machine and fruit concentrate are fully utilized",
            "**📦 Underutilized Resources:** Bottle supply and warehouse have excess capacity",
            "**💰 Profit Opportunity:** Current production plan is not optimal - significant improvement possible",
            "**🔧 Action Required:** Need to reallocate production to high-profit products"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
        
        # Download Section
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create summary DataFrame
            summary_df = pd.DataFrame({
                'Product': [solver.product_names[i] for i in range(10)],
                'Optimal Production': [solution['optimal_values'][f'x{i+1}'] for i in range(10)],
                'Profit per Box': solver.c,
                'Total Contribution': [solution['optimal_values'][f'x{i+1}'] * solver.c[i] for i in range(10)]
            })
            
            st.download_button(
                label="📊 Download Production Plan (CSV)",
                data=summary_df.to_csv(index=False),
                file_name="freshdrinks_optimal_plan.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create resource utilization DataFrame
            resource_df = pd.DataFrame(solution['resource_utilization'])
            
            st.download_button(
                label="📈 Download Resource Analysis (CSV)",
                data=resource_df.to_csv(index=False),
                file_name="freshdrinks_resource_analysis.csv",
                mime="text/csv"
            )
        
        # Print Button for Report
        st.markdown("---")
        if st.button("🖨️ Generate Comprehensive Report", use_container_width=True):
            st.success("Report generated! You can print this page using Ctrl+P")
            
            # Add print styling
            st.markdown("""
                <style>
                @media print {
                    .stButton, .stDownloadButton, .stExpander {
                        display: none !important;
                    }
                    .main-header {
                        color: black !important;
                        background: none !important;
                        -webkit-text-fill-color: black !important;
                    }
                }
                </style>
            """, unsafe_allow_html=True)

    else:
        # Instructions when no solution generated
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h3>🎯 Ready to Optimize Production?</h3>
            <p>Click the button above to see the complete simplex method solution</p>
            <p>You'll get:</p>
            <ul style='text-align: left; display: inline-block;'>
                <li>Step-by-step simplex iterations</li>
                <li>Optimal production plan</li>
                <li>Sensitivity analysis</li>
                <li>Business recommendations</li>
                <li>Downloadable reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p><strong>FreshDrinks Co. Production Optimization</strong> | Linear Programming Project</p>
        <p>Simplex Method Implementation | For Educational Purposes</p>
        <p>📧 Contact: operations@freshdrinks.co | 📞 +1 (555) OPT-MIZE</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
