import streamlit as st
import numpy as np
# Import your classes from the /evolution folder
# from evolution.preferential_matroid import PreferentialMatroid
# from evolution.critical_evolution_matroid import CriticalEvolutionMatroid

st.title("Stochastic Matroid Evolution Lab")

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Global Parameters")
q = st.sidebar.selectbox("Field Order (q)", [2, 3, 5, 7])
d = st.sidebar.slider("Ambient Dimension (d)", 2, 20, 5)
model_type = st.sidebar.radio(
    "Select Evolution Model",
    ("Kelly-Oxley (Uniform)", "Altschuler-Hanany (Critical)", "Preferential Attachment")
)

# --- LATEX DOCUMENTATION ---
if model_type == "Kelly-Oxley (Uniform)":
    st.latex(r"P(r_{n+1} = k+1) = \frac{q^d - q^k}{q^d}")
    st.info("Focus: Uniform random distribution and global rank convergence.")
elif model_type == "Preferential Attachment":
    bias = st.sidebar.slider("Bias Level (p)", 0.0, 1.0, 0.8)
    st.latex(r"v_{new} \in \text{span}(B) \text{ with probability } p")
    st.info("Focus: Structural 'hubs' and slow rank growth.")

# --- SIMULATION CONTROL ---
if st.button("Run Simulation Step"):
    # Logic to initialize your model class and call .step()
    # You can use st.session_state to 'remember' the matroid between clicks
    st.write(f"Running {model_type}...")
    # [Table or Graph would go here]