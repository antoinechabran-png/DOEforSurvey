import streamlit as st
import pandas as pd
import numpy as np
import random
import io

# --- Utility Functions ---

def generate_random_codes(n):
    """Generates random codes like B53, C12."""
    codes = []
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ" # Removed I and O to avoid confusion
    while len(codes) < n:
        code = f"{random.choice(letters)}{random.randint(10, 99)}"
        if code not in codes:
            codes.append(code)
    return codes

def to_excel(df):
    """Converts dataframe to an Excel file in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='RotationPlan')
    return output.getvalue()

# --- Design Logics ---

def generate_complete_block(num_assessors, products):
    """Each assessor sees every product in a randomized order."""
    data = []
    for i in range(1, num_assessors + 1):
        row = [f"Assessor {i}"] + random.sample(products, len(products))
        data.append(row)
    
    cols = ["Assessor"] + [f"Rank {j+1}" for j in range(len(products))]
    return pd.DataFrame(data, columns=cols)

def generate_ibd(total_products, products_per_assessor, target_assessors_per_prod, products):
    """
    Generates an Incomplete Block Design.
    Ensures each product is evaluated a specific number of times.
    """
    # Total evaluations needed = (Target per product * Total products)
    # Total assessors = Total evaluations / products per assessor
    total_evals = target_assessors_per_prod * total_products
    num_assessors = int(np.ceil(total_evals / products_per_assessor))
    
    # Create a pool where each product appears 'target' times
    pool = products * target_assessors_per_prod
    random.shuffle(pool)
    
    # Pad pool if it doesn't divide perfectly by products_per_assessor
    while len(pool) < (num_assessors * products_per_assessor):
        pool.append(random.choice(products))
        
    data = []
    for i in range(num_assessors):
        start = i * products_per_assessor
        end = start + products_per_assessor
        row = [f"Assessor {i+1}"] + pool[start:end]
        data.append(row)
        
    cols = ["Assessor"] + [f"Rank {j+1}" for j in range(products_per_assessor)]
    return pd.DataFrame(data, columns=cols)

def generate_triangular(num_assessors, products):
    """Generates AB-A or BA-B style triads for all product pairs."""
    import itertools
    pairs = list(itertools.combinations(products, 2))
    data = []
    
    # For simplicity, we cycle through pairs and create a triad
    for i in range(1, num_assessors + 1):
        p1, p2 = random.choice(pairs)
        # Randomly choose if it's AAB or BBA
        triad = [p1, p1, p2] if random.random() > 0.5 else [p2, p2, p1]
        random.shuffle(triad)
        data.append([f"Assessor {i}"] + triad)
        
    cols = ["Assessor", "Sample 1", "Sample 2", "Sample 3"]
    return pd.DataFrame(data, columns=cols)

# --- Streamlit UI ---

st.set_page_config(page_title="Sensory Rotation Planner", layout="wide")
st.title("🧪 Sensory Design Generator")

st.sidebar.header("Design Settings")
design_type = st.sidebar.selectbox(
    "Select Design Type", 
    ["Complete Block Design", "Incomplete Block Design", "Triangular Design"]
)

# 1. Product Inputs
st.subheader("1. Product Configuration")
input_method = st.radio("Product Code Method", ["Random Generation", "Manual Entry / Excel Upload"])

product_list = []
if design_type == "Incomplete Block Design":
    num_prods = st.number_input("Total number of products", min_value=2, value=6)
else:
    num_prods = st.number_input("Number of products", min_value=2, value=3)

if input_method == "Random Generation":
    product_list = generate_random_codes(num_prods)
    st.info(f"Generated Codes: {', '.join(product_list)}")
else:
    col1, col2 = st.columns(2)
    with col1:
        manual_input = st.text_area("Enter codes (comma separated)", "P1, P2, P3")
    with col2:
        uploaded_file = st.file_uploader("Or upload Excel (Col A)", type=["xlsx"])
    
    if uploaded_file:
        df_upload = pd.read_excel(uploaded_file)
        product_list = df_upload.iloc[:, 0].dropna().astype(str).tolist()
    else:
        product_list = [x.strip() for x in manual_input.split(",") if x.strip()]
    
    if len(product_list) != num_prods:
        st.warning(f"Expected {num_prods} products, but found {len(product_list)} codes.")

# 2. Design Specific Inputs
st.subheader("2. Design Parameters")
df_result = pd.DataFrame()

if design_type == "Complete Block Design":
    n_assessors = st.number_input("Number of assessors", min_value=1, value=10)
    if st.button("Generate Plan"):
        df_result = generate_complete_block(n_assessors, product_list)

elif design_type == "Incomplete Block Design":
    k = st.number_input("Number of products tested per assessor", min_value=1, max_value=num_prods-1, value=3)
    r = st.number_input("Desired assessors per product", min_value=1, value=5)
    if st.button("Generate Plan"):
        df_result = generate_ibd(num_prods, k, r, product_list)

elif design_type == "Triangular Design":
    n_assessors = st.number_input("Number of assessors", min_value=1, value=10)
    if st.button("Generate Plan"):
        df_result = generate_triangular(n_assessors, product_list)

# 3. Results and Export
if not df_result.empty:
    st.success("Rotation plan generated successfully!")
    st.dataframe(df_result)
    
    excel_data = to_excel(df_result)
    st.download_button(
        label="📥 Download Excel Plan",
        data=excel_data,
        file_name=f"rotation_plan_{design_type.lower().replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
