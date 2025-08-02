import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Page title
st.set_page_config(page_title="🛍️ Customer Intelligence Tool", layout="wide")
st.title("🛍️ Customer Intelligence: Product Recommender + Segment Predictor")

# --- Create dummy data instead of loading files ---

@st.cache_data
def load_data():
    # Dummy product data
    data = pd.DataFrame({
        'product_name': ['Apple', 'Banana', 'Carrot', 'Dates', 'Eggplant', 'Fig', 'Grape']
    })
    return data

@st.cache_resource
def load_similarity():
    # Dummy similarity matrix (7x7)
    # Similarity with itself is 1, others random between 0.7-0.99
    sim = np.eye(7)
    for i in range(7):
        for j in range(i+1,7):
            val = np.random.uniform(0.7, 0.99)
            sim[i,j] = val
            sim[j,i] = val
    return sim

@st.cache_resource
def load_model():
    # Dummy KMeans model with 3 clusters trained on dummy data
    X = np.array([[10, 2, 300],
                  [5, 5, 150],
                  [20, 1, 500],
                  [15, 3, 400],
                  [7, 4, 200],
                  [3, 6, 100],
                  [8, 2, 220]])
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)
    return model

df = load_data()
similarity = load_similarity()
kmeans = load_model()

# ---------------------------------------
# Product Recommender Function
# ---------------------------------------
def recommend_products(product_name):
    try:
        index = df[df['product_name'].str.lower() == product_name.lower()].index[0]
    except IndexError:
        return []
    
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    recommended = [df.iloc[i[0]].product_name for i in distances[1:6]]
    return recommended

# ---------------------------------------
# Customer Segment Predictor Function
# ---------------------------------------
def predict_cluster(recency, frequency, monetary):
    input_data = np.array([[recency, frequency, monetary]])
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_data)
    cluster = kmeans.predict(scaled_input)[0]

    if cluster == 0:
        return "High-Value"
    elif cluster == 1:
        return "Regular"
    elif cluster == 2:
        return "Occasional"
    else:
        return "At-Risk"

# ---------------------------------------
# Tabs UI
# ---------------------------------------
tab1, tab2 = st.tabs(["🔎 Product Recommender", "🧠 Customer Segment Predictor"])

# -------------------- Tab 1 --------------------
with tab1:
    st.subheader("🔍 Get Similar Products")
    product_list = df['product_name'].dropna().unique()
    selected_product = st.selectbox("Select a product:", sorted(product_list))

    if st.button("Recommend Similar Products"):
        recommendations = recommend_products(selected_product)
        if recommendations:
            st.success("✅ Recommended Products:")
            for i, prod in enumerate(recommendations, start=1):
                st.write(f"{i}. {prod}")
        else:
            st.error("⚠️ Product not found or no recommendations available.")

# -------------------- Tab 2 --------------------
with tab2:
    st.subheader("📋 Predict Customer Segment")
    
    rec = st.number_input("Enter Recency (days since last purchase):", min_value=0, value=30)
    freq = st.number_input("Enter Frequency (number of purchases):", min_value=0, value=5)
    mon = st.number_input("Enter Monetary (total amount spent):", min_value=0.0, value=100.0)

    if st.button("Predict Segment"):
        segment = predict_cluster(rec, freq, mon)
        st.write(f"### 🧠 Predicted Segment: **{segment}**")
