import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prince import MCA

# ----------------------------
# Load Models + Artifacts
# ----------------------------
mca = joblib.load("mca_transformer.joblib")
scaler = joblib.load("scaler.joblib")
kmeans = joblib.load("kmeans_model.joblib")

cluster_profiles = pd.read_csv("cluster_profiles.csv", index_col="Cluster")
ui_categories = pd.read_csv("ui_categories.csv")

# ----------------------------
# Feature list
# ----------------------------
features = [
    'family_history', 'treatment', 'Growing_Stress', 'Changes_Habits',
    'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness'
]

# ----------------------------
# Friendly Questions for UI
# ----------------------------
friendly_labels = {
    'family_history': "Does your family have a history of mental health concerns?",
    'treatment': "Are you currently undergoing treatment?",
    'Growing_Stress': "Do you feel your stress levels are increasing?",
    'Changes_Habits': "Have your habits changed recently (sleep, diet, routines)?",
    'Mood_Swings': "Do you experience mood swings?",
    'Coping_Struggles': "Do you struggle to cope with everyday challenges?",
    'Work_Interest': "How interested are you in your work recently?",
    'Social_Weakness': "Do you feel less socially engaged than usual?"
}

# ----------------------------
# Streamlit Page Layout
# ----------------------------
st.title("🧠 Mental Health Cluster Prediction App")
st.write("Please answer the questions below to find your cluster profile.")

st.markdown("---")

# ----------------------------
# Collect User Inputs
# ----------------------------
user_input = {}
columns = st.columns(2)

for i, feature in enumerate(features):
    with columns[i % 2]:
        values = ui_categories[feature].dropna().tolist()
        user_input[feature] = st.selectbox(friendly_labels[feature], values)

# ----------------------------
# Predict Function
# ----------------------------
def predict_cluster(user_dict):
    user_df = pd.DataFrame([user_dict]).astype(str)
    
    # MCA transform
    user_mca = mca.transform(user_df)
    user_arr = np.asarray(user_mca)

    # Scale
    user_scaled = scaler.transform(user_arr)

    # Predict cluster
    cluster_pred = kmeans.predict(user_scaled)[0]
    return cluster_pred

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🔍 Predict My Cluster"):
    cluster = predict_cluster(user_input)
    
    st.success(f"### 🎯 You belong to **Cluster {cluster}**")
    
    st.markdown("### 🧩 Cluster Profile")
    st.write(cluster_profiles.loc[[cluster]])
    
    st.markdown("### 📝 Your Answers")
    st.json(user_input)