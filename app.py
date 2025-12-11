import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prince import MCA
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from datetime import datetime

# ============================================================
# Load Models & Artifacts
# ============================================================

mca = joblib.load("mca_transformer.joblib")
scaler = joblib.load("scaler.joblib")
kmeans = joblib.load("kmeans_model.joblib")
ui_categories = pd.read_csv("ui_categories.csv")

features = [
    'family_history', 'treatment', 'Growing_Stress', 'Changes_Habits',
    'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness'
]

# ============================================================
# UI Labels
# ============================================================

friendly_labels = {
    'family_history': "Does your family have a history of mental health concerns?",
    'treatment': "Are you currently undergoing any mental health treatment?",
    'Growing_Stress': "Do you feel your stress levels have been increasing recently?",
    'Changes_Habits': "Have you noticed recent changes in your habits (sleep, diet, routines)?",
    'Mood_Swings': "Do you experience noticeable mood swings?",
    'Coping_Struggles': "Do you struggle to cope with everyday challenges?",
    'Work_Interest': "How interested are you in your work lately?",
    'Social_Weakness': "Do you feel less socially engaged or more withdrawn than usual?"
}

# ============================================================
# Cluster Descriptions
# ============================================================

cluster_descriptions = {

    0: {
        "title": "Stable and Balanced",
        "description": (
            "You show signs of emotional steadiness, healthy routines, and an effective ability to handle daily stress. "
            "This profile reflects resilience, consistency, and a strong internal balance."
        ),
        "suggestions": [
            "Maintain consistent sleep and activity routines.",
            "Stay connected with supportive individuals.",
            "Practice mindfulness or brief reflection exercises.",
            "Identify early stress signals and respond proactively.",
            "Engage in hobbies that promote emotional stability."
        ]
    },

    1: {
        "title": "Mild Stress but Resilient",
        "description": (
            "Your profile reflects mild stress but also strong resilience and adaptability. "
            "Your coping skills remain stable, and small adjustments can improve well-being."
        ),
        "suggestions": [
            "Break tasks into manageable steps.",
            "Take short restorative breaks.",
            "Set healthy boundaries.",
            "Use deep breathing during stressful moments.",
            "Stay connected with supportive people."
        ]
    },

    2: {
        "title": "Low Stress and Strong Well-Being",
        "description": (
            "Your answers reflect emotional well-being and stable coping patterns. "
            "You maintain habits that support long-term mental health."
        ),
        "suggestions": [
            "Continue your positive routines.",
            "Maintain meaningful social connections.",
            "Practice gratitude or reflective journaling.",
            "Pursue fulfilling hobbies.",
            "Perform occasional stress-level check-ins."
        ]
    },

    3: {
        "title": "Moderate Emotional Variability",
        "description": (
            "Your responses indicate emotional fluctuations or changes in habits and energy. "
            "These may occur during life transitions or inconsistent routines."
        ),
        "suggestions": [
            "Establish consistent sleep habits.",
            "Avoid multitasking when possible.",
            "Schedule relaxing activities regularly.",
            "Stay engaged in light social interactions.",
            "Identify triggers behind emotional shifts."
        ]
    },

    4: {
        "title": "High Stress and Strain Signals",
        "description": (
            "You may be experiencing elevated stress or emotional strain. "
            "Restorative actions and supportive routines can help you recharge."
        ),
        "suggestions": [
            "Create structured breaks during demanding tasks.",
            "Use grounding exercises like slow breathing.",
            "Set limits to protect personal time.",
            "Reduce non-essential commitments.",
            "Reach out to someone you trust."
        ]
    },

    5: {
        "title": "Emerging Social or Emotional Struggles",
        "description": (
            "Your responses suggest reduced emotional energy or lower social engagement. "
            "This may arise during overwhelming or isolating phases."
        ),
        "suggestions": [
            "Reconnect gradually with trusted individuals.",
            "Engage in gentle physical movement.",
            "Spend time outdoors for clarity and calm.",
            "Celebrate small wins to build momentum.",
            "Return to comforting routines."
        ]
    }
}

# ============================================================
# PDF GENERATOR
# ============================================================

def generate_pdf_report(cluster_info, user_input, user_name):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    width, height = letter
    x_margin = 50
    y = height - 60
    page_number = 1

    def draw_footer(page_num):
        c.setFont("Helvetica", 9)
        c.drawRightString(width - 50, 30, f"Page {page_num}")

    # Title
    c.setFont("Helvetica-Bold", 22)
    c.drawString(x_margin, y, "Your Mental Well-Being Report")
    y -= 40

    # Date + Time (dd/mm/yyyy, HH:MM AM/PM)
    now = datetime.now()
    formatted_datetime = now.strftime("%d/%m/%Y, %I:%M %p")

    c.setFont("Helvetica", 12)

    if user_name.strip():
        c.drawString(x_margin, y, f"Prepared For: {user_name}")
        y -= 20

    c.drawString(x_margin, y, f"Date Generated: {formatted_datetime}")
    y -= 30

    # Cluster Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_margin, y, f"Cluster: {cluster_info['title']}")
    y -= 15

    c.line(x_margin, y, width - x_margin, y)
    y -= 30

    # Interpretation
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x_margin, y, "Interpretation")
    y -= 20

    c.setFont("Helvetica", 12)
    text = c.beginText(x_margin, y)
    text.setLeading(16)
    for line in cluster_info["description"].split(". "):
        text.textLine(line.strip() + ".")
    c.drawText(text)
    y = text.getY() - 20

    c.line(x_margin, y + 10, width - x_margin, y + 10)
    y -= 25

    # Suggestions
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x_margin, y, "Personalized Suggestions")
    y -= 20

    c.setFont("Helvetica", 12)
    for s in cluster_info["suggestions"]:
        c.drawString(x_margin + 15, y, f"• {s}")
        y -= 18

        if y < 120:
            draw_footer(page_number)
            c.showPage()
            page_number += 1
            y = height - 60
            c.setFont("Helvetica-Bold", 14)
            c.drawString(x_margin, y, "Personalized Suggestions (continued)")
            y -= 30

    c.line(x_margin, y, width - x_margin, y)
    y -= 30

    # User Responses
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x_margin, y, "Your Responses")
    y -= 20

    c.setFont("Helvetica", 12)
    for key, value in user_input.items():
        pretty_key = key.replace("_", " ").title()
        c.drawString(x_margin + 10, y, f"{pretty_key}: {value}")
        y -= 16

        if y < 100:
            draw_footer(page_number)
            c.showPage()
            page_number += 1
            y = height - 60
            c.setFont("Helvetica-Bold", 14)
            c.drawString(x_margin, y, "Your Responses (continued)")
            y -= 30

    draw_footer(page_number)
    c.save()
    buffer.seek(0)
    return buffer

# ============================================================
# Prediction Function
# ============================================================

def predict_cluster(user_dict):
    df = pd.DataFrame([user_dict]).astype(str)
    mca_data = mca.transform(df)
    scaled = scaler.transform(np.asarray(mca_data))
    return int(kmeans.predict(scaled)[0])

# ============================================================
# FRONT-END UI (Modern Layout + Simple Results Section)
# ============================================================

st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 5px;'>Mental Health Cluster Insight Tool</h1>
    <p style='text-align: center; color: grey; font-size: 16px;'>
        A supportive tool designed to help you understand your emotional well-being patterns.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Name Section
st.markdown("### Your Details")
user_name = st.text_input("Enter your name (optional):")

st.markdown(
    """
    <p style='color: #444;'>
        Please answer all questions honestly. Your responses help generate a personalized emotional profile.
    </p>
    """,
    unsafe_allow_html=True
)

# Survey Section
st.markdown("### Your Responses")

user_input = {}
for feature in features:
    options = ui_categories[feature].dropna().tolist()
    user_input[feature] = st.selectbox(friendly_labels[feature], options)

st.markdown("<hr>", unsafe_allow_html=True)

# Results Section
st.markdown("### Results")

if st.button("Generate My Well-Being Insights"):
    cid = predict_cluster(user_input)
    info = cluster_descriptions[cid]

    st.success(f"Cluster Identified: {info['title']}")
    st.write(info["description"])

    st.subheader("Personalized Suggestions")
    for s in info["suggestions"]:
        st.write(f"- {s}")

    pdf_buffer = generate_pdf_report(info, user_input, user_name)

    st.download_button(
        label="Download My Wellness Report (PDF)",
        data=pdf_buffer,
        file_name=f"Wellness_Report_{user_name or 'User'}.pdf",
        mime="application/pdf"
    )

    with st.expander("Your Responses"):
        st.json(user_input)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: left; font-size: 13px; color: gray;'>
        <strong>Disclaimer:</strong><br>
        This tool provides general mental well-being insights based on your responses. It is not a medical diagnosis or a substitute for professional mental health care. If you are experiencing emotional distress, please reach out to a qualified mental health professional.
    </div>
    """,
    unsafe_allow_html=True
)
