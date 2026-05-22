import streamlit as st
from PIL import Image
import pandas as pd
from pipeline import predict_pipeline

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main { background-color: #f7f9fc; }

.title-box {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #101828, #1d3557);
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.result-card {
    padding: 22px;
    border-radius: 18px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="title-box">
    <h1>🧠 Brain Tumor Detection & Classification</h1>
    <p>Two-stage AI pipeline using MRI images for tumor screening and classification</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("Project Overview")
st.sidebar.markdown("""
**Medical Image Analysis**

**Pipeline**
1. Binary PyTorch model
   - Tumor vs No Tumor
2. Multi-class VGG16 model
   - Glioma
   - Meningioma
   - Pituitary

**Team Members**
- Cameron Askins
- Axel Espinosa-Chan
- Williams Okoye
- Tirth Patel
""")
st.sidebar.warning("This tool is for educational purposes only and is not a medical diagnosis system.")

# ---------- Main Layout ----------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    else:
        st.info("Upload an MRI image to begin analysis.")

with right_col:
    st.subheader("Prediction Results")

    if uploaded_file is not None:
        # Only re-run inference when a new file is uploaded.
        # file_id is unique per upload; reusing the same file reuses the cached result.
        file_id = uploaded_file.file_id
        if "result" not in st.session_state or st.session_state.get("file_id") != file_id:
            with st.spinner("Analyzing MRI image..."):
                st.session_state.result  = predict_pipeline(image)
                st.session_state.file_id = file_id

        result     = st.session_state.result
        prediction = result["prediction"]
        confidence = result["confidence"]
        stage      = result["stage"]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # "suspected → Doctor Review" also contains "Doctor Review" — caught correctly
        if "No Tumor" in prediction:
            st.success("No Tumor Detected")
        elif "Doctor Review" in prediction:
            st.warning("Low Confidence — Doctor Review Recommended")
        else:
            st.error("Tumor Detected")

        st.markdown(f"### Prediction: **{prediction}**")
        st.markdown(f"### Confidence: **{confidence * 100:.2f}%**")
        st.markdown(f"**Stage:** {stage}")

        st.markdown('</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")
        with col_b:
            if "No Tumor" in prediction:
                st.metric("Risk Level", "Low")
            elif "Doctor Review" in prediction:
                st.metric("Risk Level", "Review")
            else:
                st.metric("Risk Level", "High")

    else:
        st.info("Upload an MRI image to see results.")

# ---------- Probability Chart ----------
# Guarded against result being undefined on initial load or after reruns
if uploaded_file is not None and "result" in st.session_state:
    result = st.session_state.result
    st.markdown("---")
    st.subheader("Probability Distribution")

    if "all_probabilities" in result and "class_labels" in result:
        chart_df = pd.DataFrame({
            "Class":       result["class_labels"],
            "Probability": result["all_probabilities"]
        })
        st.bar_chart(chart_df, x="Class", y="Probability")
        st.dataframe(
            chart_df.assign(
                Probability=chart_df["Probability"].apply(lambda x: f"{x * 100:.2f}%")
            ),
            use_container_width=True
        )
    else:
        # Model 1 early exit — no class breakdown available
        st.info("Tumor screener returned No Tumor with high confidence — multiclass stage not reached.")

# ---------- Method Section ----------
st.markdown("---")
st.subheader("How the System Works")

step1, step2, step3 = st.columns(3)
with step1:
    st.markdown("### 1. Upload\nThe user uploads a brain MRI image.")
with step2:
    st.markdown("### 2. Analyze\nThe binary model checks for tumor presence.")
with step3:
    st.markdown("### 3. Classify\nIf a tumor is detected, the second model predicts tumor type.")

# ---------- Footer ----------
st.markdown("""
<div class="footer">
Built with Streamlit, PyTorch, and TensorFlow/Keras | Medical Image Analysis Project
</div>
""", unsafe_allow_html=True)
