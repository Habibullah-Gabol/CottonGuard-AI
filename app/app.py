import sys
import io
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

# 1. PROJECT CONFIGURATION
BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import project modules AFTER setting the project path
from src.gradcam import generate_gradcam
from src.predict import load_model, predict_image, get_device


CLASS_LABELS = {
    "bacterial_blight": "Bacterial Blight",
    "curl_virus": "Curl Virus",
    "fussarium_wilt": "Fusarium Wilt",
    "healthy": "Healthy Leaf",
}

CLASS_ICONS = {
    "bacterial_blight": "🦠",
    "curl_virus": "🍂",
    "fussarium_wilt": "🍄",
    "healthy": "✅",
}

CLASS_DESCRIPTIONS = {
    "bacterial_blight": (
        "Bacterial blight is a cotton disease that can affect leaves "
        "and other plant tissues. Seek expert confirmation before "
        "deciding on treatment."
    ),
    "curl_virus": (
        "Curl virus can cause leaf curling and changes in leaf "
        "appearance. Field symptoms and expert assessment are "
        "important for confirmation."
    ),
    "fussarium_wilt": (
        "Fusarium wilt is a soil-associated fungal disease that can "
        "cause wilting and vascular damage. Confirm suspected cases "
        "with an agricultural expert."
    ),
    "healthy": (
        "The model did not identify one of the three trained disease "
        "classes. Continue regular crop monitoring."
    ),
}

STAKEHOLDER_GUIDANCE = {
    "Farmer": (
        "Upload a clear photograph of a cotton leaf, review the "
        "predicted condition, and consult an agricultural expert "
        "before taking treatment decisions."
    ),
    "Agricultural Expert": (
        "Review the predicted class and probability distribution. "
        "Use field observations and laboratory confirmation where "
        "appropriate."
    ),
    "Researcher / Student": (
        "Inspect the predicted class and class probabilities. Use "
        "these outputs for research analysis, not as a substitute "
        "for independent validation."
    ),
}

# 2. STREAMLIT PAGE CONFIGURATION

st.set_page_config(
    page_title="CottonGuard AI | Smart Crop Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 3. CUSTOM CSS

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --primary: #1b6b3a;
        --primary-dark: #134d29;
        --primary-light: #e8f5ee;
        --accent: #22c55e;
        --text-dark: #0f2e1d;
        --text-body: #475569;
        --text-muted: #64748b;
        --border: #e2e8f0;
        --bg: #f8fafc;
        --card-bg: #ffffff;
        --warning: #f59e0b;
        --danger: #dc2626;
        --success: #10b981;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-dark);
    }

    .stApp {
        background: var(--bg);
    }

    /* ---------------- Sidebar ---------------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f7f2 0%, #e8f2ea 100%);
        border-right: 1px solid var(--border);
    }

    /* ---------------- Hero ---------------- */
    .hero {
        background: linear-gradient(135deg, #134d29 0%, #1b6b3a 55%, #22c55e 130%);
        padding: 42px 36px;
        border-radius: 20px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 12px 32px rgba(19, 77, 41, 0.18);
    }
    .hero h1 {
        color: white;
        font-size: 36px;
        font-weight: 800;
        margin: 0 0 10px 0;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: #d4ecd9;
        font-size: 16px;
        margin: 0;
        max-width: 720px;
        line-height: 1.55;
    }

    /* ---------------- Section headings ---------------- */
    .section-title {
        color: var(--text-dark);
        font-size: 22px;
        font-weight: 700;
        margin: 8px 0 16px 0;
        letter-spacing: -0.01em;
    }
    .section-subtitle {
        color: var(--text-muted);
        font-size: 14px;
        margin: -10px 0 20px 0;
    }

    /* ---------------- Info / feature cards ---------------- */
    .info-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 22px;
        height: 100%;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .info-card:hover {
        box-shadow: 0 6px 20px rgba(15, 46, 29, 0.06);
        transform: translateY(-2px);
    }
    .info-card h3 {
        color: var(--text-muted);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0 0 10px 0;
    }
    .info-card p {
        color: var(--text-dark);
        font-size: 20px;
        font-weight: 700;
        margin: 0;
    }
    .info-card small {
        color: var(--text-muted);
        font-size: 12px;
    }

    /* ---------------- Step badge ---------------- */
    .step-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 26px 0 14px 0;
    }
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--primary);
        color: white;
        font-weight: 700;
        font-size: 14px;
        flex-shrink: 0;
    }
    .step-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-dark);
        margin: 0;
    }
    .step-subtitle {
        font-size: 13px;
        color: var(--text-muted);
        margin: 2px 0 0 0;
    }

    /* ---------------- Result card ---------------- */
    .result-card {
        border-radius: 16px;
        padding: 26px 28px;
        margin: 14px 0 20px 0;
        border: 1px solid;
    }
    .result-card.healthy {
        background: #ecfdf5;
        border-color: #a7f3d0;
    }
    .result-card.disease {
        background: #fffbeb;
        border-color: #fde68a;
    }
    .result-card .label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0 0 6px 0;
    }
    .result-card.healthy .label { color: #047857; }
    .result-card.disease .label { color: #b45309; }
    .result-card h2 {
        font-size: 26px;
        margin: 0 0 8px 0;
        font-weight: 800;
    }
    .result-card.healthy h2 { color: #065f46; }
    .result-card.disease h2 { color: #92400e; }
    .result-card .conf {
        font-size: 14px;
        color: var(--text-body);
        margin: 0;
    }

    /* ---------------- Probability bars ---------------- */
    .prob-row { margin: 14px 0; }
    .prob-head {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 14px;
    }
    .prob-name { font-weight: 600; color: var(--text-dark); }
    .prob-value { color: var(--text-muted); font-variant-numeric: tabular-nums; }
    .prob-track {
        height: 8px;
        background: #e2e8f0;
        border-radius: 6px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }

    /* ---------------- Primary button ---------------- */
    div.stButton > button[kind="primary"] {
        background: var(--primary);
        border: none;
        border-radius: 10px;
        min-height: 50px;
        font-weight: 700;
        font-size: 15px;
        width: 100%;
        transition: background 0.2s ease, transform 0.1s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background: var(--primary-dark);
    }
    div.stButton > button[kind="primary"]:active {
        transform: scale(0.99);
    }

    /* ---------------- Download button ---------------- */
    div.stDownloadButton > button {
        border-radius: 10px;
        min-height: 46px;
        font-weight: 600;
        border: 1px solid var(--border);
        background: white;
        color: var(--text-dark);
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    div.stDownloadButton > button:hover {
        border-color: var(--primary);
        background: var(--primary-light);
        color: var(--primary-dark);
    }

    /* ---------------- Metric cards ---------------- */
    div[data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-muted);
        font-size: 13px;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        color: var(--text-dark);
        font-weight: 700;
        font-size: 20px;
    }

    /* ---------------- Expander ---------------- */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid var(--border);
        overflow: hidden;
    }

    /* ---------------- Footer ---------------- */
    .footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 12px;
        padding: 30px 0 12px 0;
        margin-top: 40px;
        border-top: 1px solid var(--border);
        line-height: 1.7;
    }
    .footer b { color: var(--text-dark); }
    </style>
    """,
    unsafe_allow_html=True,
)

# 4. MODEL LOADING

@st.cache_resource
def get_model():
    return load_model(device=get_device())


# 5. SIDEBAR

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:18px 0 10px 0;">
            <div style="font-size:48px; line-height:1;">🌿</div>
            <h2 style="color:#134d29; margin:6px 0 2px 0; font-weight:800;">
                CottonGuard AI
            </h2>
            <p style="color:#6b806f; font-size:12px; margin:0;">
                Intelligent Cotton Crop Protection
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        ["Disease Detection", "About the Project"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("#### 👤 Your Role")
    stakeholder = st.selectbox(
        "Select your role",
        list(STAKEHOLDER_GUIDANCE.keys()),
        label_visibility="collapsed",
    )
    st.info(STAKEHOLDER_GUIDANCE[stakeholder])

    st.divider()

    st.markdown("#### ⚙️ Model Information")
    st.markdown(
        f"""
        - **Architecture:** ResNet18
        - **Framework:** PyTorch
        - **Classes:** {len(CLASS_LABELS)}
        - **Device:** `{get_device()}`
        """
    )

    st.caption("CottonGuard AI · Research Prototype")


# 6. DISEASE DETECTION PAGE

if page == "Disease Detection":

    # ---------------- Hero ----------------
    st.markdown(
        """
        <div class="hero">
            <h1>🌿 Smart Cotton Disease Detection</h1>
            <p>
                Detect cotton leaf diseases using deep learning.
                Upload a clear leaf image, run the analysis, and
                review the model's predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Quick facts ----------------
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Disease Classes</h3>
                <p>4 Conditions</p>
                <small>3 diseases + healthy leaf</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Model</h3>
                <p>ResNet18</p>
                <small>Fine-tuned on cotton leaf images</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """
            <div class="info-card">
                <h3>Analysis Time</h3>
                <p>Seconds</p>
                <small>Fast inference on the selected device</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # ---------------- Step 1: Upload ----------------
    st.markdown(
        """
        <div class="step-header">
            <span class="step-badge">1</span>
            <div>
                <p class="step-title">Upload a Cotton Leaf Image</p>
                <p class="step-subtitle">
                    Use a clear, well-lit photograph of a single leaf.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_col, tips_col = st.columns([1.6, 1], gap="large")

    with upload_col:
        uploaded_file = st.file_uploader(
            "Choose an image from your device",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Supported formats: JPG, JPEG, PNG, WEBP, BMP. Max 10 MB.",
            label_visibility="collapsed",
        )

    with tips_col:
        st.markdown(
            """
            <div class="info-card" style="background:#f0f7f2; border-style:dashed;">
                <h3>📸 Image Tips</h3>
                <ul style="color:#475569; font-size:13.5px; padding-left:18px; margin:0; line-height:1.85;">
                    <li>Capture the leaf clearly, in focus</li>
                    <li>Use good natural lighting</li>
                    <li>Avoid blurry or shadowed photos</li>
                    <li>Keep the leaf fully visible</li>
                    <li>Use a simple, uncluttered background</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- Handle upload ----------------
    if uploaded_file is None:
        st.info("👆 Upload a cotton leaf image above to begin.")
    else:
        try:
            image_bytes = uploaded_file.getvalue()

            if len(image_bytes) > 10 * 1024 * 1024:
                st.error("Please upload an image smaller than 10 MB.")
                st.stop()

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Reset analysis if a different file is uploaded
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("current_file_id") != file_id:
                st.session_state["current_file_id"] = file_id
                st.session_state.pop("prediction_result", None)
                st.session_state.pop("gradcam_result", None)

            # ---------------- Step 2: Preview + Analyze ----------------
            st.markdown(
                """
                <div class="step-header">
                    <span class="step-badge">2</span>
                    <div>
                        <p class="step-title">Preview &amp; Run Analysis</p>
                        <p class="step-subtitle">
                            Confirm the image looks correct, then run the model.
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            preview_col, action_col = st.columns([1, 1], gap="large")

            with preview_col:
                st.image(image, use_container_width=True)
                st.caption(
                    f"**{uploaded_file.name}** · "
                    f"{image.width} × {image.height} px"
                )

            with action_col:
                st.markdown(
                    """
                    <div class="info-card" style="height:auto;">
                        <h3>Ready for analysis</h3>
                        <p style="font-size:15px; font-weight:500; color:#475569;">
                            The image has been loaded successfully.
                            Click the button below to run the CottonGuard AI
                            prediction model.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("")

                analyze = st.button(
                    "🔍 Analyze Cotton Leaf",
                    type="primary",
                    use_container_width=True,
                )

                if analyze:
                    with st.spinner("Analyzing your image..."):
                        model = get_model()
                        result = predict_image(
                            image, model, device=get_device()
                        )

                    predicted_class = result["predicted_class"]
                    confidence = float(result["confidence"])

                    if predicted_class not in CLASS_LABELS:
                        raise ValueError(
                            f"Unknown predicted class: {predicted_class}"
                        )

                    st.session_state["prediction_result"] = {
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "probabilities": result["probabilities"],
                        "image_name": uploaded_file.name,
                    }

            # ---------------- Step 3: Results ----------------
            saved = st.session_state.get("prediction_result")
            if saved and saved["image_name"] == uploaded_file.name:

                predicted_class = saved["predicted_class"]
                confidence = saved["confidence"]
                probabilities = saved["probabilities"]
                is_healthy = predicted_class == "healthy"
                card_class = "healthy" if is_healthy else "disease"
                icon = CLASS_ICONS[predicted_class]

                st.markdown(
                    """
                    <div class="step-header">
                        <span class="step-badge">3</span>
                        <div>
                            <p class="step-title">Analysis Results</p>
                            <p class="step-subtitle">
                                Review the model's prediction and probabilities.
                            </p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Main result banner
                st.markdown(
                    f"""
                    <div class="result-card {card_class}">
                        <p class="label">Model Prediction</p>
                        <h2>{icon} {CLASS_LABELS[predicted_class]}</h2>
                        <p class="conf">
                            Confidence: <b>{confidence:.2%}</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Condition", CLASS_LABELS[predicted_class])
                m2.metric("Model Confidence", f"{confidence:.2%}")
                m3.metric("Classes Evaluated", len(CLASS_LABELS))

                # Probability distribution
                st.markdown("### Class Probability Distribution")

                prob_df = pd.DataFrame(
                    [
                        {"Class": CLASS_LABELS[name], "Probability": float(v)}
                        for name, v in probabilities.items()
                    ]
                ).sort_values("Probability", ascending=False)

                for _, row in prob_df.iterrows():
                    p = max(0.0, min(1.0, float(row["Probability"])))
                    # Color: primary for top prediction, muted for others
                    is_top = row["Class"] == CLASS_LABELS[predicted_class]
                    color = "#1b6b3a" if is_top else "#94a3b8"
                    st.markdown(
                        f"""
                        <div class="prob-row">
                            <div class="prob-head">
                                <span class="prob-name">{row['Class']}</span>
                                <span class="prob-value">{p:.2%}</span>
                            </div>
                            <div class="prob-track">
                                <div class="prob-fill"
                                     style="width:{p*100:.2f}%; background:{color};">
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Interpretation
                st.markdown("### Understanding the Prediction")
                st.info(CLASS_DESCRIPTIONS[predicted_class])

                # GRAD-CAM — EXPLAINABLE AI

                st.divider()

                st.markdown("### 🔬 Explainable AI — Grad-CAM")

                st.write(
                    "See which regions of the cotton leaf influenced the "
                    "ResNet18 model's prediction."
                )

                with st.expander("Generate visual explanation", expanded=False):

                    st.info(
                        "Grad-CAM highlights image regions that contributed "
                        "to the selected class prediction. It is an explanation "
                        "of model behavior, not proof of disease location."
                    )

                    generate_button = st.button(
                        "🔥 Generate Grad-CAM Heatmap",
                        key="generate_gradcam",
                        use_container_width=True,
                    )

                    if generate_button:

                        try:
                            with st.spinner("Generating Grad-CAM visualization..."):

                                gradcam_output = generate_gradcam(
                                    image=image,
                                    model=get_model(),
                                    class_idx=list(probabilities.keys()).index(
                                        predicted_class
                                    ),
                                    device=get_device(),
                                )

                                st.session_state["gradcam_result"] = {
                                    "image_name": uploaded_file.name,
                                    "heatmap": gradcam_output["heatmap"],
                                    "overlay": gradcam_output["overlay"],
                                }

                        except Exception as exc:
                            st.error(f"Grad-CAM generation failed: {exc}")

                    saved_cam = st.session_state.get("gradcam_result")

                    if (
                        saved_cam is not None
                        and saved_cam["image_name"] == uploaded_file.name
                    ):

                        st.markdown("#### Visual Explanation")

                        original_col, heatmap_col, overlay_col = st.columns(3)

                        with original_col:
                            st.image(
                                image.resize((224, 224)),
                                caption="Original Image",
                                use_container_width=True,
                            )

                        with heatmap_col:
                            st.image(
                                saved_cam["heatmap"],
                                caption="Grad-CAM Heatmap",
                                use_container_width=True,
                            )

                        with overlay_col:
                            st.image(
                                saved_cam["overlay"],
                                caption="Grad-CAM Overlay",
                                use_container_width=True,
                            )

                        st.success(
                            "Grad-CAM visualization generated successfully."
                        )

                        st.caption(
                            "Red and yellow regions generally indicate stronger "
                            "Grad-CAM activation for the selected class. "
                            "Interpret the visualization cautiously."
                        )

                # Downloadable report
                report_df = pd.DataFrame(
                    [
                        {
                            "Image": saved["image_name"],
                            "Predicted Class": CLASS_LABELS[name],
                            "Probability": float(v),
                            "Prediction Date": datetime.now().isoformat(
                                timespec="seconds"
                            ),
                        }
                        for name, v in probabilities.items()
                    ]
                )
                csv_data = report_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Download Prediction Report (CSV)",
                    data=csv_data,
                    file_name="cottonguard_prediction.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.caption(
                    "The model's confidence is an estimated probability, "
                    "not a guarantee of correctness."
                )

                st.warning(
                    "**Research prototype.** Do not make pesticide or "
                    "crop-treatment decisions based only on this result. "
                    "Consult a qualified agricultural expert."
                )

        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")


# 7. ABOUT THE PROJECT PAGE

elif page == "About the Project":

    st.markdown(
        """
        <div class="hero">
            <h1>About CottonGuard AI</h1>
            <p>
                Deep learning for intelligent cotton crop health analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎯 Our Objective")
    st.write(
        "CottonGuard AI is a research prototype designed to classify "
        "cotton leaf images into four categories using a fine-tuned "
        "ResNet18 convolutional neural network."
    )

    st.markdown("### 🌱 Supported Disease Classes")
    for class_name, display_name in CLASS_LABELS.items():
        with st.expander(f"{CLASS_ICONS[class_name]}  {display_name}"):
            st.write(CLASS_DESCRIPTIONS[class_name])

    st.markdown("### 🧠 Model Architecture")
    st.write("**ResNet18 — Convolutional Neural Network**")
    st.write(
        "The model was fine-tuned for four-class cotton leaf "
        "classification. It processes images at 224 × 224 pixels "
        "using ImageNet normalization."
    )

    st.markdown("### 🛠️ Technology Stack")
    st.markdown(
        """
        - **Python** — core language
        - **PyTorch** — deep learning framework
        - **Torchvision** — model & transforms
        - **Streamlit** — web interface
        - **Pandas** — reporting
        - **Pillow** — image processing
        """
    )

    st.markdown("### ⚠️ Important Limitations")
    st.warning(
        "This application is an **experimental research tool**. "
        "Its predictions may be affected by image quality, dataset "
        "differences, and disease-class variation. It does **not** "
        "replace laboratory testing or professional agricultural diagnosis."
    )


# 8. FOOTER

st.markdown(
    """
    <div class="footer">
        <b>CottonGuard AI</b><br>
        AI-Powered Cotton Leaf Disease Classification<br>
        Built with PyTorch and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)