"""Streamlit interface for CottonGuard AI. Run: streamlit run app.py"""
import pandas as pd
import streamlit as st
from PIL import Image

from src.predict import get_device, load_model, predict_image

st.set_page_config(
    page_title="CottonGuard AI",
    page_icon="🌿",
    layout="centered",
)

CLASS_LABELS = {
    "bacterial_blight": "Bacterial blight",
    "curl_virus": "Curl virus",
    "fussarium_wilt": "Fusarium wilt",
    "healthy": "Healthy",
}


@st.cache_resource
def get_model():
    """Cache the model so it is loaded only once per Streamlit session."""
    return load_model(device=get_device())


st.title("🌿 CottonGuard AI")
st.subheader("Cotton Leaf Disease Classification")
st.write(
    "Upload a cotton leaf image to receive an experimental four-class prediction."
)
st.warning(
    "Research prototype only. Predictions are not a definitive agricultural "
    "diagnosis. The model showed limited cross-dataset generalization, especially "
    "for curl-virus and Fusarium-wilt examples in the external evaluation."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Analyze image", type="primary"):
            with st.spinner("Loading model and analyzing image..."):
                prediction = predict_image(
                    image,
                    get_model(),
                    device=get_device(),
                )

            label = prediction["predicted_class"]
            st.markdown("### Prediction")
            st.success(f"Predicted class: **{CLASS_LABELS[label]}**")
            st.metric("Model softmax confidence", f"{prediction['confidence']:.2%}")

            st.markdown("### Class probabilities")
            probability_frame = pd.DataFrame(
                {
                    "Probability": prediction["probabilities"],
                }
            ).rename(index=CLASS_LABELS)
            st.bar_chart(probability_frame)

            st.caption(
                "Softmax confidence is not a guarantee of correctness. "
                "Seek expert confirmation before making treatment decisions."
            )
    except Exception as exc:
        st.error(f"Could not process the uploaded image: {exc}")
