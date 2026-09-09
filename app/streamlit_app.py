"""
streamlit_app.py

Purpose: Client-facing demo dashboard. Uploads an image, calls the
defect-inspection API, and displays results visually - the piece a
non-technical stakeholder (QA manager) would actually use, versus the
raw Swagger docs which are a developer tool.

Run: streamlit run app/streamlit_app.py
"""

import io

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Defect Inspection Demo", layout="centered")

st.title("Manufacturing Defect Inspection")
st.caption("Upload a product image to check for surface defects")

# Configurable API URL - lets the same dashboard point at a local dev
# server or the live AWS deployment without changing code, just this field.
api_url = st.text_input(
    "API URL",
    value="http://localhost:8000",
    help="Point this at your local server (http://localhost:8000) or your live AWS URL",
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if st.button("Run Inspection", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                uploaded_file.seek(0)  # reset pointer after PIL already read it above
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{api_url}/predict", files=files, timeout=10)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API: {e}")
                st.stop()

        # Draw bounding boxes directly on the image so a non-technical
        # viewer can see exactly what was flagged and where, not just
        # a table of coordinates.
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for det in result["detections"]:
            xmin, ymin, xmax, ymax = det["box_xyxy"]
            # Confident detections in red, "needs review" in orange -
            # mirrors the same policy distinction built into the API.
            color = "orange" if det["needs_review"] else "red"
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            draw.text((xmin, max(0, ymin - 12)), label, fill=color)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image)
        with col2:
            st.subheader("Detected")
            st.image(annotated)

        if result["defect_found"]:
            n_review = sum(1 for d in result["detections"] if d["needs_review"])
            st.warning(f"{len(result['detections'])} potential defect(s) found "
                       f"({n_review} flagged for human review)")
        else:
            st.success("No defects detected")

        st.subheader("Details")
        st.table([
            {
                "Defect Type": d["class_name"],
                "Confidence": f"{d['confidence']:.1%}",
                "Status": "Needs Review" if d["needs_review"] else "Confirmed",
            }
            for d in result["detections"]
        ])