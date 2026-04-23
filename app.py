"""
PCB Image Uploader
"""
import os
from pathlib import Path
from datetime import datetime
import streamlit as st

UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(exist_ok=True)

st.title("Image Upload")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "gif", "webp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Preview", use_container_width=True)

    ext = Path(uploaded_file.name).suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    save_path = UPLOAD_DIR / filename

    if st.button("Save Image"):
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved to `{save_path}`")

st.divider()
st.subheader("Saved Images")

images = sorted(UPLOAD_DIR.iterdir()) if UPLOAD_DIR.exists() else []

if not images:
    st.info("No images saved yet.")
else:
    cols = st.columns(3)
    for i, img_path in enumerate(images):
        with cols[i % 3]:
            st.image(str(img_path), caption=img_path.name, use_container_width=True)

            if st.button("🗑️ Delete", key=f"btn_{img_path.name}", use_container_width=True):
                try:
                    os.remove(img_path)
                    st.toast(f"Deleted: {img_path.name}")
                    st.rerun()
                except (OSError, PermissionError) as e:
                    st.error(f"Error: {e}")
