
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Fake Currency Detection",
    page_icon="💵",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "currency_model.keras"
CLASSES_PATH = BASE_DIR / "classes.json"

st.title("💵 Fake Currency Detection System")
st.write("Upload a currency image to check whether it is Fake or Genuine.")

# Load AI model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"AI Model not found: {MODEL_PATH.name}"
        )
    return tf.keras.models.load_model(MODEL_PATH)

# Load class names
def load_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Classes file not found: {CLASSES_PATH.name}"
        )

    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)

    return classes

# Check files
try:
    model = load_model()
    classes = load_classes()

    st.success("AI Model Loaded Successfully!")

except Exception as e:
    st.error(f"AI Model Error: {e}")
    st.info(
        "Make sure currency_model.keras and classes.json "
        "are present in the same folder as app.py."
    )
    st.stop()

# Upload image
uploaded_file = st.file_uploader(
    "Upload Currency Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Currency Image",
        use_container_width=True
    )

    if st.button("Detect Currency", type="primary"):

        with st.spinner("Analyzing Currency..."):

            try:
                # Preprocess image
                image = image.resize((224, 224))

                image_array = np.array(image)
                image_array = image_array.astype("float32")
                image_array = np.expand_dims(image_array, axis=0)

                # Prediction
                prediction = model.predict(
                    image_array,
                    verbose=0
                )

                prediction = np.array(prediction)

                # Handle binary or multiclass output
                if prediction.size == 1:
                    probability = float(prediction.flatten()[0])
                    index = 1 if probability >= 0.5 else 0
                    confidence = (
                        probability if index == 1
                        else 1 - probability
                    )

                else:
                    scores = prediction.flatten()
                    index = int(np.argmax(scores))
                    confidence = float(scores[index])

                # Get predicted class
                if isinstance(classes, list):
                    class_name = classes[index]
                else:
                    class_name = classes[str(index)]

                class_name = str(class_name).lower()

                st.subheader("Detection Result")

                if class_name in ["fake", "forged"]:
                    st.error("⚠️ Result: FAKE CURRENCY")

                elif class_name in ["genuine", "real"]:
                    st.success("✅ Result: GENUINE CURRENCY")

                else:
                    st.warning(f"Predicted Class: {class_name}")

                st.write(
                    f"Confidence: {confidence * 100:.2f}%"
                )

                st.progress(
                    min(max(confidence, 0.0), 1.0)
                )

                st.caption(
                    "This is an experimental AI prediction, "
                    "not an official currency authentication."
                )

            except Exception as e:
                st.error(f"Prediction Error: {e}")
