import streamlit as st
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json
import base64
import io

# Set the title and description
st.title("EfficientNet Object Detection")
st.markdown("""
    Upload an image and let the EfficientNet model detect objects in it.
    This model can identify a variety of objects.
""")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    vps_model_client = model.ModelClient()
    model_id = "mdl-e49tnffczhv4l"
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    input_data = img_str
    if st.button('🔍 Detect'):
        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=img_str)
            
            # Extract class and confidence
            detected_class = api_response[0]
            confidence = api_response[1]
            
            # Display the result with styling
            st.markdown(f"""
                <div style="text-align: center; margin-top: 20px;">
                    <p style="font-size: 24px; color: #333;"><strong>Prediction:</strong></p>
                    <p style="font-size: 20px; color: #4CAF50;">Class: {detected_class}</p>
                    <p style="font-size: 20px; color: #FF5733;">Confidence: {confidence:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException:
            st.error("Unauthorized exception")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f5f5;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle, .stMarkdown, .stButton, .stImage {
            text-align: center;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stImage > img {
            border: 2px solid #4CAF50;
            border-radius: 8px;
        }
        pre {
            background: #e0f7fa;
            padding: 15px;
            border-radius: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
            border: 1px solid #4CAF50;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)
