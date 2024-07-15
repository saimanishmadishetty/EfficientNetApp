import streamlit as st
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import base64
import io

# Set the title and description with new font style
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }
        .title {
            font-size: 2.5rem;
            color: #4CAF50;
            text-align: center;
        }
        .description {
            font-size: 1.25rem;
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
        }
        .uploaded-image {
            border: 2px solid #4CAF50;
            border-radius: 8px;
        }
        .prediction-container {
            text-align: center;
            margin-top: 20px;
        }
        .prediction-title {
            font-size: 24px;
            color: #333;
        }
        .prediction-class {
            font-size: 20px;
            color: #4CAF50;
        }
        .confidence {
            font-size: 20px;
            color: #FF5733;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Image Classification App</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload an image and let the EfficientNet model classify it. This model can identify a variety of objects and scenes.</div>', unsafe_allow_html=True)

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Center the classify button
st.markdown("""
    <style>
        .stButton button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

if st.button('🔍 Classify'):
    st.session_state.classify = True
else:
    st.session_state.classify = False

if uploaded_file is not None:
    vps_model_client = model.ModelClient()
    model_id = "mdl-i5bsdoczyhmkp"
    image = Image.open(uploaded_file)
    
    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    input_data = img_str

    if st.session_state.classify:
        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=img_str)
            detected_classes = api_response[0].split(', ')
            confidence = api_response[1]
        except UnauthorizedException:
            st.error("Unauthorized exception")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")
    else:
        detected_classes = []
        confidence = 0.0

    # Layout for image and prediction
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        if st.session_state.classify:
            st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 24px; color: #333;"><strong>Prediction:</strong></p>', unsafe_allow_html=True)
            
            for detected_class in detected_classes:
                st.markdown(f'<p style="font-size: 20px; color: #4CAF50;">{detected_class}</p>', unsafe_allow_html=True)
                
            st.markdown(f'<p style="font-size: 20px; color: #FF5733;">Confidence: {confidence:.2%}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="text-align: center; margin-top: 20px;">
                    <p style="font-size: 24px; color: #333;"><strong>Prediction:</strong></p>
                    <p style="font-size: 20px; color: #FF5733;">Upload an image and click "Classify" to see the prediction.</p>
                </div>
            """, unsafe_allow_html=True)

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
