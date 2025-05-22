import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle
from io import BytesIO

# Set custom page layout
st.set_page_config(page_title="Sentiment Prediction App", layout="wide")

# Custom styling
st.markdown(
    """
    <style>
    /* General styling */
    .stButton>button {
        background-color: #007ACC;
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #005A8D;
    }
    .stDataFrame {
        background: #f9f9f9;
        border-radius: 10px;
        padding: 10px;
    }

    /* Fixed footer at the bottom */
    .stFooter {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgb(38, 39, 48);
        color: white;
        text-align: center;
        padding: 4px;
        font-size: 12px;
        font-weight: 500;
    }

    /* Custom styling for download button */
    .stDownloadButton>button {
        background-color: #007ACC;
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 500;
    }
    .stDownloadButton>button:hover {
        background-color: #005A8D;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Initialize model_loaded flag
model_loaded = False

# Load the model
try:
    model = load_model(r"D:/sentiment/sentiment/best_model.h5")
    model_loaded = True
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load the tokenizer
TOKENIZER_PATH = r"D:/sentiment/sentiment/tokenizer.pickle"
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading the tokenizer: {e}")

# Constants for padding
MAX_LEN = 50

# Define a function for tokenization and padding
def preprocess_input(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    return padded

# Define file processing functions
def process_csv(uploaded_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Ensure the text column exists (e.g., 'Text')
    if 'Text' not in df.columns:
        st.error("CSV file must contain a 'Text' column.")
        return None, None
    
    # Apply sentiment prediction on each row
    predictions = []
    for text in df['Text']:
        processed_input = preprocess_input(text)
        prediction = model.predict(processed_input, verbose=0)
        sentiment_classes = ["Negative", "Neutral", "Positive"]
        predicted_class = sentiment_classes[np.argmax(prediction)]
        predictions.append(predicted_class)
    
    # Add predictions to the DataFrame
    df['Predicted Sentiment'] = predictions
    
    # Convert the dataframe back to a file for download
    output_file = BytesIO()
    df.to_csv(output_file, index=False)
    output_file.seek(0)
    
    return output_file, df

def process_excel(uploaded_file):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)
    
    # Ensure the text column exists (e.g., 'Text')
    if 'Text' not in df.columns:
        st.error("Excel file must contain a 'Text' column.")
        return None, None
    
    # Apply sentiment prediction on each row
    predictions = []
    for text in df['Text']:
        processed_input = preprocess_input(text)
        prediction = model.predict(processed_input, verbose=0)
        sentiment_classes = ["Negative", "Neutral", "Positive"]
        predicted_class = sentiment_classes[np.argmax(prediction)]
        predictions.append(predicted_class)
    
    # Add predictions to the DataFrame
    df['Predicted Sentiment'] = predictions
    
    # Convert the dataframe back to a file for download
    output_file = BytesIO()
    df.to_excel(output_file, index=False, engine='openpyxl')
    output_file.seek(0)
    
    return output_file, df

# App Title
st.title("💬 **Sentiment Prediction App**")

# Create columns layout
left, spacer, right = st.columns([1.5, 0.1, 2])

# Left Section: Single Sentence Prediction
with left:
    st.header("📍 **Single Sentence Prediction**")
    st.write("Enter a sentence, and the app will predict whether it's **Positive**, **Negative**, or **Neutral**.")
    user_input = st.text_input("Enter your sentence:", placeholder="Type something meaningful here...")

    if st.button("🔍 Predict Sentiment", key="text_predict"):
        if not model_loaded:
            st.error("Model is not loaded. Please try again later.")
        elif user_input.strip() == "":
            st.warning("Please enter a valid sentence!")
        else:
            try:
                processed_input = preprocess_input(user_input)
                prediction = model.predict(processed_input, verbose=0)
                sentiment_classes = ["Negative", "Neutral", "Positive"]
                predicted_class = sentiment_classes[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                st.success(f"✅ **Prediction:** {predicted_class}")
                st.info(f"📊 **Confidence Level:** {confidence:.2f}%")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
with right:
    st.header("📂 **Process Excel or CSV File**")
    st.write("Upload an `.xlsx` or `.csv` file containing text to analyze.")
    uploaded_file = st.file_uploader("Upload a file", type=["xlsx", "csv"])

    if st.button("📄 Process File", key="file_process"):
        if not model_loaded:
            st.error("Model is not loaded. Please try again later.")
        elif uploaded_file is None:
            st.warning("Please upload a file!")
        else:
            if uploaded_file.name.endswith(".xlsx"):
                output_file, df = process_excel(uploaded_file)
            elif uploaded_file.name.endswith(".csv"):
                output_file, df = process_csv(uploaded_file)
            else:
                st.error("Unsupported file type.")
                output_file, df = None, None

            if output_file and df is not None:
                st.success("✅ **File processed successfully!**")
                st.write("### 🔍 **Results Preview:**")
                st.dataframe(df)

                # Display sentiment counts
                sentiment_counts = df["Predicted Sentiment"].value_counts()
                total_positive = sentiment_counts.get("Positive", 0)
                total_negative = sentiment_counts.get("Negative", 0)
                total_neutral = sentiment_counts.get("Neutral", 0)

                st.markdown("### 📊 **Sentiment Distribution:**")
                st.write(f"- ✅ **Total Positive Samples:** {total_positive}")
                st.write(f"- ❌ **Total Negative Samples:** {total_negative}")
                st.write(f"- ⚖️ **Total Neutral Samples:** {total_neutral}")


                # Styled Download Button
                st.download_button(
                    label="💾 Download Processed File",
                    data=output_file,
                    file_name="processed_file.xlsx" if uploaded_file.name.endswith(".xlsx") else "processed_file.csv",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if uploaded_file.name.endswith(".xlsx") else "text/csv",
                    key="download_processed_file"
                )
st.markdown(
    """
    <div class="stFooter">
        Created with ❤️ by the Team: Srujan Y S, Sanjana M, Chandana V N, Simran | Group Project for Sentiment Analysis | With the guidance of Chetan K R.
    </div>
    """,
    unsafe_allow_html=True
)