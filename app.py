import streamlit as st
from transformers import pipeline
import urllib.parse

# Load the fine-tuned model
save_dir = "/Users/harsita/Desktop/btt/btt-google-2e/mlm_model_save"
mask_filler = pipeline('fill-mask', model=save_dir, tokenizer=save_dir, framework='tf')

# Set page title and layout
st.set_page_config(page_title="Google 2E", layout="centered")

# Add custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #ffffff;
        }
        .title {
            text-align: center;
            font-size: 80px;
            font-family: Arial, sans-serif;
            font-weight: bold;
            color: #4285F4;
            margin-bottom: 20px;
            margin-top: 100px;
        }
        .title span {
            color: #4285F4;
        }
        .title span:nth-child(2) {
            color: #EA4335;
        }
        .title span:nth-child(3) {
            color: #FBBC04;
        }
        .title span:nth-child(5) {
            color: #34A853;
        }
        .title span:nth-child(6) {
            color: #EA4335;
        }
        button {
            margin-top: 20px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 25px;
            background-color: #4285F4;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background-color: #357ae8;
        }
        .result {
            margin-top: 30px;
            font-size: 18px;
            font-family: Arial, sans-serif;
        }
        .result .prediction-box {
            margin: 10px 0;
            padding: 20px;
            font-size: 18px;
            font-weight: bold;
            color: white;
            background-color: #4285F4;
            border-radius: 10px;
            cursor: pointer;
            text-align: center;
            transition: background-color 0.3s;
        }
        .result .prediction-box:hover {
            background-color: #357ae8;
        }
    </style>
    <div class="title">
        <span>G</span><span>o</span><span>o</span><span>g</span><span>l</span><span>e</span><span> </span><span>2</span><span>E</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Input from the user
query = st.text_input(
    "Enter your query (include <mask>):",
    placeholder="e.g., 20 newtons equals how many <mask>?"
)

# Submit button
if st.button("Predict"):
    if "<mask>" not in query:
        st.error("Your query must include <mask>. Please try again.")
    else:
        # Perform inference
        results = mask_filler(query, top_k=5)
        
        st.markdown("<div class='result'>", unsafe_allow_html=True)
        st.markdown("<p><b>Top Predictions:</b></p>", unsafe_allow_html=True)
        
        for i, res in enumerate(results, start=1):
            prediction = res['sequence']
            encoded_query = urllib.parse.quote(prediction)
            search_url = f"https://www.google.com/search?q={encoded_query}"
            
            # Instead of an ordered list, display clickable boxes
            # Using Streamlit's markdown to create clickable boxes with the Google search URL
            st.markdown(
                f"""
                <a href="{search_url}" target="_blank">
                    <div class="prediction-box">
                        {prediction}
                    </div>
                </a>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)

# Add additional styling or instructions
st.info("Enter a query with `<mask>` to get the top predictions based on your model.")
