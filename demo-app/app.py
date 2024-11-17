import streamlit as st
from transformers import pipeline
import urllib.parse

# load the fine-tuned model
save_dir = "/Users/harsita/Desktop/btt/btt-google-2e/demo-app/mlm_model_save"
mask_filler = pipeline('fill-mask', model=save_dir, tokenizer=save_dir, framework='tf')

# set page title and layout
st.set_page_config(page_title="Google 2E", layout="centered")

# add custom CSS for styling
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
        div[data-baseweb="input"] > div {
        margin-top: 0;  /* Removes any extra spacing from the top */
    }
    </style>
    <div class="title">
        <span>G</span><span>o</span><span>o</span><span>g</span><span>l</span><span>e</span><span> </span><span>2</span><span>E</span>
    </div>
    """,
    unsafe_allow_html=True
)

# add additional styling or instructions
st.info("Enter a query with <mask> to get it's the top predictions based on our masked language model.")

# input from the user
query = st.text_input(
    label="Search",
    placeholder="e.g., 20 newtons equals how many <mask>?"
)

# submit button
if st.button("Predict"):
    if "<mask>" not in query:
        st.error("Your query must include <mask>. Please try again.")
    else:
        # perform inference
        results = mask_filler(query, top_k=5)
        
        st.markdown("<div class='result'>", unsafe_allow_html=True)
        st.markdown("<p><b>Top Predictions:</b></p>", unsafe_allow_html=True)
        
        for i, res in enumerate(results, start=1):
            prediction = res['sequence']
            encoded_query = urllib.parse.quote(prediction)
            search_url = f"https://www.google.com/search?q={encoded_query}"
            
            # using Streamlit's markdown to create clickable entries with the Google search URL
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


