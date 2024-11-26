import streamlit as st
from transformers import pipeline, TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
import urllib.parse

# load models
mlm_model_path = "/Users/harsita/Desktop/btt/btt-google-2e/demo-app/mlm_model_save"
gpt_model_path = "/Users/harsita/Desktop/btt/btt-google-2e/demo-app/gpt_model_save2"

mask_filler = pipeline('fill-mask', model=mlm_model_path, tokenizer=mlm_model_path, framework='tf')
gpt_model = TFGPT2LMHeadModel.from_pretrained(gpt_model_path)
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_path)

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
            margin-top: 0; /* Removes any extra spacing from the top */
        }
    </style>
    <div class="title">
        <span>G</span><span>o</span><span>o</span><span>g</span><span>l</span><span>e</span><span> </span><span>2</span><span>E</span>
    </div>
    """,
    unsafe_allow_html=True
)

# create tabs for the two functionalities
tabs = st.tabs(["MaskedLM", "GPT2"])

with tabs[0]:
    # display informational message to user
    st.info("Enter a query with `<mask>` to get top predictions based on our masked language model.")

    # input field for user 
    query = st.text_input(label="Search", placeholder="e.g., 20 newtons equals how many <mask>?")

    # button for prediction
    if st.button("Predict", key="mask_filler_button"):
        # check if query contains <mask> token
        if "<mask>" not in query:
            st.error("Your query must include <mask>. Please try again.")
        else:
            # get top 5 predictions from maskedlm model
            results = mask_filler(query, top_k=5)

            # display the results 
            st.markdown("<div class='result'>", unsafe_allow_html=True)
            st.markdown("<p><b>Top Predictions:</b></p>", unsafe_allow_html=True)

            # loop through results and create clickable links for each prediction
            for res in results:
                prediction = res['sequence']
                encoded_query = urllib.parse.quote(prediction)
                search_url = f"https://www.google.com/search?q={encoded_query}"

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

with tabs[1]:
    # message for user
    st.info("Enter a prompt to get predictions from our GPT model.")

    # input field for user 
    prompt = st.text_input(label="Enter your prompt:", placeholder="e.g., Interesting facts about Egypt")

    # button for text generation
    if st.button("Generate", key="gpt_button"):
        # check if the prompt is empty
        if not prompt:
            st.error("Please enter a prompt.")
        else:
            # tokenize and truncate the input prompt
            input_ids_full = gpt_tokenizer(prompt, return_tensors='tf')
            n = 3  # number of tokens to truncate for input
            input_ids = tf.expand_dims(input_ids_full['input_ids'][0][:-n], 0)
            attention_mask = input_ids_full['attention_mask'][:, :-n]

            # generate predictions with controlled parameters
            outputs = gpt_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=gpt_tokenizer.eos_token_id,
                max_length=input_ids.shape[1] + n + 10,
                num_return_sequences=5,  # generate top 5 sequences
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=17,
                top_p=0.5,
                temperature=0.8,
                repetition_penalty=2.0
            )

            # decode and post-process the outputs
            predictions = [gpt_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            # display results
            st.markdown("<div class='result'>", unsafe_allow_html=True)
            st.markdown("<p><b>Generated Texts:</b></p>", unsafe_allow_html=True)

            # loop through predictions and create clickable links for each prediction
            for prediction in predictions:
                if not prediction.endswith("?"):
                    prediction += "?"
                encoded_query = urllib.parse.quote(prediction)
                search_url = f"https://www.google.com/search?q={encoded_query}"

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