# Demo App - Google 2E
## Search Query Recommendation System

### Overview
This application provides two main functionalities:
1. **Masked Language Model Predictions**: Enter a query with `<mask>` to get top predictions based on our masked language model.
2. **GPT Model Text Generation**: Enter a prompt to get predictions from our GPT model.

### Features

#### Masked Language Model Predictions
- **Input**: User can enter a query containing the `<mask>` token.
- **Output**: The application returns the top 5 predictions for the masked token.
- **Usage**:
  - Enter a query in the format `e.g., 20 newtons equals how many <mask>?`.
  - Click the "Predict" button.
  - If the query does not contain `<mask>`, an error message is displayed.
  - The top 5 predictions are displayed as clickable links that open a Google search for the predicted text.

#### GPT Model Text Generation
- **Input**: User can enter a prompt.
- **Output**: The application generates and returns the top 5 sequences based on the prompt.
- **Usage**:
  - Enter a prompt in the format `e.g., Interesting facts about Egypt`.
  - Click the "Generate" button.
  - If the prompt is empty, an error message is displayed.
  - The top 5 generated texts are displayed as clickable links that open a Google search for the generated text.

### Model Directory Setup

Make sure you have the fine-tuned model directories `mlm_model_save` and `gpt_model_save2` in the project directory. If the models are located elsewhere, update the file path in the code accordingly.

For example, the following line in the code:
```python
save_dir = "/Users/harsita/Desktop/btt/btt-google-2e/demo-app/mlm_model_save"
```

Should be updated to reflect the correct path where mlm_model_save is located on your machine. Do this for both models.

### Run the Application
`streamlit run app.py`

Once the app is running, navigate to http://localhost:8501 in your browser to start using the tool.

### How It Works
1. Tab Switch: Choose which model you would like to get top queries from.

2. Enter Your Query: Example, picked MaskedLM model. The user inputs a sentence with a <mask> to indicate the missing word, such as:
20 newtons equals how many <mask>?

3. Submit Prediction: Upon clicking the "Predict" button, the app processes the query and generates the top 5 predictions for the <mask>.

4. View and Click Predictions: The predictions are displayed in individual styled boxes. Each prediction is clickable and will open a new tab with a Google search for the selected prediction.

5. Enter Your Prompt: Example, picked GPT model. The user inputs a prompt, such as:
Once upon a time

6. Submit Generation: Upon clicking the "Generate" button, the app processes the prompt and generates the top 5 sequences.

7. View and Click Generated Texts: The generated texts are displayed in individual styled boxes. Each generated text is clickable and will open a new tab with a Google search for the selected text.

### Dependencies
- `streamlit`
- `transformers`
- `tensorflow`
- `urllib`

### Example Usage
1. Masked Language Model Predictions:
- Input: 20 newtons equals how many <mask>?
- Output: Top 5 predictions with clickable links.

2. GPT Model Text Generation:
- Input: Once upon a time
- Output: Top 5 generated texts with clickable links.

### Notes
- Ensure that the <mask> token is included in the query for masked language model predictions.
- The generated texts are controlled by various parameters such as `top_k`, `top_p`, `temperature`, and `repetition_penalty` to ensure diversity and relevance.