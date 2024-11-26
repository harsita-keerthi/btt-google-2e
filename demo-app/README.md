# Demo App - Search Query Recommendation System

This an application that shows Google 2E's model that implements a search query recommendation system with a a web-based application built with **Streamlit**. It leverages a fine-tuned language model to predict missing words in a sentence. It generates predictions based on a user query containing a placeholder `<mask>` and displays the top predictions in clickable boxes. When clicked, the predictions open a new tab to perform a Google search for that prediction.

## Features

- **Mask Filler**: Enter a query with a `<mask>` to get the top word predictions from a fine-tuned language model.
- **Top Predictions**: View the top 5 predictions for the `<mask>` in your query.
- **Clickable Predictions**: Each prediction is displayed in a styled clickable box. When clicked, it opens a Google search for that prediction in a new tab.
- **Customizable**: The app allows for easy adjustments to the model and can be adapted for other NLP tasks.

## Tech Stack

- **Streamlit**: For building the web interface.
- **Hugging Face Transformers**: For the `fill-mask` model pipeline.
- **Python**: The backend language for logic and computations.
- **CSS**: For custom styling of the page elements.
- **Google Search URL Encoding**: To open Google search results for the predictions.

## Installation

To run this project on your local machine, follow these steps:

### Prerequisites

- Python 3.7 or higher
- `pip` (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/your-username/google-2e-masking-prediction.git
cd google-2e-masking-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

```bash
pip install streamlit transformers
```

### Model Directory Setup

Make sure you have the fine-tuned model directory `mlm_model_save` in the project directory. If the model is located elsewhere, update the file path in the code accordingly.

For example, the following line in the code:
```python
save_dir = "/Users/harsita/Desktop/btt/btt-google-2e/demo-app/mlm_model_save"
```

Should be updated to reflect the correct path where `mlm_model_save` is located on your machine.

### Run the Application

```bash
streamlit run app.py
```

Once the app is running, navigate to `http://localhost:8501` in your browser to start using the tool.

## How It Works

1. **Enter Your Query**: The user inputs a sentence with a `<mask>` to indicate the missing word, such as:  
   `20 newtons equals how many <mask>?`
   
2. **Submit Prediction**: Upon clicking the "Predict" button, the app processes the query and generates the top 5 predictions for the `<mask>`.

3. **View and Click Predictions**: The predictions are displayed in individual styled boxes. Each prediction is clickable and will open a new tab with a Google search for the selected prediction.