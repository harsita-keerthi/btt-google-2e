# [Break Through Tech: Google 2E](https://docs.google.com/presentation/d/120Zn7rPPT29uXdwBKD9wdJlObjj7Nk3GWQwikFf93TM/edit?usp=sharing)

## Search Query Recommendation System
This project focuses on developing an intelligent query autocompletion system using advanced natural language processing techniques. The system combines GPT-2 and Masked Language Models to generate accurate and contextually relevant query suggestions.

<img width="971" alt="Screenshot 2024-12-07 at 9 19 54 AM" src="https://github.com/user-attachments/assets/a25ba2ad-e68b-458d-81f3-0585e9972338">

## Project Overview

The project aims to enhance user experience in search interfaces by providing smart query autocompletion. It utilizes two main models:

1. Fine-tuned GPT-2 model
2. Masked Language Model (DistilRoBERTa-base)

These models are combined to overcome individual limitations and produce more accurate and complete query suggestions.

## Process

### Data Preparation

- Used a Well Formedness Dataset containing query ratings and sentences
- Performed data cleaning, exploration, and preprocessing
- Applied filtering and class balancing techniques

### Model Implementation

#### GPT-2 Model
- Utilized Hugging Face's transformer library
- Implemented tokenization and fine-tuning processes
- Optimized training parameters

#### Masked Language Model (MLM)
- Used DistilRoBERTa-base architecture
- Implemented tokenization and masking techniques
- Trained the model using TensorFlow

### Model Combination

To address limitations of individual models:
1. Generate initial output using GPT-2
2. Append a masked token to the GPT-2 output
3. Use MLM to predict the masked token and complete the query

## User Interface

- Developed a Streamlit web application
- Implemented tabs for different functionalities:
  - Masked Language Model predictions
  - GPT-2 text generation
  - Combined model predictions

## Conclusions

The project successfully demonstrates:
- Effective query autocompletion using combined NLP models
- A user-friendly interface for interacting with the system
- Potential for further improvements and personalization

## Future Work

- Implement user authentication
- Develop personalized query suggestions based on user history
- Introduce real-time suggestions as users type
- Implement query ranking based on popularity

## Acknowledgements
Our Team:
- Harsita Keerthikanth
- Alena Chao
- Sofia Nguyen
- Vanessa Huynh
- Erica Xue
Special thanks to challenge advisors Kanay Gupta and Smrithika Appaiah, and course support Mako Ohara.
