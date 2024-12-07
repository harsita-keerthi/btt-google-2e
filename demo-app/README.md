# [Demo App - Google 2E](https://drive.google.com/file/d/19itx49YZDUKfZklzAj0m22UMsGmbyxR2/view?usp=sharing)  
## Search Query Recommendation System  

### Overview  
This application provides an interactive platform for generating search query recommendations using advanced language models. Users can leverage three functionalities:  

1. **Masked Language Model Predictions**: Predict the most likely completions for queries with masked tokens.  
2. **GPT Model Text Generation**: Generate contextual sequences from prompts using a GPT model.  
3. **Combined Model Predictions**: Generate enhanced outputs by combining predictions from both Masked Language and GPT models.  

### Features  

#### **Masked Language Model Predictions**  
- **Input**: Enter a query containing the `<mask>` token.  
- **Output**: The top 5 predictions for the masked token.  
- **Usage**:  
  - Input a query in the format: `e.g., 20 newtons equals how many <mask>?`.  
  - Click the **"Predict"** button.  
  - If the query does not contain `<mask>`, an error message is displayed.  
  - Predictions are shown as clickable links that open Google search results for each prediction.  

#### **GPT Model Text Generation**  
- **Input**: Enter a text prompt.  
- **Output**: The top 5 generated sequences based on the prompt.  
- **Usage**:  
  - Input a prompt in the format: `e.g., Interesting facts about Egypt`.  
  - Click the **"Generate"** button.  
  - If the input is empty, an error message is displayed.  
  - Generated sequences are displayed as clickable links leading to Google search results.  

#### **Combined Model Predictions**  
- **Input**: Enter a text query containing `<mask>` or a generic prompt.  
- **Output**: Combined predictions from MaskedLM and GPT for a richer output.  
- **Usage**:  
  - Input a query in the desired format.  
  - Click the **"Generate Combined"** button.  
  - Predictions are displayed as clickable links leading to Google search results.  

### Directory Structure  

Ensure the fine-tuned model directories, `mlm_model_save` and `gpt_model_save2`, are present in the project directory. Update the paths in the code if they are located elsewhere.  

Example:
```python
save_dir = "/path/to/your/project/mlm_model_save"
```  

Update this for both model directories.  

### How to Run  

1. Launch the app with:  
   ```bash
   streamlit run app.py
   ```  

2. Open the application in your browser at [http://localhost:8501](http://localhost:8501).  

### How It Works  

#### **Step 1: Select a Model**  
Choose one of the tabs to use the desired functionality: **MaskedLM**, **GPT**, or **Combined**.  

#### **Step 2: Enter Your Input**  
- For MaskedLM, include `<mask>` in the query.  
- For GPT or Combined, enter a text prompt.  

#### **Step 3: Submit Your Query**  
- Click the **"Predict"**, **"Generate"**, or **"Generate Combined"** button.  

#### **Step 4: View Results**  
- Top 5 results are displayed as clickable styled boxes.  
- Clicking a result opens a Google search for that text in a new tab.  

### Dependencies  

- `streamlit`  
- `transformers`  
- `tensorflow`  
- `urllib`  

### Notes  

1. Ensure `<mask>` is included for MaskedLM predictions.  
2. Generated outputs are fine-tuned with hyperparameters such as `top_k`, `top_p`, `temperature`, and `repetition_penalty` for improved diversity and relevance.  
3. Model paths must be correctly configured for the application to run smoothly.   
