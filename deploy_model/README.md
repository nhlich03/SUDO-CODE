# Sentiment LSTM API (IMDB Movie Reviews)

This project deploys a basic sentiment analysis model (LSTM with TensorFlow/Keras) as a RESTful API using FastAPI.

The model was trained on the IMDB Dataset of 50K Movie Reviews from Kaggle. The API receives a piece of English text (a movie review or any sentence) and returns whether the sentiment is positive or negative, together with a confidence score.

## 1. Project structure

Project folder:

    deploy_model/
        main.py
        requirements.txt
        model/
            sentiment_lstm_imdb.keras
            tokenizer_imdb.pkl

Description:

1. main.py  
   FastAPI application file. It loads the trained model and tokenizer, defines the API endpoints, and handles prediction.

2. requirements.txt  
   List of Python dependencies required to run the API.

3. model/sentiment_lstm_imdb.keras  
   Trained TensorFlow/Keras LSTM model for binary sentiment classification.

4. model/tokenizer_imdb.pkl  
   Fitted Keras Tokenizer used during training to convert text to integer sequences.

## 2. Model training (summary)

The model was trained separately (e.g., in a Kaggle notebook) with the following pipeline:

1. Dataset: IMDB Dataset of 50K Movie Reviews (binary labels: “positive” and “negative”).  
2. Preprocessing:  
   - Map labels: positive → 1, negative → 0.  
   - Split into training and test sets (e.g., 80/20).  
   - Tokenize text with Keras Tokenizer(num_words=20000, oov_token="<OOV>").  
   - Pad sequences to fixed length, for example MAX_SEQ_LEN = 200.  
3. Model architecture:  
   - Embedding layer with dimension 128.  
   - LSTM layer with 128 units.  
   - Dropout layer with rate 0.5.  
   - Dense output layer with sigmoid activation for binary classification.  
4. Training:  
   - Loss: binary_crossentropy.  
   - Optimizer: adam.  
   - Metrics: accuracy.  
   - Batch size: 128.  
   - Number of epochs: around 5–10, depending on hardware.

After training:

- The model was saved as: sentiment_lstm_imdb.keras  
- The tokenizer was saved as: tokenizer_imdb.pkl  

These two files are placed in the model/ directory for deployment.

## 3. Requirements

The project uses Python 3 and the following main libraries:

- FastAPI  
- Uvicorn  
- TensorFlow  
- NumPy  
- Pydantic  

All dependencies are listed in requirements.txt:

    fastapi
    uvicorn[standard]
    tensorflow
    numpy<2
    pydantic

## 4. Installation and setup

1. Navigate to the project directory:

    cd path/to/deploy_model

2. (Recommended) Create and activate a virtual environment.

   On Windows:

       python -m venv venv
       venv\Scripts\activate

   On macOS / Linux:

       python3 -m venv venv
       source venv/bin/activate

3. Install the required packages:

       pip install -r requirements.txt

4. Ensure that the directory model/ exists and contains:

       sentiment_lstm_imdb.keras
       tokenizer_imdb.pkl

The paths inside main.py assume this exact structure.

## 5. Running the API

From inside the deploy_model folder (with the virtual environment activated):

    uvicorn main:app --reload

By default, the API will be available at:

    http://127.0.0.1:8000

Interactive API documentation (Swagger UI) is available at:

    http://127.0.0.1:8000/docs

Redoc documentation is available at:

    http://127.0.0.1:8000/redoc

## 6. API endpoints

### 6.1 Health check

- Path: /  
- Method: GET  

Response example:

    {
      "message": "Sentiment LSTM API is running"
    }

### 6.2 Predict sentiment

- Path: /predict  
- Method: POST  

Request body (JSON):

    {
      "text": "The movie was awesome and I really enjoyed it!"
    }

Response body (JSON):

    {
      "text": "The movie was awesome and I really enjoyed it!",
      "pred_label": 1,
      "pred_sentiment": "positive",
      "confidence": 0.98
    }

Explanation:

- text: Original input text.  
- pred_label: Numerical label for sentiment. 1 for positive, 0 for negative.  
- pred_sentiment: Human-readable label, either "positive" or "negative".  
- confidence: Confidence score in the predicted class, between 0 and 1.

## 7. Example usage

### 7.1 Using Swagger UI

1. Open the browser at http://127.0.0.1:8000/docs  
2. Click on the POST /predict endpoint  
3. Click “Try it out”  
4. Replace the default JSON with your own text  
5. Click “Execute” to see the prediction

### 7.2 Using curl (Linux / macOS)

    curl -X POST "http://127.0.0.1:8000/predict"       -H "Content-Type: application/json"       -d '{"text": "The movie was boring and terrible."}'

### 7.3 Using curl (Windows PowerShell)

    curl -X POST "http://127.0.0.1:8000/predict" `
      -H "Content-Type: application/json" `
      -d "{ \"text\": \"The movie was boring and terrible.\" }"

## 8. Implementation details

Key implementation points in main.py:

1. The model and tokenizer are loaded once at startup using load_model and pickle.load.  
2. Input text is converted to integer sequences using the fitted tokenizer.  
3. Sequences are padded to a fixed length MAX_SEQ_LEN (must match training configuration).  
4. The model outputs a probability p between 0 and 1.  
5. If p ≥ 0.5, the sentiment is considered positive; otherwise negative.  
6. Confidence is defined as p for the positive class or 1 − p for the negative class.