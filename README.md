# Stock Market Sentiment Analysis using RNN

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![NLTK](https://img.shields.io/badge/NLTK-3.x-green.svg)](https://www.nltk.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.x-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-1.x-yellow.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.x-blue.svg)](https://numpy.org/)

## Project Overview
This project implements a Recurrent Neural Network (RNN) to perform sentiment analysis on stock market-related text data. The model classifies text as either positive or negative sentiment, which can be useful for financial market analysis and prediction.

## Table of Contents
1. [Technologies and Frameworks](#technologies-and-frameworks)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training Process](#training-process)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Visualization Techniques](#visualization-techniques)

## Technologies and Frameworks
- **TensorFlow 2.x & Keras**: Core deep learning framework for building and training the RNN model.
- **NLTK 3.5+**: Used for text preprocessing, including tokenization and stop word removal.
- **Scikit-learn 0.22+**: Utilized for data splitting, label encoding, and evaluation metrics.
- **Pandas 1.0+ & NumPy 1.18+**: Used for data manipulation and numerical operations.
- **Matplotlib 3.0+**: Employed for data visualization and plotting results.
- **WordCloud 1.8+**: Used to generate word cloud visualizations of the text data.

## Project Structure
```
stock_sentiment_analysis/
│
├── stock_sentiment_analysis.py  # Main script
├── requirements.txt             # Project dependencies
├── stock_data.csv               # Input dataset
├── best_model.hdf5              # Saved best model (generated during training)
├── tokenizer.pickle             # Saved tokenizer (generated after training)
└── encoder.pickle               # Saved label encoder (generated after training)
```

## Dataset
The project uses a stock market sentiment dataset (`stock_data.csv`) with two columns:
- `text`: The stock market-related text
- `sentiment`: The sentiment label (-1 for Negative, 1 for Positive)

## Data Preprocessing
1. **Text Cleaning**: 
   - Regular expression: `@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+`
   - Removes special characters, URLs, and numbers
   - Converts text to lowercase
2. **Tokenization**: Splits text into individual words
3. **Stop Word Removal**: Uses NLTK's English stop words list
4. **Stemming**: Implements SnowballStemmer for English
5. **Sequence Padding**: Pads sequences to MAX_SEQUENCE_LENGTH (30)

## Model Architecture
The model uses an RNN architecture with the following components:
1. **Embedding Layer**: 
   - Input dimension: vocab_size (determined dynamically)
   - Output dimension: 300 (EMBEDDING_DIM)
   - Input length: 30 (MAX_SEQUENCE_LENGTH)
2. **LSTM Layer**: 
   - 100 units
   - Dropout: 0.2
   - Recurrent Dropout: 0.2
3. **Dense Layer**: 
   - 1 unit (output layer)
   - Activation: sigmoid

## Training Process
1. **Data Split**: 80% training, 20% testing
2. **Tokenization**: Using Keras Tokenizer
3. **Label Encoding**: Using sklearn's LabelEncoder
4. **Model Compilation**: 
   - Loss function: binary_crossentropy
   - Optimizer: adam
   - Metrics: accuracy
5. **Training**: 
   - Batch size: 1024
   - Epochs: 10
   - Validation split: 0.1
   - Uses ModelCheckpoint to save the best model

## Evaluation Metrics
- Accuracy
- Loss
- (Additional metrics like Precision, Recall, and F1-Score can be implemented)

## Visualization Techniques
1. **Sentiment Distribution**: Bar plot using matplotlib
2. **Word Clouds**: Separate visualizations for positive and negative sentiments
3. **Training History**: Plot of accuracy and loss over epochs

## Usage
Run the main script:
```
python stock_sentiment_analysis.py
```
This will perform data preprocessing, model training, evaluation, and generate visualizations.

## Model Persistence
- The best model is saved as 'best_model.hdf5'
- Tokenizer and Label Encoder are saved as pickle files for future use in predictions
