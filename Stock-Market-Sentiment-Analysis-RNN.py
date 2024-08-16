import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend to avoid showing plots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import re
import random
import pickle

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)

# Print TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Constants
MEDIA_DIR = 'media'  # Change this if needed, e.g., './media'
TRAIN_SIZE = 0.8
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 300
LR = 1e-3
BATCH_SIZE = 1024
EPOCHS = 10

# Data Loading and Preprocessing
def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(file_path, encoding='latin', header=None)
    df.drop(index=df.index[0], axis=0, inplace=True)
    df.columns = ['text', 'sentiment']
    return df

def preprocess_text(text, stem=False):
    """
    Preprocess the text data.
    """
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def encode_sentiment(df):
    """
    Encode sentiment labels.
    """
    lab_to_sentiment = {"-1": "Negative", "1": "Positive"}
    df.sentiment = df.sentiment.apply(lambda x: lab_to_sentiment.get(x, "Unknown"))
    return df

# Visualization Functions
def plot_sentiment_distribution(df):
    """
    Plot the distribution of sentiments and save the image.
    """
    val_count = df.sentiment.value_counts()
    plt.figure(figsize=(8,4))
    plt.bar(val_count.index, val_count.values)
    plt.title("Sentiment Data Distribution")
    plt.savefig(os.path.join(MEDIA_DIR, 'sentiment_distribution.png'))
    plt.close()
    print(f"Sentiment distribution plot saved in {MEDIA_DIR}/sentiment_distribution.png")

def generate_word_cloud(df, sentiment):
    """
    Generate a word cloud for a given sentiment and save the image.
    """
    plt.figure(figsize=(20,20))
    wc = WordCloud(max_words=2000, width=1600, height=800).generate(" ".join(df[df.sentiment == sentiment].text))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f"Word Cloud for {sentiment} Sentiment")
    plt.axis('off')
    plt.savefig(os.path.join(MEDIA_DIR, f'word_cloud_{sentiment.lower()}.png'))
    plt.close()
    print(f"Word cloud for {sentiment} sentiment saved in {MEDIA_DIR}/word_cloud_{sentiment.lower()}.png")

# Data Preparation for Model
def prepare_data_for_model(df):
    """
    Prepare data for model training.
    """
    train_data, test_data = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=7)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data.text)
    
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.text), maxlen=MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.text), maxlen=MAX_SEQUENCE_LENGTH)
    
    encoder = LabelEncoder()
    encoder.fit(train_data.sentiment.to_list())
    
    y_train = encoder.transform(train_data.sentiment.to_list()).reshape(-1,1)
    y_test = encoder.transform(test_data.sentiment.to_list()).reshape(-1,1)
    
    return x_train, y_train, x_test, y_test, tokenizer, encoder

# Model Definition
def create_model(vocab_size, embedding_dim, input_length):
    """
    Create and compile the RNN model.
    """
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main Execution
if __name__ == "__main__":
    # Create media directory if it doesn't exist
    os.makedirs(MEDIA_DIR, exist_ok=True)

    # Load and preprocess data
    df = load_data('stock_data.csv')
    df = encode_sentiment(df)
    df.text = df.text.apply(preprocess_text)
    
    # Visualizations
    plot_sentiment_distribution(df)
    generate_word_cloud(df, 'Positive')
    generate_word_cloud(df, 'Negative')
    
    # Prepare data for model
    x_train, y_train, x_test, y_test, tokenizer, encoder = prepare_data_for_model(df)
    
    # Print shapes of prepared data
    print("Training X Shape:", x_train.shape)
    print("Testing X Shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    
    # Print vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary Size:", vocab_size)
    
    # Create and train the model
    model = create_model(vocab_size, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)
    
    # Define model checkpoint
    checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    # Train the model
    history = model.fit(x_train, y_train, 
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS, 
                        validation_split=0.1,
                        callbacks=[checkpoint])
    
    # Evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    # Plot training history
    plt.figure(figsize=(12,6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(MEDIA_DIR, 'model_accuracy.png'))
    plt.close()
    print(f"Model accuracy plot saved in {MEDIA_DIR}/model_accuracy.png")
    
    # Save the tokenizer and encoder for future use
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('encoder.pickle', 'wb') as handle:
        pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Training complete. Model saved as 'best_model.hdf5'. Tokenizer and encoder saved as pickle files.")
    print(f"All visualizations saved in {MEDIA_DIR}")