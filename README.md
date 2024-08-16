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

### Data Distribution
- Total samples: 5791
- Positive sentiment: 3641 (62.87%)
- Negative sentiment: 2150 (37.13%)

This distribution shows that the dataset is slightly imbalanced, with more positive sentiment samples than negative ones.

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
   - Input dimension: 7947 (vocabulary size)
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
1. **Data Split**: 80% training (4632 samples), 20% testing (1159 samples)
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

## Results and Evaluation

### Training History
- The model was trained for 10 epochs
- Best validation accuracy: 0.76509 (achieved in the final epoch)
- Training accuracy reached 0.9549 in the final epoch

### Model Performance
- **Test Loss**: 0.5132
- **Test Accuracy**: 0.7938 (79.38%)

### Training Visualization
The training history plot shows:
- Rapid increase in training accuracy over the first few epochs
- More gradual improvement in validation accuracy
- Some fluctuation in validation accuracy, indicating potential for overfitting

### Word Clouds
- **Positive Sentiment**: Prominent words include "buy", "long", "good", "stock", "call", "breakout"
- **Negative Sentiment**: Prominent words include "short", "put", "low", "break", "drop", "bearish"

These word clouds provide insights into the vocabulary associated with positive and negative sentiments in stock market discussions.

## Conclusions
- The model achieves a good accuracy of 79.38% on the test set, demonstrating its effectiveness in sentiment classification for stock market-related text.
- There's a noticeable gap between training and validation accuracy, suggesting some overfitting. Future improvements could focus on regularization techniques.
- The word clouds reveal intuitive associations between certain terms and sentiment polarities in stock market context.

## Model Persistence
- The best model is saved as 'best_model.hdf5'
- Tokenizer and Label Encoder are saved as pickle files for future use in predictions

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is open-source and available under the MIT License.
