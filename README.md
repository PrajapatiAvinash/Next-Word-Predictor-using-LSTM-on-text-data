# Deep Learning Text Classification with LSTM

This project implements a text classification model using Long Short-Term Memory (LSTM) networks. The model classifies text data into 282 unique classes.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
    - [Tokenization](#tokenization)
    - [Padding](#padding)
3. [Model Architecture](#model-architecture)
    - [Embedding Layer](#embedding-layer)
    - [LSTM Layer](#lstm-layer)
    - [Dense Layer](#dense-layer)
4. [Training the Model](#training-the-model)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
7. [Requirements](#requirements)
8. [Conclusion](#conclusion)

## Introduction
This project focuses on text classification using an LSTM model. We have a dataset containing text data with 282 unique words. The goal is to classify the text into one of the 282 classes using an LSTM-based neural network.

## Data Preprocessing
### Tokenization
Tokenization is the process of converting text into tokens, which are numerical representations of the words in the text. This is achieved using the `Tokenizer` class from the Keras library. Each unique word in the dataset is assigned a unique integer.

### Padding
After tokenization, the sequences are padded to ensure that all input sequences are of the same length. This is important for training the LSTM model, as it requires fixed-length input sequences.

## Model Architecture
The model is built using Keras' `Sequential` API. It consists of the following layers:

### Embedding Layer
The Embedding layer is the first layer in the model. It converts the integer-encoded words into dense vectors of fixed size. The embedding layer helps the model understand the relationships between words by mapping each word to a high-dimensional vector.

```python
from keras.layers import Embedding

# Adding Embedding layer
embedding_layer = Embedding(input_dim=282, output_dim=64, input_length=max_sequence_length)
```

### LSTM Layer
The LSTM layer is the core of the model. LSTM networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies. They are well-suited for sequence prediction problems such as text classification.

```python
from keras.layers import LSTM

# Adding LSTM layer
lstm_layer = LSTM(units=100, return_sequences=False)
```

### Dense Layer
The Dense layer is a fully connected layer that performs the classification. It has 282 units, corresponding to the 282 unique classes in the dataset, with a softmax activation function to output probability distributions over the classes.

```python
from keras.layers import Dense

# Adding Dense layer
dense_layer = Dense(units=282, activation='softmax')
```

## Training the Model
The model is compiled using the categorical cross-entropy loss function and the Adam optimizer. The model is trained on the preprocessed text data.

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

## Evaluation
After training, the model is evaluated on a test set to determine its accuracy.

```python
# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')
```

## Usage
To use the model for prediction, simply pass new text data through the same preprocessing steps (tokenization and padding) and use the `predict` method.

```python
predictions = model.predict(new_text_data)
```

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas

Install the required packages using:

```bash
pip install tensorflow keras numpy pandas
```

## Conclusion
This project demonstrates the implementation of an LSTM-based text classification model. The model preprocesses the text data, builds an LSTM network with embedding, LSTM, and dense layers, and trains the model to classify text into 282 unique classes.
