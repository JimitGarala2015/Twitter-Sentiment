!pip install keras_preprocessing
!pip install vaderSentiment
!pip install fasttext
!pip install numpy==1.26.4
#!pip install numpy==1.26.4 --force-reinstall

import pandas as pd
import numpy as np
import tensorflow as tf
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Bidirectional, Dense, SimpleRNN, Dropout, Flatten, GlobalMaxPooling1D, MaxPooling1D, Multiply, Lambda
from tensorflow.keras.layers import Attention, Lambda, TimeDistributed, Multiply, Permute, Activation, Softmax
from tensorflow.keras import layers, models, optimizers, backend as K
from tensorflow.keras import backend as K
from keras.optimizers import Adam
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import fasttext.util
import matplotlib.pyplot as plt
import seaborn as sns

print(np.__version__)

# Download resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

# Preprocess text
def preprocess_text(text):
    if text is None:
        return ''
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions (@usernames)
    text = re.sub(r'@\w+', '', text)
    # Remove special characters, numbers, and symbols
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Join tokens back into a single string
    return ' '.join(tokens)

# Add VADER sentiment scores
def add_sentiment_scores(df, text_column):
    analyzer = SentimentIntensityAnalyzer()
    scores = df[text_column].apply(lambda x: analyzer.polarity_scores(x))
    df['compound'] = scores.apply(lambda score_dict: score_dict['compound'])
    df['neg'] = scores.apply(lambda score_dict: score_dict['neg'])
    df['neu'] = scores.apply(lambda score_dict: score_dict['neu'])
    df['pos'] = scores.apply(lambda score_dict: score_dict['pos'])
    return df

# Load FastText
def load_fasttext_embeddings():
    fasttext.util.download_model('en', if_exists='ignore')
    ft_model = fasttext.load_model('cc.en.300.bin')
    return ft_model

def get_embedding_matrix(tokenizer, ft_model, vocab_size, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, index in tokenizer.word_index.items():
        if index < vocab_size:
            embedding_vector = ft_model.get_word_vector(word)
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

# Define RNN model
def build_rnn_model(vocab_size, embedding_dim, input_length, embedding_matrix):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=input_length, trainable=False),
        SimpleRNN(300, return_sequences=True),
        SimpleRNN(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Similarly, define LSTM, GRU, Conv1D, BiLSTM + Conv1D models

# Define GRU model
def build_gru_model(vocab_size, embedding_dim, input_length, embedding_matrix):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=input_length, trainable=False),
        GRU(200, return_sequences=True),
        GRU(100),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define LSTM model
def build_lstm_model(vocab_size, embedding_dim, input_length, embedding_matrix):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=input_length, trainable=False),
        LSTM(300, return_sequences=True),
        LSTM(100),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define CONV1D model
def build_conv1d_model(vocab_size, embedding_dim, input_length, embedding_matrix):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=input_length, trainable=False),
        Conv1D(64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define Bi-LSTM+CONV1D model
def build_bilstm_conv1d_model(vocab_size, embedding_dim, input_length, embedding_matrix):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=input_length, trainable=False),
        Conv1D(32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(200, return_sequences=False)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define RCNN model
def build_rcnn_model(vocab_size, embedding_dim, input_length, embedding_matrix):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=input_length, trainable=False),
        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        SimpleRNN(300, return_sequences=True),
        SimpleRNN(64),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define SENN model
def build_senn_model(vocab_size, embedding_dim, input_length, embedding_matrix):
    # Input layer
    input_layer = Input(shape=(input_length,))

    # Embedding layer
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                 input_length=input_length, trainable=False)(input_layer)

    # Explanation module (e.g., extracting key features)
    explanation = Dense(64, activation='relu', name="Explanation_Module")(embedding_layer)

    # Coefficient module (weights for the explanations)
    coefficients = Dense(64, activation='softmax', name="Coefficient_Module")(embedding_layer)

    # Combine explanations with coefficients
    combined = Multiply(name="Feature_Combination")([explanation, coefficients])
    flattened = Flatten()(combined)

    # Output module
    output = Dense(1, activation='sigmoid', name="Output_Module")(flattened)

    # Build and compile the model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Custom Squash activation for Capsule Networks
def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

# Custom Capsule Layer
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.num_capsules * self.dim_capsule),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        # Calculate u_hat
        u_hat = K.dot(inputs, self.kernel)
        u_hat = K.reshape(u_hat, (-1, inputs.shape[1], self.num_capsules, self.dim_capsule))

        # Initialize the routing logits (b)
        b = K.zeros_like(u_hat[:, :, :, 0])  # Shape: [batch_size, num_inputs, num_capsules]

        for i in range(self.routings):
            # Compute the coupling coefficients (c) using softmax
            c = tf.nn.softmax(b, axis=2)  # Normalize over the num_capsules axis
            c = tf.expand_dims(c, -1)  # Shape: [batch_size, num_inputs, num_capsules, 1]

            # Compute s by summing the weighted predictions
            s = K.sum(c * u_hat, axis=1)  # Sum over the num_inputs axis

            # Apply squash activation to get v
            v = squash(s)

            # Update the logits (b) for the next iteration
            if i < self.routings - 1:
                b += K.sum(u_hat * K.expand_dims(v, 1), axis=-1)

        return v

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsules, self.dim_capsule)

def build_capsnet_model(vocab_size, embedding_dim, input_length, embedding_matrix):
    inputs = layers.Input(shape=(input_length,))
    embedding = layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                  input_length=input_length, trainable=False)(inputs)

    primary_caps = layers.Conv1D(128, 9, strides=1, activation='relu')(embedding)
    primary_caps = layers.Reshape((-1, 8))(primary_caps)
    primary_caps = layers.Lambda(squash)(primary_caps)

    caps_layer = CapsuleLayer(num_capsules=10, dim_capsule=16, routings=3)(primary_caps)
    caps_output = layers.Flatten()(caps_layer)  # Flatten capsule outputs

    output = layers.Dense(1, activation='sigmoid')(caps_output)
    model = models.Model(inputs=inputs, outputs=output)

    model.compile(optimizer=optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def attention_layer(inputs):
    """
    Implements a trainable attention mechanism using predefined layers.
    """
    # Alignment scores
    alignment_scores = Dense(1, activation='tanh')(inputs)
    alignment_scores = Flatten()(alignment_scores)

    # Softmax to generate attention weights
    attention_weights = Softmax()(alignment_scores)

    # Expand dimensions for broadcasting
    attention_weights = Lambda(lambda x: K.expand_dims(x, axis=-1))(attention_weights)

    # Apply attention weights to inputs
    weighted_inputs = Multiply()([inputs, attention_weights])

    # Sum the weighted inputs along the time dimension
    output = Lambda(lambda x: K.sum(x, axis=1))(weighted_inputs)
    return output

def build_han_model(vocab_size, embedding_dim, input_length, embedding_matrix):
    # Word-level encoder
    word_input = Input(shape=(input_length,), name="Word_Input")
    word_embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                input_length=input_length, trainable=False)(word_input)
    word_encoder = Bidirectional(GRU(100, return_sequences=True))(word_embedding)
    word_attention = attention_layer(word_encoder)

    # Output layer
    output = Dense(1, activation='sigmoid', name="Output_Layer")(word_attention)

    # Build and compile the model
    model = Model(inputs=word_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Train and evaluate models
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return y_pred, history, accuracy, precision, recall, f1

def plot_confusion_matrix(y_true, y_pred, class_names):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the matrix (optional)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

class_names = ['0', '1']

df = pd.read_csv('Bitcoin_tweets_dataset_2.csv', engine = 'python', encoding = 'latin-1')

df = df[['text']]

# Display the top 5 rows
print("Top 5 rows:")
print(df.head(5))

# Display the bottom 5 rows
print("\nBottom 5 rows:")
print(df.tail(5))

df['clean_text'] = df['text'].apply(preprocess_text)

# Display the top 5 rows
print("Top 5 rows:")
print(df.head(5))

# Display the bottom 5 rows
print("\nBottom 5 rows:")
print(df.tail(5))

df = add_sentiment_scores(df, 'clean_text')

# Display the top 5 rows
print("Top 5 rows:")
print(df.head(5))

# Display the bottom 5 rows
print("\nBottom 5 rows:")
print(df.tail(5))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
max_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_len)
y = (df['compound'] > 0).astype(int)
#y = df['compound'].apply(lambda x: 0 if x < 0 else (1 if x == 0 else 2)).values
#y_one_hot = to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

embedding_dim = 300
vocab_size = len(tokenizer.word_index) + 1

ft_model = load_fasttext_embeddings()

embedding_matrix = get_embedding_matrix(tokenizer, ft_model, vocab_size, embedding_dim)

rnn_model = build_rnn_model(vocab_size, embedding_dim, max_len, embedding_matrix)

gru_model = build_gru_model(vocab_size, embedding_dim, max_len, embedding_matrix)

lstm_model = build_lstm_model(vocab_size, embedding_dim, max_len, embedding_matrix)

conv1d_model = build_conv1d_model(vocab_size, embedding_dim, max_len, embedding_matrix)

bilstm_conv1d_model = build_bilstm_conv1d_model(vocab_size, embedding_dim, max_len, embedding_matrix)

rcnn_model = build_rcnn_model(vocab_size, embedding_dim, max_len, embedding_matrix)

senn_model = build_senn_model(vocab_size, embedding_dim, max_len, embedding_matrix)

capsnet_model = build_capsnet_model(vocab_size, embedding_dim, max_len, embedding_matrix)

han_model = build_han_model(vocab_size, embedding_dim, max_len, embedding_matrix)

y_pred, history, accuracy, precision, recall, f1 = train_and_evaluate_model(rnn_model, X_train, y_train, X_test, y_test)

y_pred, history, accuracy, precision, recall, f1 = train_and_evaluate_model(gru_model, X_train, y_train, X_test, y_test)

accuracy, precision, recall, f1 = train_and_evaluate_model(lstm_model, X_train, y_train, X_test, y_test)

accuracy, precision, recall, f1 = train_and_evaluate_model(conv1d_model, X_train, y_train, X_test, y_test)

accuracy, precision, recall, f1 = train_and_evaluate_model(bilstm_conv1d_model, X_train, y_train, X_test, y_test)

y_pred, history, accuracy, precision, recall, f1 = train_and_evaluate_model(rcnn_model, X_train, y_train, X_test, y_test)

accuracy, precision, recall, f1 = train_and_evaluate_model(senn_model, X_train, y_train, X_test, y_test)

accuracy, precision, recall, f1 = train_and_evaluate_model(capsnet_model, X_train, y_train, X_test, y_test)

accuracy, precision, recall, f1 = train_and_evaluate_model(han_model, X_train, y_train, X_test, y_test)

# Extract loss values
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Plot the loss vs epochs graph
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, label='Training Loss', color= 'blue')
plt.plot(epochs, val_loss, label='Validation Loss', color= 'orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss vs. Epochs')
plt.legend()
plt.grid()
plt.show()

# Extract accuracy values
train_accuracy = history.history['accuracy']  # Training accuracy
val_accuracy = history.history.get('val_accuracy')  # Validation accuracy (optional)
epochs = range(1, len(train_accuracy) + 1)

# Plot accuracy vs epochs
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_accuracy, label='Training Accuracy', color='blue')
if val_accuracy:
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs. Epochs')
plt.legend()
plt.grid()
plt.show()

# Convert one-hot encoded labels to integer labels
#y_test = y_test.argmax(axis=1)  # Convert to [0, 1, 2]
#y_pred = y_pred.argmax(axis=1)  # Convert to [0, 1, 1]

plot_confusion_matrix(y_test, y_pred, class_names)

# Example data: Model names and their accuracies
model_names = ['RNN', 'GRU', 'LSTM', 'CONV1D', 'Bi-LSTM+CONV1D', 'RCNN', 'SENN', 'Capsnet', 'HAN']
accuracies = [0.70, 0.98, 0.97, 0.96, 0.96, 0.95, 0.95, 0.94, 0.97]  # Replace with actual accuracies

# Plot a bar graph
plt.figure(figsize=(12, 10))
sns.barplot(x=model_names, y=accuracies, palette='viridis')

# Add annotations
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f"{acc:.2f}", ha='center', va='bottom', fontsize=10)

# Customize the plot
plt.title('Comparison of Model Accuracies', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1)  # Accuracy is typically between 0 and 1
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

print(f"RNN Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

print(f"GRU Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

print(f"LSTM Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

print(f"CONV1D Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

print(f"Bi-LSTM+CONV1D Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

print(f"RCNN Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

print(f"SENN Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

print(f"CapsNet Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

print(f"HAN Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")