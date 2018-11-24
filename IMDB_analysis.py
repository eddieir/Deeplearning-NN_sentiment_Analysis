from keras.layers import Embedding, LSTM, Dense, Dropout
from keras import Sequential
from keras.preprocessing import sequence
import keras
from keras.datasets import imdb

# Set the vocabulary size and load in training and test data.
vocabulary_size = 500000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
print('Loaded dataset with {} training samples, {} test samples'.format(
    len(X_train), len(X_test)))

print('---review---')
print(X_train[6])
print('---label---')
print(y_train[6])

word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print('---review with words---')
print([id2word.get(i, ' ') for i in X_train[6]])
print('---label---')
print(y_train[6])

# Maximum review length and minimum review length.
print('Maximum review length: {}'.format(
    len(max((X_train + X_test), key=len))))
print('Minimum review length: {}'.format(
    len(min((X_test + X_test), key=len))))

# Pad sequences
# In order to feed this data into our RNN, all input documents must have the same length.
#  We will limit the maximum review length to max_words by truncating longer reviews and padding shorter reviews with a null value (0). We can accomplish this using the pad_sequences() function in Keras.
#  For now, set max_words to 5000.

max_words = 5000
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


# ---------------Design the RNN model for sentiment analysis ---------------------

embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# now let's train and evaluate the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch_size = 64
num_epochs = 5

X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid),
          batch_size=batch_size, epochs=num_epochs)
