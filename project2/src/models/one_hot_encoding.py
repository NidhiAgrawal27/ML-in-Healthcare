from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Embedding
from tensorflow.keras.callbacks import EarlyStopping


def sentence_one_hot_encoding(dataset, vocab_size, max_sentence_length):
    sentences = []
    for data in dataset.data:
        sentences.append(data['sentence'])

    oneHotencoded_sentences = [
        one_hot(sentence, vocab_size) for sentence in sentences
    ]
    # Padding shorter sentences
    X = pad_sequences(oneHotencoded_sentences,
                      maxlen=max_sentence_length,
                      padding='post')

    return X


def get_one_hot_model(vocab_size, embeded_vector_size, max_sentence_length):
    model = Sequential()
    model.add(
        Embedding(vocab_size,
                  embeded_vector_size,
                  input_length=max_sentence_length,
                  name="embedding"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    return model


def compile_fit(model, X, Y):
    model.compile(optimizer='adam',
                  loss='SparseCategoricalCrossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    model.fit(X,
              Y,
              validation_split=0.2,
              epochs=50,
              batch_size=128,
              callbacks=[
                  EarlyStopping(monitor='val_loss',
                                patience=3,
                                restore_best_weights=True)
              ])
    return model
