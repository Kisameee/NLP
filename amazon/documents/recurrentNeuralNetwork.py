from keras.models import load_model, save_model
from keras.layers import concatenate
from keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, Dense, TimeDistributed, Flatten
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Flatten, Dropout, Input, LSTM, Bidirectional, TimeDistributed
from keras.datasets import imdb
from sklearn import metrics


class RecurrentNeuralNetwork():
    def __init__(self, model=None):
        self._model = model

    def load_weights(self, *args, **kwargs):
        return self._model.load_weights(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self._model.fit(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        return self._model.fit_generator(*args, **kwargs)

    def predict_generator(self, *args, **kwargs):
        return self._model.predict_generator(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)

    @staticmethod
    def probas_to_classes(proba):
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    @classmethod
    def build_sequence(cls, word_embeddings, input_shape: dict, out_shape: int, units=100, dropout_rate=0.5):

        word_input = Input(shape=(None,), dtype='int32', name='word_input')

        weights = word_embeddings.syn0
        word_embeddings = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1],
                                    weights=[weights], name="word_embeddings_layer", trainable=False,
                                    mask_zero=True)(word_input)

        pos_input = Input(shape=(None,), dtype='int32', name='pos_input')
        pos_embeddings = Embedding(input_shape['pos'][0], input_shape['pos'][1], name='pos_embeddings_layer',
                                   mask_zero=True)(pos_input)

        shape_input = Input(shape=(None,), dtype='int32', name='shape_input')
        shape_embeddings = Embedding(input_shape['shape'][0], input_shape['shape'][1], name='shape_embeddings_layer',
                                     mask_zero=True)(shape_input)

        merged_input = concatenate([word_embeddings, pos_embeddings, shape_embeddings], axis=-1)

        #################################

        bilstm = Bidirectional(LSTM(100, activation='tanh', return_sequences=True), name='bi-lstm')(merged_input)

        lstm = LSTM(100, activation='tanh', name='lstm')(bilstm)

        lstm_layer = Dropout(0.5, name='second_dropout')(lstm)

        output = Dense(1, activation='sigmoid', name='output')(lstm_layer)

        # Build and compile model
        model = Model(inputs=[word_input], outputs=output)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())

        ###################################
        print(model.summary())
        return RecurrentNeuralNetwork(model)

    @classmethod
    def build_classification(cls, word_embeddings, input_shape: dict, out_shape: int, units=100,
                             dropout_rate=0.5):

        word_input = Input(shape=(None,), dtype='int32', name='word_input')

        weights = word_embeddings.syn0
        word_embeddings = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1],
                                    weights=[weights], name="word_embeddings_layer", trainable=False,
                                    mask_zero=True)(word_input)

        pos_input = Input(shape=(None,), dtype='int32', name='pos_input')
        pos_embeddings = Embedding(input_shape['pos'][0], input_shape['pos'][1], name='pos_embeddings_layer',
                                   mask_zero=True)(pos_input)

        shape_input = Input(shape=(None,), dtype='int32', name='shape_input')
        shape_embeddings = Embedding(input_shape['shape'][0], input_shape['shape'][1], name='shape_embeddings_layer',
                                     mask_zero=True)(shape_input)

        merged_input = concatenate([word_embeddings, pos_embeddings, shape_embeddings], axis=-1)

        # Build the rest of the model here

        print(model.summary())
        return Recurrent(model)

    @classmethod
    def load(cls, filename):
        return RecurrentNeuralNetwork(load_model(filename))

    def save(self, filename):
        save_model(self._model, filename)
        self._model.save(filename)
