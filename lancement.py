############Charger les données################
print('Reading training data')
documents = YourParser().read_file('/Path/to/trainingdata')

#############Transformer en vecteurs de caractéristiques (feature vectors)######################

print('Create features')
vectorizer = Vectorizer(word_embedding_path='/Path/to/embeddings file')
word, pos, shape = vectorizer.encode_features(documents)
labels = vectorizer.encode_annotations(documents)
print('Loaded {} data samples'.format(len(features)))


#############Padding (Rembourrage)#############

from keras.utils import np_utils

print('Split training/validation')
max_length = 60
# --------------- Features ----------------
# 1. Split features to training and testing set
word_train, word_validation = # split 80% and 20%
# 2. Padd sequences
word_train = = sequence.pad_sequences(word_train, maxlen=max_length)
word_validation = sequence.pad_sequences(word_validation, maxlen=max_length)
# Repeat for POS and Shape

# --------------- Labels -------------------
# 1. Convert to one-hot vectors
labels = [np_utils.to_categorical(y_group, num_classes=len(vectorizer.labels)) for y_group in labels]
# 2. Split labels to training and test set
y_train, y_validation = # split 80% and 20%
# 3. (only for sequence tagging) Pad sequences
y_train =  sequence.pad_sequences(y_train, maxlen=max_length)
y_validation =  sequence.pad_sequences(y_validation, maxlen=max_length)


#########Entraînemer et Sauvegarder le modèle##############

from .yourpackage import RecurrentNeuralNetwork

print('Building network...')
model = RecurrentNeuralNetwork.build_sequence(word_embeddings=vectorizer.word_embeddings,
                                      input_shape={'pos': (len(vectorizer.pos2index), 10),
                                                   'shape': (len(vectorizer.shape2index), 2)},
                                      out_shape=len(vectorizer.labels),
                                      units=100, dropout_rate=0.5)
# or
model = RecurrentNeuralNetwork.build_classification(word_embeddings=vectorizer.word_embeddings,
                                            input_shape={'pos': (len(vectorizer.pos2index), 10),
                                                         'shape': (len(vectorizer.shape2index), 2),
                                                         'max_length': max_length},
                                            out_shape=len(vectorizer.labels),
                                            units=100, dropout_rate=0.5)

print('Train...')
trained_model_name = 'ner_weights.h5'

# Callback that stops training based on the loss fuction
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Callback that saves the best model across epochs
saveBestModel = ModelCheckpoint(trained_model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.fit([word_train, pos_train, shape_train], y_train,
          validation_data=([word_validation, pos_validation, shape_validation], y_validation),
          batch_size=32,  epochs=10, callbacks=[saveBestModel, early_stopping])

# Load the best weights in the model
model.load_weights(trained_model_name)

# Save the complete model
model.save('rnn.h5')






######VALIDATION########

# Use the test data: Unpadded feature vectors + unpaded and numerical (not one-hot vectors) labels

y_prediction, y_validation = [], []
# For each sample (one at a time)
    # Predict labels and convert from probabilities to classes
    # RecurrentNeuralNetwork.probas_to_classes()

print(classification_report(y_validation, y_prediction))