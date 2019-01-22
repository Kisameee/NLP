import os

import numpy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

from amazon.documents import Vectorizer
from amazon.documents.amazonReviewParser import AmazonReviewParser
from amazon.documents.recurrentNeuralNetwork import RecurrentNeuralNetwork
from amazon.data import DATA_DIR
from keras.utils import np_utils

############Charger les données################
print('Reading training data')
file= os.path.join(DATA_DIR, 'test.json')
documents = AmazonReviewParser().read_file(file)

#############Transformer en vecteurs de caractéristiques (feature vectors)######################
print('Create features')
vectorizer = Vectorizer(word_embedding_path=os.path.join(DATA_DIR, 'glove.txt'))
word, pos, shape = vectorizer.encode_features(documents)
labels = vectorizer.encode_annotations(documents)
print('Loaded {} data samples'.format(len(labels)))


#############Padding (Rembourrage)#############
print('Split training/validation')
max_length = 60
# --------------- Features ----------------
# 1. Split features to training and testing set
split = int(len(word)*80/100)
word_train, word_validation = word[0:split], word[split:]
pos_train, pos_validation = pos[0: split], pos[split:]
shape_train, shape_validation = shape[0: split], shape[split:]
# 2. Padd sequences
# deja fait

# Repeat for POS and Shape

# --------------- Labels -------------------
# 1. Convert to one-hot vectors
labels = [np_utils.to_categorical(y_group, num_classes=len(vectorizer.indexes)) for y_group in labels]
# 2. Split labels to training and test set
y_train, y_validation =  [labels[0: split]], [labels[split:]]
#########Entraînemer et Sauvegarder le modèle##############
print('Building network...')
model = RecurrentNeuralNetwork.build_classification(word_embeddings=vectorizer.word_embeddings,
                                            input_shape={'pos': (len(vectorizer.pos2index), 10),
                                                         'shape': (len(vectorizer.shapes), 2),
                                                         'max_length': max_length},
                                            units=100, dropout_rate=0.5)

print('Train...')
trained_model_name = 'ner_weights.h5'

# Callback that stops training based on the loss fuction
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Callback that saves the best model across epochs
saveBestModel = ModelCheckpoint(trained_model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

x_train = [word_train, pos_train, shape_train]
x_validation = [word_validation, pos_validation, shape_validation]

model.fit(x_train, y_train,
          validation_data=(x_validation, y_validation),
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