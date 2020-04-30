from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import RMSprop
import keras
from keras import backend as K
import os
import hppi
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def encoder(input_img):

    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    return encoded


def decoder(encoded, input_dims):

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dims, activation='sigmoid')(decoded)

    return decoded


def fc(encoded, num_classes):
    flat = Flatten()(encoded)
    dense = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(dense)
    return out


def load_test_data(data_path):

    hppids = hppi.read_data_sets(data_path, one_hot=True)
    
    inp_dims = len(hppids.test.datas[0])

    test_datas = np.reshape(hppids.test.datas, (len(hppids.test.datas), 1, inp_dims))
    
    return test_datas, hppids.test.labels


def load_train_data(data_path):

    hppids = hppi.read_data_sets(data_path, one_hot=True)

    train_length, train_datas, train_labels, valid_length, valid_datas, valid_labels = hppids.train.shuffle().split(
        ratio=0.8)

    inp_dims = len(train_datas[0])

    train_datas = np.reshape(train_datas, (len(train_datas), 1, inp_dims))
    valid_datas = np.reshape(valid_datas, (len(valid_datas), 1, inp_dims))

    return train_length, train_datas, train_labels, valid_length, valid_datas, valid_labels, inp_dims


def train_autoencoder(input_img, train_datas, valid_datas):

    number_of_epochs = 20
    batch_size = 500

    autoencoder = Model(input_img, decoder(encoder(input_img), inp_dims))
    autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

    autoencoder.summary()

    autoencoder.fit(train_datas, train_datas, batch_size=batch_size, epochs=number_of_epochs,
                                        verbose=1, validation_data=(valid_datas, valid_datas))

    autoencoder.save_weights('autoencoder.h5')

    return autoencoder


def train_full_model(input_img, autoencoder, train_datas, valid_datas, train_labels, valid_labels):

    num_classes = 2

    encoded = encoder(input_img)
    full_model = Model(input_img, fc(encoded, num_classes))

    for l1, l2 in zip(full_model.layers[:4], autoencoder.layers[0:4]):
        l1.set_weights(l2.get_weights())

    for layer in full_model.layers[0:4]:
        layer.trainable = False

    full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                       metrics=['accuracy'])

    full_model.summary()

    full_model.fit(train_datas, train_labels, batch_size=64, epochs=20, verbose=1,
                                    validation_data=(valid_datas, valid_labels))

    full_model.save_weights('autoencoder_classification.h5')

    for layer in full_model.layers[0:4]:
        layer.trainable = True

    full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                       metrics=['accuracy'])

    classify_train = full_model.fit(train_datas, train_labels, batch_size=64, epochs=20, verbose=1,
                                    validation_data=(valid_datas, valid_labels))

    full_model.save_weights('classification_complete.h5')

    accuracy = classify_train.history['accuracy']
    val_accuracy = classify_train.history['val_accuracy']
    loss = classify_train.history['loss']
    val_loss = classify_train.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, label='Training accuracy')
    plt.plot(epochs, val_accuracy, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    return full_model


def test_full_model(full_model, test_datas, test_labels):

    y_pred = full_model.predict(test_datas, batch_size=64, verbose=1)
    y_pred_one_hot = K.one_hot(np.argmax(y_pred, axis=1), num_classes=2)

    test_accuracy = sum(np.argmax(test_labels, axis=1) == np.argmax(y_pred, axis=1)) / len(test_labels)

    print(classification_report(test_labels, y_pred_one_hot, target_names = ["Class 0", "Class 1"]))

    cf_matrix = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(y_pred, axis=1))

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0: .2f}%".format(value) for value in (cf_matrix.flatten() / np.sum(cf_matrix))]
    labels = [f"{v1}\n {v2}\n {v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap = 'Blues')

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return test_accuracy


def get_data_paths():

    other_species_ct = [r"/data/predict_PPI/C.elegan/bin/ct",
                        r"/data/predict_PPI/Drosophila/bin/ct",
                        r"/data/predict_PPI/E.coli/bin/ct",
                        r"/data/predict_PPI/Human/bin/ct"]

    other_species_ac = [r"/data/predict_PPI/C.elegan/bin/ac",
                        r"/data/predict_PPI/Drosophila/bin/ac",
                        r"/data/predict_PPI/E.coli/bin/ac",
                        r"/data/predict_PPI/Human/bin/ac"]

    benchmark_ac = [r"/data/03-ac-bin"]
    benchmark_ct = [r"/data/02-ct-bin"]

    return benchmark_ct
    # return benchmark_ac
    # return benchmark_ct + other_species_ct
    # return benchmark_ac + other_species_ac


if __name__ == "__main__":

    species_name = []
    test_accuracy = []

    for path in get_data_paths():
        data_path = os.getcwd() + path
        train_length, train_datas, train_labels, valid_length, valid_datas, valid_labels, inp_dims = load_train_data(data_path)
        test_datas, test_labels = load_test_data(data_path)

        input_img = Input(shape=(1, inp_dims))

        autoencoder  = train_autoencoder(input_img, train_datas, valid_datas)
        full_model = train_full_model(input_img, autoencoder, train_datas, valid_datas, train_labels, valid_labels)
        num_classes = 2

        encoded = encoder(input_img)

        full_model = Model(input_img, fc(encoded, num_classes))

        full_model.load_weights(os.path.join(os.getcwd(), 'classification_complete.h5'))

        accuracy = test_full_model(full_model, test_datas, test_labels)

        if data_path.split("/")[-1].endswith("bin"):
            species_name.append("Benchmark Dataset")
        else:
            species_name.append(data_path.split("/")[-3])
        test_accuracy.append(accuracy)

    plt.bar(species_name, test_accuracy, label='Test Accuracy', width=0.5)
    plt.title('Test Accuracy Summary')
    plt.legend()
    plt.show()









