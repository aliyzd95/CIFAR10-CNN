from prepare_dataset import *
from keras import layers
from keras import backend


class MinPooling2D(MaxPooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs):
        super(MaxPooling2D, self).__init__(pool_size, strides, padding, data_format, **kwargs)

    def pooling_function(self, pool_size, strides, padding, data_format):
        return -backend.pool2d(-self, pool_size, strides, padding, data_format, pool_mode='max')


def max_pooling_model(n=32, s=5):
    model = Sequential()
    model.add(Conv2D(n, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(n * 2, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(n * 4, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def min_pooling_model(n=32, s=5):
    model = Sequential()
    model.add(Conv2D(n, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MinPooling2D((2, 2)))
    model.add(Conv2D(n * 2, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MinPooling2D((2, 2)))
    model.add(Conv2D(n * 4, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MinPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def mean_pooling_model(n=32, s=5):
    model = Sequential()
    model.add(Conv2D(n, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Conv2D(n * 2, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Conv2D(n * 4, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# MaxPooling
def test_downsampling():
    accs = []
    # define models with different pooling approaches
    models = [max_pooling_model, min_pooling_model, mean_pooling_model]
    models_names = ['max_pooling', 'min_pooling', 'mean_pooling']
    for i, m in enumerate(models):
        model = m()
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=0)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        accs.append(acc)
        print(f'Accuracy for {models_names[i]} model= {acc * 100.0}')
    print(accs)


test_downsampling()