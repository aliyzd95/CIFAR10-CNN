from prepare_dataset import *
from keras.models import load_model
from keras.layers import BatchNormalization

NUMBER_OF_EPOCHS = 100
BATCH_SIZE = 64
NUMBER_OF_KERNELS = 32  # 32-->64-->128
SHAPE_OF_KERNELS = 5  # 5*5
DROPOUT_RATE = 0.2


# define model
def define_model():
    model = Sequential()
    model.add(Conv2D(NUMBER_OF_KERNELS, (SHAPE_OF_KERNELS, SHAPE_OF_KERNELS), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(NUMBER_OF_KERNELS, (SHAPE_OF_KERNELS, SHAPE_OF_KERNELS), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Conv2D(NUMBER_OF_KERNELS * 2, (SHAPE_OF_KERNELS, SHAPE_OF_KERNELS), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(NUMBER_OF_KERNELS * 2, (SHAPE_OF_KERNELS, SHAPE_OF_KERNELS), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Conv2D(NUMBER_OF_KERNELS * 4, (SHAPE_OF_KERNELS, SHAPE_OF_KERNELS), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(NUMBER_OF_KERNELS * 4, (SHAPE_OF_KERNELS, SHAPE_OF_KERNELS), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(10, activation='softmax'))
    # opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# plot results of the evaluated model
def plot_results(history):
    # plot loss
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='red', label='test')
    plt.legend()
    # plot accuracy
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='blue')
    plt.plot(history.history['val_accuracy'], color='red')
    plt.show()


# evaluate model
def evaluate_model():
    # define model
    model = define_model()
    # fit model
    history = model.fit(X_train, y_train, epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test), verbose=2)
    # evaluate model
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    # save model to a file
    model.save('final_model.h5')
    print(f'Accuracy= {acc * 100.0}')
    # plot learning curves
    plot_results(history)


m = load_model('final_model_0.h5')
m.summary()
filters, biases = m.layers[-9].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, index = 5, 1
for i in range(n_filters):
    f = filters[:, :, :, i * 20]
    for j in range(3):
        ax = plt.subplot(n_filters, 3, index)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, j], cmap='gray')
        index += 1
plt.show()
