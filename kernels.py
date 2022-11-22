from prepare_dataset import *


def different_kernels_model(n=32, s=5):
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


# 32
def test_number_of_kernels():
    accs = []
    for n_kernels in (2 ** p for p in range(1, 6)):
        model = different_kernels_model(n=n_kernels)
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=2)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        accs.append(acc)
        print(f'Accuracy number-of-kernels:{n_kernels} model= {acc * 100.0}')
    print(accs)
    plt.figure()
    plt.title('Accuracy for different number of kernels')
    plt.xlabel('number of kernels')
    plt.ylabel('accuracy')
    plt.plot([2 ** p for p in range(1, 6)], accs, 'o-', color='red')
    plt.show()


# 5 * 5
def test_shape_of_kernels():
    accs = []
    for s_kernels in range(1, 6):
        model = different_kernels_model(s=s_kernels)
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=2)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        accs.append(acc)
        print(f'Accuracy for shape-of-kernels:{s_kernels}*{s_kernels} model= {acc * 100.0}')
    print(accs)
    plt.figure()
    plt.title('Accuracy for different shape of kernels')
    plt.xlabel('shape of kernels')
    plt.ylabel('accuracy')
    plt.plot([p for p in range(1, 6)], accs, 'o-', color='red')
    plt.show()


def visualize_filters():
    model = different_kernels_model()
    model.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test), verbose=2)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy= {acc * 100.0}')
    model.summary()
    filters, biases = model.layers[-5].get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    n_filters, index = 5, 1
    for i in range(n_filters):
        f = filters[:, :, :, i]
        for j in range(3):
            ax = plt.subplot(n_filters, 3, index)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:, :, j], cmap='gray')
            index += 1
    plt.show()


visualize_filters()
