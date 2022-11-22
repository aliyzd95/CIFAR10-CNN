from prepare_dataset import *


def dropout_model(n=32, s=5, p=0.2):
    model = Sequential()
    model.add(Conv2D(n, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(p))
    model.add(Conv2D(n * 2, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(p))
    model.add(Conv2D(n * 4, (s, s), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(p))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 0.9
def test_dropout():
    accs = []
    for keep_prop in np.arange(0.5, 1.0, 0.1):
        rate = round(1 - keep_prop, 1)
        model = dropout_model(p=rate)
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=0)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        accs.append(acc)
        print(f'Accuracy for rate:{rate} model= {acc * 100.0}')
    print(accs)
    plt.figure()
    plt.title('Accuracy for different shape of kernels')
    plt.xlabel('keep_prop value')
    plt.ylabel('accuracy')
    plt.plot(np.arange(0.5, 1.0, 0.1), accs, 'o-', color='red')
    plt.show()


test_dropout()