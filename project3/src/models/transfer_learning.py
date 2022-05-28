import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping


def get_transfer_model(input_shape):

    transfer_model = VGG16(input_shape = input_shape, weights='imagenet', include_top=False)

    # Freeze last few layers
    for layer in transfer_model.layers[:15]:
        layer.trainable = False

    for i, layer in enumerate(transfer_model.layers):
        print(i, layer.name, layer.trainable)

    x = transfer_model.output
    x = Flatten()(x)
    x = Dense(512,activation='relu')(x)
    x = Dense(512,activation='relu')(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(1,activation='sigmoid')(x)

    model = Model(inputs = transfer_model.input, outputs=x)
    model.summary()

    return model


def model_fit(model, X, y):

    model.compile(optimizer=Adam(learning_rate=0.0001, decay=1e-1),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    def scheduler(epoch, lr):
        if epoch < 5: return lr
        else: return lr * np.exp(-0.2) # lr - (lr*.02)

    lr_sched = LearningRateScheduler(scheduler)
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 7, restore_best_weights=True)

    model.fit(X, y, batch_size=128, epochs=100, validation_split=0.2, callbacks = [early_stop, lr_sched])
    
    return model

