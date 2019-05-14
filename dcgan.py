import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, LeakyReLU
from keras.optimizers import Adam

def create_generator():
    generator = Sequential()

    generator.add(Dense(4 * 4 * 1024, activation="relu", input_shape=(100,)))
    generator.add(Reshape((4 , 4, 1024)))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(UpSampling2D())
    generator.add(Conv2D(512,kernel_size=5,  padding="same", activation="relu"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(UpSampling2D())
    generator.add(Conv2D(256,kernel_size=5,  padding="same", activation="relu"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(UpSampling2D())
    generator.add(Conv2D(128,kernel_size=5,  padding="same", activation="relu"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(UpSampling2D())
    generator.add(Conv2D(3,kernel_size=5, padding="same", activation="relu"))

    generator.summary()

    noise = Input(shape=(100,))
    img = generator(noise)

    return Model(noise,img)
    # kernel 5, stride 4


def create_discriminator():
    img_shape = (64,64,3)

    discriminator = Sequential()

    discriminator.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(Conv2D(128, kernel_size=3, strides=2, input_shape=img_shape, padding="same",activation="relu"))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(Conv2D(256, kernel_size=3, strides=2, input_shape=img_shape, padding="same",activation="relu"))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(BatchNormalization(momentum=0.8))

    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation="sigmoid"))

    discriminator.summary()

    img = Input(shape=img_shape)
    validity = discriminator(img)

    return Model(img, validity)


def train():
    generator = create_generator()
    discriminator = create_discriminator()

    #generator生成
    z = Input(shape=(100,))
    img = generator(z)

    #discriminator生成
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])


    valid = discriminator(img)


    # Generator学習用のCombined_modelを作る
    discriminator.trainable = False
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    combined.summary()


    # データ読み込み部
    X_train = np.load("test_X.npy")/255.
    Y_train = np.ones(X_train.shape[0])

    BATCH_SIZE = 25
    EPOCHS = 200000


    for epoch in range(EPOCHS):
        img_ids = np.random.randint(0,X_train.shape[0], BATCH_SIZE)
        imgs = X_train[img_ids]

        noise = np.random.uniform(-1,1, (BATCH_SIZE, 100))

        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, np.ones((BATCH_SIZE,1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((BATCH_SIZE,1)))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.uniform(-1,1, (BATCH_SIZE, 100))

        g_loss = combined.train_on_batch(noise, np.ones((BATCH_SIZE,1)))

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))






if __name__ == "__main__":

    train()
