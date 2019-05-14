import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, LeakyReLU, Concatenate
from keras.optimizers import Adam


def create_encoder_block(input, filter):
    x = LeakyReLU(0.2)(input)
    x = Conv2D(filter, (3,3), strides=(2,2), padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    return x

def create_decoder_block(input, filter, concate_block):
    x = Activation('relu')(input)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(filter, (3,3), padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    x = Concatenate(axis=-1)([x, concate_block])
    return x

def create_generator():
    generator = Sequential()

    pix2pix_input = Input((64,64,3))
    encoder = Conv2D(64, (3,3), strides=(2,2), padding="same")(pix2pix_input)

    # encoder 1 layer
    encoder_block_1 = create_encoder_block(encoder, 128)

    # encoder 2 layer
    encoder_block_2 = create_encoder_block(encoder_block_1, 256)

    # encoder 3 layer
    encoder_block_3 = create_encoder_block(encoder_block_2, 512)

    # decoder 1 layer
    decoder_block_1 = create_decoder_block(encoder_block_3, 256, encoder_block_2)

    # decoder 2 layer
    decoder_block_2 = create_decoder_block(decoder_block_1, 128, encoder_block_1)

    # decoder 3 layer
    decoder_block_3 = create_decoder_block(decoder_block_2, 64, encoder)

    generator = Activation('relu')(decoder_block_3)
    generator = UpSampling2D(size=(2,2))(generator)
    generator = Conv2D(3, (3, 3), padding="same")(generator)
    generator = Activation('tanh')(generator)

    generator = Model(inputs=[pix2pix_input], outputs=[generator])
    generator.summary()

    return generator



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
    z = Input(shape=(64,64,3))
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

        gen_imgs = generator.predict(imgs)

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((BATCH_SIZE,1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((BATCH_SIZE,1)))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.uniform(-1,1, (BATCH_SIZE, 100))
        discriminator.trainable = False
        g_loss = combined.train_on_batch(imgs, np.ones((BATCH_SIZE,1)))

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))


if __name__ == "__main__":
    train()
