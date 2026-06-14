import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):

    def __init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1,
                                                         beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer, loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: tf.keras.losses.MeanSquaredError()(x, tf.ones(
            x.shape)) + tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape))
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1,
                                                             beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer, loss=discriminator.loss)

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument):

        # 1. Diskriminatorun öyrədilməsi (disc_iter dəfə təkrarlanır)
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Real və fake nümunələrin əldə edilməsi
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                # Diskriminatorun hər iki nümunə üzərindən proqnozları
                real_preds = self.discriminator(real_samples, training=True)
                fake_preds = self.discriminator(fake_samples, training=True)

                # Diskriminatorun loss-unun hesablanması
                discr_loss = self.discriminator.loss(real_preds, fake_preds)

            # Qradiyentlərin tapılması və optimizer-ə tətbiqi
            discr_gradients = tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(discr_gradients,
                                                             self.discriminator.trainable_variables))

        # 2. Generatorun öyrədilməsi
        with tf.GradientTape() as tape:
            # Yeni fake nümunələrin əldə edilməsi
            fake_samples = self.get_fake_sample(training=True)

            # Diskriminatorun fake nümunələr üzərindən proqnozu
            gen_preds = self.discriminator(fake_samples, training=False)

            # Generatorun loss-unun hesablanması
            gen_loss = self.generator.loss(gen_preds)

        # Qradiyentlərin tapılması və optimizer-ə tətbiqi
        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
