#!/usr/bin/env python3
""" This module defines the Simple_GAN class. """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """ This class defines a simple GAN model. """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        Constructor for the Simple_GAN class.
        Args:
            generator: The generator model.
            discriminator: The discriminator model.
            latent_generator: A function that generates latent vectors.
            real_examples: A tensor with real examples.
            batch_size: The batch size for training.
            disc_iter: The number of iterations to train the discriminator.
            learning_rate: The learning rate for the optimizer
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # define the generator loss and optimizer
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        # define the discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape))
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        This method generates fake samples using the generator model.
        Args:
            size: The number of samples to generate.
            training: A boolean indicating if the model is training.
        Returns:
            A tensor with the generated samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        This method returns real samples from the dataset.
        Args:
            size: The number of samples to return.
        Returns:
            A tensor with the real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        This method trains the GAN model for one step.
        Args:
            useless_argument: A useless argument.
        Returns:
            A dictionary with the losses of the generator and discriminator.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Get real and fake samples
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                # Evaluate the discriminator on real and fake samples
                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)

                # Calculate the discriminator loss
                discr_loss = self.discriminator.loss(real_output, fake_output)

            # Update discriminator weights
            discr_gradients = tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            # Get fake samples
            fake_samples = self.get_fake_sample(training=True)

            # Evaluate the discriminator on fake samples
            fake_output = self.discriminator(fake_samples, training=False)

            # Calculate the generator loss
            gen_loss = self.generator.loss(fake_output)

        # Update generator weights
        gen_gradients = tape.gradient(gen_loss,
                                      self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
