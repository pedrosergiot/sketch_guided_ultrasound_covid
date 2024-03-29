__all__ = ["SketchGuided", "MyCSVLogger", "GANMonitor", 
           "generator_loss", "discriminator_loss", "wasserstein_loss",
           "residual_block", "encoder_block", "decoder_block"]

import os
import csv
import tensorflow as tf
import pandas as pd
import numpy as np


class SketchGuided(tf.keras.Model):
  def __init__(self,
                discriminator,
                generator
                ):

    super(SketchGuided, self).__init__()
    self.discriminator = discriminator
    self.generator = generator

  def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, wasserstein_loss):
    super(SketchGuided, self).compile()
    self.d_optimizer = d_optimizer
    self.g_optimizer = g_optimizer
    self.d_loss_fn = d_loss_fn
    self.g_loss_fn = g_loss_fn
    self.wasserstein_loss = wasserstein_loss


  #def train_step(self, real_images):
  def train_step(self, data):
    input_image, target = data

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      # Generate fake images
      gen_output = self.generator(input_image, training=True)
      # Get the logits for the fake images
      disc_generated_output = self.discriminator([input_image, gen_output], training=True)
      # Get the logits for the real images
      disc_real_output = self.discriminator([input_image, target], training=True)

      gen_total_loss, gen_gan_loss, gen_l1_loss = self.g_loss_fn(disc_generated_output, gen_output, target)
      disc_loss = self.d_loss_fn(disc_real_output, disc_generated_output)

    # Calculate the gradients for generator and discriminator
    generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

    # Apply the gradientes to the optimizer
    self.g_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
    self.d_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

    wasser_loss = self.wasserstein_loss(target, gen_output)
    
    return {"disc_loss": disc_loss, "gen_total_loss": gen_total_loss, 
            "gen_l1_loss": gen_l1_loss, "wasser_loss": wasser_loss}


  def test_step(self, data):
    input_image, target = data
    generated_images = self.generator(input_image)
    # Get the logits for generated images
    disc_generated_output = self.discriminator([input_image, generated_images])
    # Get the logits for the real images
    disc_real_output = self.discriminator([input_image, target])

    disc_loss = self.d_loss_fn(disc_real_output, disc_generated_output)

    wasser_loss = self.wasserstein_loss(target, generated_images)

    return {"disc_loss": disc_loss, "wasser_loss": wasser_loss}


# Keras callback to periodically save generated images
class GANMonitor(tf.keras.callbacks.Callback):
  def __init__(self, houghs, targets, num_img=6, gen_img_path='generated_images/'):
    self.houghs = houghs
    self.targets = targets
    self.num_img = num_img
    self.gen_img_path = gen_img_path

  def on_epoch_end(self, epoch, logs=None):
    random_inds = np.random.randint(0, tf.shape(self.houghs)[0], self.num_img)

    generated_images = self.model.generator(self.houghs[random_inds])
    real_images = self.targets[random_inds]

    generated_images = (generated_images * 127.5) + 127.5
    real_images = (real_images * 127.5) + 127.5

    for i in range(self.num_img):
      img = generated_images[i].numpy()
      img = tf.keras.preprocessing.image.array_to_img(img)
      path_img = self.gen_img_path + "img_{i}_{epoch}_generated.png".format(i=i, epoch=epoch)
      dir_img = os.path.dirname(path_img)
      if not os.path.exists(dir_img):
          os.makedirs(dir_img)
      img.save(path_img)

      img = (self.targets[random_inds])[i]
      img = tf.keras.preprocessing.image.array_to_img(img)
      path_img = self.gen_img_path + "img_{i}_{epoch}_real.png".format(i=i, epoch=epoch)
      img.save(path_img)


class MyCSVLogger(tf.keras.callbacks.Callback):

  def __init__(self, filename):
    self.filename = tf.compat.path_to_str(filename)

  def on_train_begin(self, epoch, logs=None):
    if not os.path.exists(self.filename):
      self.csv_file = open(self.filename, 'a')
      self.writer = csv.writer(self.csv_file)
      self.writer.writerow(['disc_loss', 'gen_total_loss', 'gen_l1_loss', 'val_disc_loss'])
      self.csv_file.close()

  def on_epoch_end(self, epoch, logs=None):
    pd.DataFrame(logs, index=[epoch]).to_csv(self.filename, mode = 'a', header=False, index=False)



def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


def wasserstein_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def residual_block(x: tf.Tensor, filters: int, kernel_size: int = 3) -> tf.Tensor:
	y = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same")(x)
	y = tf.keras.layers.ReLU()(y)
	y = tf.keras.layers.BatchNormalization()(y)
 
	y = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same")(y)
	out = tf.keras.layers.Add()([x, y]) 
	out = tf.keras.layers.ReLU()(out)
	out = tf.keras.layers.BatchNormalization()(out)
	return out


# define an encoder block
def encoder_block(layer_in, n_filters, filter=3, strides=2, batchnorm=True, dropout=False):
	# weight initialization
	init = tf.random_normal_initializer(0., 0.02)
	# add downsampling layer
	g = tf.keras.layers.Conv2D(n_filters, filter, strides=strides, padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = tf.keras.layers.BatchNormalization()(g, training=True)
	# leaky relu activation
	if dropout:
		g = tf.keras.layers.Dropout(0.5)(g, training=True)
	# relu activation
	g = tf.keras.layers.ReLU()(g)
	return g


# define a decoder block
def decoder_block(layer_in, n_filters, filter=3, strides=2, batchnorm=True, dropout=False):
	# weight initialization
	init = tf.random_normal_initializer(0., 0.02)
	# add upsampling layer
	g = tf.keras.layers.Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	if batchnorm:
		g = tf.keras.layers.BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = tf.keras.layers.Dropout(0.5)(g, training=True)
	# relu activation
	g = tf.keras.layers.ReLU()(g)
	return g