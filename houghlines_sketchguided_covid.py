import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

from joblib import load
from sketchguidedFuncs import *

class_trained = sys.argv[1]

num_folds = 10
kfold_shuffle = False
test_fold = int(sys.argv[2])

images_path = '../ultrasound_images/croped_images_gray_smaller/'
hough_path = '../images_edges_detection/croped_images_gray_smaller/PATH_TO_HOUGH_LINES/'
current_path = './houghlines_sketchguided/kfold10/Fold' + str(test_fold) + '/'


# define the discriminator
def define_discriminator(in_shape=(128,128,1)):
  # weight initialization
  init = tf.random_normal_initializer(0., 0.02)
  # source image input (Hough lines)
  in_src_image = tf.keras.layers.Input(shape=in_shape)
  # target image input (original imagem from which the lines are extracted)
  in_target_image = tf.keras.layers.Input(shape=in_shape)
  # concatenating both inputs
  merged = tf.keras.layers.Concatenate()([in_src_image, in_target_image])
  # 112
  d = tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
  d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
  # 56
  d = tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = tf.keras.layers.BatchNormalization()(d)
  d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
  # 28
  d = tf.keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = tf.keras.layers.BatchNormalization()(d)
  d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
  # 14
  d = tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = tf.keras.layers.BatchNormalization()(d)
  d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
  # second last output layer
  d = tf.keras.layers.Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
  d = tf.keras.layers.BatchNormalization()(d)
  d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
  # patch output
  d = tf.keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
  patch_out = tf.keras.layers.Activation('sigmoid')(d)
  # define model
  model = tf.keras.Model([in_src_image, in_target_image], patch_out, name='discriminator')
  
  return model


# define the standalone generator model
def define_generator(in_shape=(128,128,1)):
  # weight initialization
  init = tf.random_normal_initializer(0., 0.02)
  # define model
  in_image = tf.keras.layers.Input(shape=in_shape)
  # encoder model: C64-C128-C256
  e1 = encoder_block(in_image, 64, strides=2, batchnorm=False)
  e2 = encoder_block(e1, 128, strides=2)
  e3 = encoder_block(e2, 256, strides=2)
  # bottleneck, no batch norm and relu
  # residual blocks (10 or 15 of those)
  b = residual_block(e3, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  b = residual_block(b, filters=256, kernel_size=(3,3))
  # decoder model: C256-C128-C64
  d5 = decoder_block(b, 256, strides=2, dropout=False)
  d6 = decoder_block(d5, 128, strides=2, dropout=False)
  d7 = decoder_block(d6, 64, strides=2, dropout=False)
  # output
  g = tf.keras.layers.Conv2DTranspose(1, (3,3), strides=(1,1), padding='same', kernel_initializer=init)(d7)
  out_image = tf.keras.layers.Activation('tanh')(g)
  # define model
  model = tf.keras.Model(in_image, out_image, name='generator')

  return model


real_images = load(images_path + 'images_' + class_trained)['images']
hough_images = load(hough_path + 'isfcanny5_houghlines_edges_' + class_trained)

real_images = np.expand_dims(real_images, axis=-1)  # all channels are equal since images are in grayscale
hough_images = np.expand_dims(hough_images, axis=-1)

real_images = (real_images - 127.5)/127.5
hough_images = (hough_images - 127.5)/127.5


# Separating fold K-Fold
kf = KFold(n_splits=num_folds, shuffle=kfold_shuffle)

folds_ind = [test_index for _, test_index in kf.split(real_images)]

test_real_images = real_images[folds_ind[test_fold]]
test_hough_images = hough_images[folds_ind[test_fold]]

if test_fold == 9:
  val_fold = 0
else:
  val_fold = test_fold + 1
  
val_real_images = real_images[folds_ind[val_fold]]
val_hough_images = hough_images[folds_ind[val_fold]]

train_real_images = real_images[np.concatenate([x for j,x in enumerate(folds_ind) if j not in [test_fold, val_fold]])]
train_hough_images = hough_images[np.concatenate([x for j,x in enumerate(folds_ind) if j not in [test_fold, val_fold]])]


gen_img_path = current_path + class_trained + '/generated_images/'

# MODEL PARAMETERS
IMG_SHAPE = (128, 128, 1)
BATCH_SIZE = 16

# Set the number of epochs for trainining.
epochs = 2000

# Defining optimizers
generator_optimizer = Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)

discriminator = define_discriminator(in_shape=IMG_SHAPE)
generator = define_generator(in_shape=IMG_SHAPE)

# Instantiate the WGAN model.
sketchguided = SketchGuided(
    discriminator=discriminator,
    generator=generator
    )

# Compile the WGAN model.
sketchguided.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss, 
    wasserstein_loss=wasserstein_loss
)


# Instantiate the customer `GANMonitor` Keras callback.
cbk = GANMonitor(val_hough_images, val_real_images, num_img=6, gen_img_path=gen_img_path)


# Callbacks to save best model and history
checkpoint_path = current_path + class_trained + '/checkpoint/'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

STEPS_PER_EPOCH = train_real_images.shape[0]/BATCH_SIZE
checkpoint_cbk = ModelCheckpoint(filepath=checkpoint_path + 'epoch-{epoch:04d}.ckpt',
                                 monitor='d_loss',
                                 save_freq=int(500*STEPS_PER_EPOCH),
                                 mode='max',
                                 save_weights_only=True,
                                 verbose=1)

logfilename = current_path + class_trained + '/history.csv'
history_cbk = MyCSVLogger(logfilename)


# Start training the model.
#sketchguided.load_weights(current_path + class_trained + '/checkpoint/epoch-0398.ckpt')
history = sketchguided.fit(train_hough_images, train_real_images,
                      validation_data=(val_hough_images, val_real_images),
                      batch_size=BATCH_SIZE, epochs=epochs,
                      callbacks=[cbk, checkpoint_cbk, history_cbk],
                      #initial_epoch=398)
                      )