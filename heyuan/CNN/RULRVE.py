#!/usr/bin/env python
# coding: utf-8

# In[1]:


import utils_backup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Lambda, LSTM, RepeatVector, Bidirectional, Masking, Dropout


# In[ ]:


import sys
sys.path.append("")


# # Data

# In[2]:


dataset = "FD003"
# sensors to work with: T30, T50, P30, PS30, phi
sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
# windows length
sequence_length = 30
# smoothing intensity
alpha = 0.1
# max RUL
threshold = 125

x_train, y_train, x_val, y_val, x_test, y_test = utils_backup.get_data(dataset, sensors, sequence_length, alpha, threshold)


# # Model

# In[3]:


# Setup the network parameters:
timesteps = x_train.shape[1]
input_dim = x_train.shape[2]
intermediate_dim = 300
batch_size = 128
latent_dim = 2
epochs = 10
# used to mask values in sequences with less than 30 cycles until 30 is reached
masking_value = -99.


# In[4]:


class Sampling(keras.layers.Layer):
  """Uses (z_mean, sigma) to sample z, the vector encoding an engine trajetory."""
  def call(self, inputs):
    mu, sigma = inputs
    batch = tf.shape(mu)[0]
    dim = tf.shape(mu)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return mu + tf.exp(0.5 * sigma) * epsilon


# In[5]:


# ----------------------- Encoder -----------------------
inputs = Input(shape=(timesteps, input_dim,), name='encoder_input')

mask = Masking(mask_value=masking_value)(inputs)

# LSTM encoding
h = Bidirectional(LSTM(intermediate_dim))(mask) 

# VAE Z layer
mu = Dense(latent_dim)(h)
sigma = Dense(latent_dim)(h)

z = Sampling()([mu, sigma])

# Instantiate the encoder model:
encoder = keras.Model(inputs, [mu, sigma, z], name='encoder')
print(encoder.summary())
# -------------------------------------------------------


# In[6]:


# ----------------------- Regressor --------------------
reg_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_reg')
reg_intermediate = Dense(200, activation='tanh')(reg_latent_inputs)
reg_outputs = Dense(1, name='reg_output')(reg_intermediate)
# Instantiate the classifier model:
regressor = keras.Model(reg_latent_inputs, reg_outputs, name='regressor')
print(regressor.summary())
# -------------------------------------------------------


# In[7]:


# ----------------------- Decoder --------------------
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
h_decoded = RepeatVector(timesteps)(latent_inputs)
h_decoded = Bidirectional(LSTM(intermediate_dim, return_sequences=True))(h_decoded) 
# decoded layer
outputs = LSTM(input_dim, return_sequences=True)(h_decoded) 

# Instantiate the decoder model:
decoder = keras.Model(latent_inputs, outputs, name='decoder')
print(decoder.summary())
# -------------------------------------------------------


# In[8]:


class RVE(keras.Model):
    def __init__(self, encoder, regressor, decoder=None, **kwargs):
        super(RVE, self).__init__(**kwargs)
        self.encoder = encoder
        self.regressor = regressor
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.reg_loss_tracker = keras.metrics.Mean(name="reg_loss")
        self.decoder = decoder
        if self.decoder!=None:
          self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        

    @property
    def metrics(self):
      if self.decoder!=None:
        return [
            self.total_loss_tracker,
            self.kl_loss_tracker,
            self.reg_loss_tracker,
            self.reconstruction_loss_tracker
        ]
      else:
        return [
            self.total_loss_tracker,
            self.kl_loss_tracker,
            self.reg_loss_tracker,
        ]

    def train_step(self, data):
        x, target_x = data
        with tf.GradientTape() as tape:
            # kl loss
            mu, sigma, z = self.encoder(x)
            kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # Regressor
            reg_prediction = self.regressor(z)
            reg_loss = tf.reduce_mean(
                keras.losses.mse(target_x, reg_prediction)
            )
            # Reconstruction
            if self.decoder!=None:
              reconstruction = self.decoder(z)
              reconstruction_loss = tf.reduce_mean(
                  keras.losses.mse(x, reconstruction)
              )
              total_loss = kl_loss + reg_loss + reconstruction_loss
              self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            else:
              total_loss = kl_loss + reg_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "reg_loss": self.reg_loss_tracker.result(),
        }


    def test_step(self, data):
        x, target_x = data

        # kl loss
        mu, sigma, z = self.encoder(x)
        kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        # Regressor
        reg_prediction = self.regressor(z)
        reg_loss = tf.reduce_mean(
            keras.losses.mse(target_x, reg_prediction)
        )
        # Reconstruction
        if self.decoder!=None:
          reconstruction = self.decoder(z)
          reconstruction_loss = tf.reduce_mean(
              keras.losses.mse(x, reconstruction)
          )

          total_loss = kl_loss + reg_loss + reconstruction_loss
        else:
          total_loss = kl_loss + reg_loss

        return {
            "loss": total_loss,
            "kl_loss": kl_loss,
            "reg_loss": reg_loss,
        }


# In[10]:


rve = RVE(encoder, regressor)
lr_finder = utils_backup.LRFinder(rve)

rve.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0000001))

# Train a model with batch size 512 for 5 epochs
# with learning rate growing exponentially from 0.0000001 to 1
lr_finder.find(x_train, y_train, start_lr=0.000001, end_lr=0.1, batch_size=batch_size, epochs=5)

# Plot the loss, ignore 20 batches in the beginning and 5 in the end
lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)


# In[ ]:


rve = RVE(encoder, regressor)
rve.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
rve.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), batch_size=batch_size)


# In[ ]:


train_mu = utils_backup.viz_latent_space(rve.encoder, np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)))
test_mu = utils_backup.viz_latent_space(rve.encoder, x_test, y_test.clip(upper=threshold))
# Evaluate
y_hat_train = rve.regressor.predict(train_mu)
y_hat_test = rve.regressor.predict(test_mu)

utils_backup.evaluate(np.concatenate((y_train, y_val)), y_hat_train, 'train')
utils_backup.evaluate(y_test, y_hat_test, 'test')


# In[ ]:





# In[ ]:




