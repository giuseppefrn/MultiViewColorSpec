import tensorflow as tf
import numpy as np

from skimage import color

# For normalized LAB
def delta_e_normalized(y_true, y_pred):
  loss = tf.sqrt(
      tf.square(y_true[:,0] * 100 - y_pred[:,0] * 100) +
      tf.square((y_true[:,1] * 240 - 120) - (y_pred[:,1] * 240 - 120)) +
      tf.square((y_true[:,2] * 240 - 120) - (y_pred[:,2] * 240 - 120))
    )
  return loss

#For no normalized LAB
def delta_e(y_true, y_pred):
  y_true = y_true.numpy().astype(np.float32)

  loss = tf.sqrt(
      tf.square(y_true[:,0] - y_pred[:,0]) +
      tf.square((y_true[:,1]) - (y_pred[:,1])) +
      tf.square((y_true[:,2]) - (y_pred[:,2]))
    )
  return loss

def delta_e_2000(y_true, y_pred):
  y_true = y_true.numpy().astype(np.float32)

  return color.deltaE_ciede2000(y_true, y_pred)