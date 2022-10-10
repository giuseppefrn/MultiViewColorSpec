import tensorflow as tf

class ColorNet(tf.keras.Model):
    def __init__(self,):
      super(ColorNet, self).__init__()

      self.conv_1 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', padding='same')
      self.batch_norm_1 = tf.keras.layers.BatchNormalization()
      self.pool_1 = tf.keras.layers.MaxPool2D()

      self.conv_2 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', padding='same')
      self.batch_norm_2 = tf.keras.layers.BatchNormalization()
      self.pool_2 = tf.keras.layers.MaxPool2D()

      self.conv_3 = tf.keras.layers.Conv2D(256,(3,3), activation='relu', padding='same')
      self.batch_norm_3 = tf.keras.layers.BatchNormalization()
      self.pool_3 = tf.keras.layers.MaxPool2D()

      self.global_pool = tf.keras.layers.GlobalMaxPool2D()

      self.add = tf.keras.layers.Add()
      self.dense_1 = tf.keras.layers.Dense(128, activation='relu')
      self.last_dense = tf.keras.layers.Dense(3)


    def forward(self, inputs):
      x = self.conv_1(inputs)
      x = self.batch_norm_1(x)
      x = self.pool_1(x)

      x = self.conv_2(x)
      x = self.batch_norm_2(x)
      x = self.pool_2(x)

      x = self.conv_3(x)
      x = self.batch_norm_3(x)
      x = self.pool_3(x)

      x = self.global_pool(x)
      return x

    def call(self, inputs):
      xs = []
      for i in inputs:
        x = self.forward(i)
        xs.append(x)

      x = self.add(xs)
      x = self.dense_1(x)
      x = self.last_dense(x)
      return x

class SingleViewColorNet(tf.keras.Model):
    def __init__(self,):
      super(SingleViewColorNet, self).__init__()

      self.conv_1 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', padding='same')
      self.batch_norm_1 = tf.keras.layers.BatchNormalization()
      self.pool_1 = tf.keras.layers.MaxPool2D()

      self.conv_2 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', padding='same')
      self.batch_norm_2 = tf.keras.layers.BatchNormalization()
      self.pool_2 = tf.keras.layers.MaxPool2D()

      self.conv_3 = tf.keras.layers.Conv2D(256,(3,3), activation='relu', padding='same')
      self.batch_norm_3 = tf.keras.layers.BatchNormalization()
      self.pool_3 = tf.keras.layers.MaxPool2D()

      self.global_pool = tf.keras.layers.GlobalMaxPool2D()

      self.dense_1 = tf.keras.layers.Dense(128, activation='relu')
      self.last_dense = tf.keras.layers.Dense(3)


    def forward(self, inputs):
      x = self.conv_1(inputs)
      x = self.batch_norm_1(x)
      x = self.pool_1(x)

      x = self.conv_2(x)
      x = self.batch_norm_2(x)
      x = self.pool_2(x)

      x = self.conv_3(x)
      x = self.batch_norm_3(x)
      x = self.pool_3(x)

      x = self.global_pool(x)
      return x

    def call(self, inputs):
      x = inputs[0] #for compability we expect inputs as a list of views
      x = self.forward(x)

      x = self.dense_1(x)
      x = self.last_dense(x)
      return x

def build_model(N_view = 16, mode='RGBD'):

  if N_view == 1:
    return build_singleview_model(mode)

  inputs = [tf.keras.Input(shape=(128,128,4)) for i in range(N_view)]

  if mode == 'RGB':
    inputs = [tf.keras.Input(shape=(128,128,3)) for i in range(N_view)]

  model = ColorNet()
  
  #to build weights 
  outs = model(inputs)

  return model

def build_singleview_model(mode='RGBD'):
  #we use list for compability
  inputs = [tf.keras.Input(shape=(128,128,4))]
  model = SingleViewColorNet()

  if mode == 'RGB':
    inputs = [tf.keras.Input(shape=(128,128,3))]

  outs = model(inputs)

  return model