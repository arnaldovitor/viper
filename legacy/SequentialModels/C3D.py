import tensorflow as tf

def C3D(input_shape, n_classes):
    model = tf.keras.Sequential(name='C3D')

    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,2), activation='relu', strides=(2,2,2), padding="same"))

    model.add(tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), activation='relu', strides=(2,2,2), padding="same"))

    model.add(tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), activation='relu', padding="same"))
    model.add(tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), activation='relu', strides=(2,2,2), padding="same"))

    model.add(tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), activation='relu', padding="same"))
    model.add(tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), activation='relu', strides=(2,2,2), padding="same"))

    model.add(tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), activation='relu', padding="same"))
    model.add(tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), activation='relu', strides=(2,2,2), padding="same"))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))

    model.add(tf.keras.layers.Dense(n_classes))

    return model