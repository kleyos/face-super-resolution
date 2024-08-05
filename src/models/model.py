from tensorflow.keras import layers, models

def build_srcnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(None, None, 3)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(layers.Conv2D(3, (5, 5), padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

model = build_srcnn_model()
model.summary()
