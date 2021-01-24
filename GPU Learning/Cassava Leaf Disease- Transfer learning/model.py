model = keras.models.Sequential()

model.add(keras.applications.Xception(
    input_shape=(img_height, img_width, 3), 
    weights=Xception_NOTOP, 
    include_top=False)
)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(5, activation='softmax'))
