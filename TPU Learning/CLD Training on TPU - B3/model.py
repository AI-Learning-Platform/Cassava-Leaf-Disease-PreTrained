model = models.Sequential()
    
model.add(efn.EfficientNetB3(
   include_top = False, 
    weights = 'noisy-student', 
    input_shape = (TARGET_SIZE, TARGET_SIZE, 3)
))

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(DROPOUT_RATE))
model.add(layers.BatchNormalization())
model.add(layers.Dense(5, activation = "softmax", dtype='float32', name='predictions'))# 5 is the dimensionality of the output space "5 options"
#model.add(layers.Activation('softmax', dtype='float32', name='predictions'))

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(optimizer = 'adam',
              loss = loss,
              metrics = ['accuracy','sparse_categorical_accuracy']) #try sparse_categorical_accuracy here