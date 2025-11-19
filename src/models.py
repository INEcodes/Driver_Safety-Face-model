import tensorflow as tf
def create_attention_block(input_tensor, num_heads=8, key_dim=64):
attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(input_tensor, input_tensor)
x = LayerNormalization()(Add()([input_tensor, attention_output]))
return x




def create_spatial_feature_extractor(input_shape):
inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)


x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)


x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)


x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)


model = Model(inputs, x, name="spatial_extractor")
return model




def create_proposed_astn_model(input_shape, num_classes, sequence_length=10):
input_sequence = Input(shape=(sequence_length, *input_shape), name="input_sequence")
spatial_extractor = create_spatial_feature_extractor(input_shape)


# Extract features per-frame using Lambda, then stack
temporal_features = []
for i in range(sequence_length):
frame_features = tf.keras.layers.Lambda(lambda x: x[:, i, :, :, :])(input_sequence)
spatial_feat = spatial_extractor(frame_features)
temporal_features.append(spatial_feat)


temporal_sequence = tf.stack(temporal_features, axis=1)


lstm_branch = LSTM(128, return_sequences=True, dropout=0.3)(temporal_sequence)
lstm_branch = create_attention_block(lstm_branch, num_heads=8, key_dim=64)
lstm_branch = LSTM(64, dropout=0.2)(lstm_branch)
lstm_branch = LSTM(64, dropout=0.2)(lstm_branch)
lstm_branch = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(lstm_branch)
lstm_branch = Dropout(0.4)(lstm_branch)


cnn_branch = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(temporal_sequence)
cnn_branch = BatchNormalization()(cnn_branch)
cnn_branch = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(cnn_branch)
cnn_branch = BatchNormalization()(cnn_branch)
cnn_branch = GlobalAveragePooling1D()(cnn_branch)
cnn_branch = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(cnn_branch)
cnn_branch = Dropout(0.3)(cnn_branch)


combined = concatenate([lstm_branch, cnn_branch])


x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.4)(x)


outputs = Dense(num_classes, activation='softmax', name="output")(x)


model = Model(inputs=input_sequence, outputs=outputs, name="ASTN_Model")
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
return model




def create_proposed_cnn_attention_model(input_shape, num_classes):
inputs = Input(shape=input_shape)
x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D((2, 2))(x1)


x2 = Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
x2 = BatchNormalization()(x2)
x2 = MaxPooling2D((2, 2))(x2)


x = concatenate([x1, x2])
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)


x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)


x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)


x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)


outputs = Dense(num_classes, activation='softmax')(x)


model = Model(inputs, outputs, name="Proposed_CNN_Attention")
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
return model
