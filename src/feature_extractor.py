"""Feature extraction helpers for using pretrained CNNs for classical ML pipelines."""
import numpy as np
from tensorflow.keras.applications import VGG16




def extract_features(generator, sample_count, input_shape=(224,224,3)):
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
features = []
labels = []


steps = max(1, sample_count // generator.batch_size)
for _ in range(steps):
batch_x, batch_y = generator.next()
batch_features = base_model.predict(batch_x)
features.append(batch_features.reshape(batch_features.shape[0], -1))
labels.append(np.argmax(batch_y, axis=1))


return np.vstack(features), np.hstack(labels)
