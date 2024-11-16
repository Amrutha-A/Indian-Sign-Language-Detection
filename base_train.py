import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

# Paths and Actions
DATA_PATH = r"C:\Users\Amrutha A\OneDrive\Desktop\Silcosys project\Distance_data"
actions = np.array(['a lot','abuse','afraid','you','free','bring','water','hiding','today', 'repeat', 'me', 'help', 'please'])
label_map = {label: num for num, label in enumerate(actions)}

# Load Data
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(30):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Residual Block Function
def residual_block(x, filters, kernel_size=3, strides=1):
    shortcut = Conv1D(filters, kernel_size=1, strides=strides, padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    x = Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

# Model definition
inputs = Input(shape=(30, 278))

# Conv1_x
x = Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)

# Conv2_x
x = residual_block(x, filters=64)
x = residual_block(x, filters=64)

# Conv3_x
x = residual_block(x, filters=128, strides=2)
x = residual_block(x, filters=128)

# Conv4_x
x = residual_block(x, filters=256, strides=2)
x = residual_block(x, filters=256)

# Conv5_x
x = residual_block(x, filters=512, strides=2)
x = residual_block(x, filters=512)

# Global Average Pooling and Fully Connected Layer with Softmax
x = GlobalAveragePooling1D()(x)
outputs = Dense(actions.shape[0], activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the Model
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

ep = [10,20,30,40]
bs = [8,16,32]

for i in ep:
    for j in bs:
        history = model.fit(X_train, y_train, epochs=i, batch_size=j, validation_data=(X_test, y_test), callbacks=[tb_callback])

        # Make Predictions
        res = model.predict(X_test)
        print("Predictions:", res)
        print("epoch: ", i)
        print("batch size: ",j)
        print(actions[np.argmax(res[1])])
        print(actions[np.argmax(y_test[1])])
        

        loss, accuracy = model.evaluate(X_test, y_test)
        print("Accuracy: ",accuracy)
        print("Loss: ",loss)
        import matplotlib.pyplot as plt
        
        plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
        # plt.text(0.75, 0.2, f'Test Accuracy: {accuracy:.2f}', fontsize=12, color='red', transform=plt.gca().transAxes)
        # plt.text(0.75, 0.15, f'Test Loss: {loss:.2f}', fontsize=12, color='red', transform=plt.gca().transAxes)

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()