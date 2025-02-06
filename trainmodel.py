import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import sys

# Ensure the script uses utf-8 encoding for output (for Windows systems)
sys.stdout.reconfigure(encoding='utf-8')

# Disable oneDNN custom operations if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Assuming 'actions', 'no_sequences', 'sequence_length', and 'DATA_PATH' are defined somewhere in 'function'
from function import *

def load_data(actions, no_sequences, sequence_length, DATA_PATH):
    # Create label map
    label_map = {label: num for num, label in enumerate(actions)}

    # Prepare data
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)

    return X, y

def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_and_save_model(model, X_train, y_train, log_dir):
    # Define log directory for TensorBoard
    tb_callback = TensorBoard(log_dir=log_dir)

    # Train the model
    model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

    # Print the model summary
    model.summary()

    # Save the model architecture to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # Save the model weights
    model.save('model.h5')

def evaluate_model(model, X_test, y_test):
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    
    # For binary classification or ROC-AUC for multi-class
    if y_test.shape[1] == 2:
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # assuming binary classification
        roc_auc = roc_auc_score(y_true_classes, y_pred_prob)
    else:
        # Handle multi-class ROC-AUC here if needed
        roc_auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    return accuracy, precision, recall, f1, roc_auc, cm

if __name__ == "__main__":
    # Parameters
    actions = ['A',"Bye",'C',"HI","OK"]  # List of actions (e.g., 'action1', 'action2', ...)
    no_sequences = 30  # Number of sequences per action
    sequence_length = 30  # Length of each sequence
    DATA_PATH = os.path.join('MP_Data') 


    # Load data
    X, y = load_data(actions, no_sequences, sequence_length, DATA_PATH)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # Build LSTM model
    input_shape = (sequence_length, X.shape[-1])
    num_classes = len(actions)
    model = build_lstm_model(input_shape, num_classes)

    # Train and save the model
    log_dir = os.path.join('Logs')
    train_and_save_model(model, X_train, y_train, log_dir)

    # Evaluate CNN model
    cnn_accuracy, cnn_precision, cnn_recall, cnn_f1, cnn_roc_auc, cnn_cm = evaluate_model(model, X_test, y_test)

    # Print or log evaluation metrics
    print("LSTM Model Evaluation:")
    print(f"Accuracy: {cnn_accuracy}, Precision: {cnn_precision}, Recall: {cnn_recall}, F1-score: {cnn_f1}")
    print(f"ROC-AUC: {cnn_roc_auc}")
    print(f"Confusion Matrix:\n{cnn_cm}\n")
