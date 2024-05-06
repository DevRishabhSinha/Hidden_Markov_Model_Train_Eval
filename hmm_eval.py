import numpy as np
import glob
import os
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import joblib

# Function to load MFCCs from .npy files across multiple directories
def load_mfccs(directory):
    data = []
    labels = []
    # Iterate over each sub-directory within the main directory
    for subdir in sorted(os.listdir(directory)):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):  # Ensure it is a directory
            for filepath in glob.glob(f"{subdir_path}/*.npy"):
                # Extract digit from filename, assuming filename format 'digit_vp_rep.npy'
                label = int(filepath.split("/")[-1].split("_")[0])
                mfccs = np.load(filepath)
                data.append(mfccs)
                labels.append(label)
    return data, labels

# Path where the MFCCs are stored
mfcc_directory = '/Users/a12345/AudioMNIST/preprocessed_data'  # Update this path to where your 'preprocessed_data' folder is located

# Load data
data, labels = load_mfccs(mfcc_directory)

lengths = [len(mfcc) for mfcc in data]

# Split the dataset while keeping the sequence structure
indices = range(len(data))
train_indices, test_indices, _, _ = train_test_split(indices, indices, test_size=0.2, random_state=42)

data_train = [data[i] for i in train_indices]
labels_train = [labels[i] for i in train_indices]
data_test = [data[i] for i in test_indices]
labels_test = [labels[i] for i in test_indices]

# Stack data for training and testing
data_train_stacked = np.vstack(data_train)
data_test_stacked = np.vstack(data_test)
train_lengths = [len(x) for x in data_train]
test_lengths = [len(x) for x in data_test]

data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)  # split data
train_lengths = [len(seq) for seq in data_train]  # lengths of sequences for training
test_lengths = [len(seq) for seq in data_test]  # lengths of sequences for testing

model = hmm.GaussianHMM(n_components=10, covariance_type="full", n_iter=1, tol=0.01)

prev_log_prob = -np.inf
tolerance = 0.01
converged = False
iteration = 0
max_iter = 5000

while not converged and iteration < max_iter:
    model.fit(np.vstack(data_train), lengths=train_lengths)  # Fit model on training data
    # Compute log probability on test set
    log_prob = model.score(np.vstack(data_test), lengths=test_lengths)
    print(f"Iteration {iteration}, Test Log Probability: {log_prob}")

    # Check convergence
    if abs(log_prob - prev_log_prob) < tolerance:
        converged = True
        print("Model has converged")
    else:
        prev_log_prob = log_prob
        iteration += 1
logprob, states = model.decode(data_test_stacked, test_lengths)
predicted_labels = np.argmax(np.bincount(states))
accuracy = np.mean(predicted_labels == labels_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Save the trained model
model_path = 'hmm_speech_recognition_model.pkl'  # Change this path if you want to save the model elsewhere
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")