import numpy as np
import glob
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import joblib
import os


def load_mfccs(directory):
    data = []
    labels = []
    for subdir in sorted(os.listdir(directory)):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for filepath in glob.glob(f"{subdir_path}/*.npy"):
                label = int(filepath.split("/")[-1].split("_")[0])
                mfccs = np.load(filepath)
                data.append(mfccs)
                labels.append(label)
    return data, labels

mfcc_directory = '/Users/a12345/AudioMNIST/preprocessed_data'
data, labels = load_mfccs(mfcc_directory)

# Instead of flattening the data, we keep it as a list of arrays
# Calculate lengths from the original data list
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

model = hmm.GaussianHMM(n_components=10, covariance_type="diag", n_iter=1000)
model.fit(data_train_stacked, train_lengths)

# For testing, we use the concatenated test data and lengths
logprob, states = model.decode(data_test_stacked, test_lengths)

# Evaluation and accuracy calculation can follow
# For simplicity here, we skip proper evaluation and assume a method to calculate accuracy

print("Model trained and tested. Ready to evaluate accuracy.")

model_path = 'hmm_speech_recognition_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
