#test
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from torch.utils.data import DataLoader
from newModelTraining.prepareData import PianoDataset
from newModelTraining.tawareModel import TAwareModel
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# Load your test dataset
dataset_test = PianoDataset('HDF5/tolerance+-5_t_101_n_5test.hdf5')
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=True)

# Initialize the model
onsetModel = TAwareModel()

# Load the saved model checkpoint
checkpoint_path = 'onset_only_model_total_epoch=1000_current_epoch=200_n=10.pth'  # Add .pt if not present
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))  # use 'cuda' if running on GPU

print(type(checkpoint))  # to debug what is loaded

state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
onsetModel.to(device)
onsetModel.load_state_dict(state_dict)
onsetModel.eval()


import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Initialize accumulators for each note
all_true = [[] for _ in range(88)]
all_pred = [[] for _ in range(88)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
onsetModel.to(device)
onsetModel.eval()

with torch.no_grad():
    for batch in dataloader_test:
        inputs, onset_labels, _ = batch
        inputs = inputs.to(device)
        onset_labels = onset_labels.to(device)

        logits = onsetModel(inputs)
        preds = torch.sigmoid(logits)
        preds_bin = (preds > 0.5).float()

        # Move to CPU numpy arrays for metrics
        y_true = onset_labels.cpu().numpy()
        y_pred = preds_bin.cpu().numpy()

        # Accumulate per note across batches
        for note_idx in range(88):
            all_true[note_idx].extend(y_true[:, :, note_idx].flatten())
            all_pred[note_idx].extend(y_pred[:, :, note_idx].flatten())

# Compute metrics and confusion matrices per note
per_note_metrics = []
for note_idx in range(88):
    y_t = np.array(all_true[note_idx])
    y_p = np.array(all_pred[note_idx])

    if y_t.sum() == 0:
        # No positive labels for this note, skip or handle specially
        precision = recall = f1 = None
        conf_matrix = None
    else:
        precision = precision_score(y_t, y_p, zero_division=0)
        recall = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        conf_matrix = confusion_matrix(y_t, y_p)

    per_note_metrics.append({
        'note': note_idx,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    })
# ========== Overall Metrics ==========
# Flatten all true and predicted labels across all notes and time
flat_true = np.concatenate([np.array(note) for note in all_true])
flat_pred = np.concatenate([np.array(note) for note in all_pred])

overall_precision = precision_score(flat_true, flat_pred, zero_division=0)
overall_recall = recall_score(flat_true, flat_pred, zero_division=0)
overall_f1 = f1_score(flat_true, flat_pred, zero_division=0)
overall_cm = confusion_matrix(flat_true, flat_pred)

print("\n========== OVERALL TEST PERFORMANCE ==========")
print(f"Overall Precision: {overall_precision:.4f}")
print(f"Overall Recall:    {overall_recall:.4f}")
print(f"Overall F1 Score:  {overall_f1:.4f}")
print(f"Confusion Matrix:\n{overall_cm}")

for note_info in per_note_metrics:  # first 10 notes as example
    print(f"Note {note_info['note']}: Precision={note_info['precision']}, Recall={note_info['recall']}, F1={note_info['f1_score']}")
    print(f"Confusion Matrix:\n{note_info['confusion_matrix']}\n")

# ====== ADD VISUALIZATION HERE ======
import seaborn as sns

# Extract arrays for plotting
precisions = []
recalls = []
f1_scores = []
notes = list(range(88))
tp_counts = []
fp_counts = []
tn_counts = []
fn_counts = []

for note_info in per_note_metrics:
    if note_info['precision'] is None:
        precisions.append(0)
        recalls.append(0)
        f1_scores.append(0)
        tp_counts.append(0)
        fp_counts.append(0)
        tn_counts.append(0)
        fn_counts.append(0)
    else:
        precisions.append(note_info['precision'])
        recalls.append(note_info['recall'])
        f1_scores.append(note_info['f1_score'])

        cm = note_info['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        tp_counts.append(tp)
        fp_counts.append(fp)
        tn_counts.append(tn)
        fn_counts.append(fn)

# Plot Precision, Recall, F1 Scores per note
plt.figure(figsize=(18,6))
plt.plot(notes, precisions, label='Precision', marker='o')
plt.plot(notes, recalls, label='Recall', marker='o')
plt.plot(notes, f1_scores, label='F1 Score', marker='o')
plt.xlabel('Note Index (0 = lowest pitch)')
plt.ylabel('Score')
plt.title('Per-Note Precision, Recall, and F1 Score')
plt.legend()
plt.grid(True)
plt.show()

# Plot heatmaps for confusion matrix components TP, FP, FN, TN
fig, axs = plt.subplots(2, 2, figsize=(15, 8))

sns.heatmap(np.array(tp_counts).reshape(1, -1), ax=axs[0,0], cmap="Greens", cbar=True)
axs[0,0].set_title('True Positives (TP) per Note')
axs[0,0].set_yticks([])
axs[0,0].set_xlabel('Note')

sns.heatmap(np.array(fp_counts).reshape(1, -1), ax=axs[0,1], cmap="Reds", cbar=True)
axs[0,1].set_title('False Positives (FP) per Note')
axs[0,1].set_yticks([])
axs[0,1].set_xlabel('Note')

sns.heatmap(np.array(fn_counts).reshape(1, -1), ax=axs[1,0], cmap="Oranges", cbar=True)
axs[1,0].set_title('False Negatives (FN) per Note')
axs[1,0].set_yticks([])
axs[1,0].set_xlabel('Note')

sns.heatmap(np.array(tn_counts).reshape(1, -1), ax=axs[1,1], cmap="Blues", cbar=True)
axs[1,1].set_title('True Negatives (TN) per Note')
axs[1,1].set_yticks([])
axs[1,1].set_xlabel('Note')

plt.tight_layout()
plt.show()

note_counts = [np.sum(all_true[note_idx]) for note_idx in range(88)]

plt.figure(figsize=(18, 4))
plt.bar(range(88), note_counts, color='skyblue')
plt.xlabel('Note Index (0 = lowest pitch)')
plt.ylabel('Number of Positive Labels')
plt.title('Quantity of Each Note in Test Set (Onset Count)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()