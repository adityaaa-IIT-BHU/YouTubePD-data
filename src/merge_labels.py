import pandas as pd
import numpy as np

print("Loading features and ground truth labels...")

# 1. Load the features you extracted
# (Make sure the filename matches what you saved in build_dataset.py)
try:
    features_df = pd.read_csv('multimodal_features.csv')
except FileNotFoundError:
    print("❌ Cannot find multimodal_features.csv. Make sure you ran the extraction script first!")
    exit()

# 2. Load the Excel sheet
labels_df = pd.read_excel('data_sheets/data_sheet.xlsx')

# 3. Recreate the Video_ID mapping 
# Row 0 = video134.mp4, Row 1 = video135.mp4, etc.
labels_df['Video_ID'] = 'video' + (labels_df.index + 134).astype(str) + '.mp4'

# 4. Clean up the Binary Label (Convert 'y'/'n' to 1/0)
# This safely handles whether they used letters or numbers in the Excel sheet
labels_df['Label_Binary'] = labels_df['parkinson y/n'].map({'y': 1, 'n': 0, 'Y': 1, 'N': 0, 1: 1, 0: 0})

# 5. Select only the columns we actually need from the massive Excel sheet
labels_subset = labels_df[['Video_ID', 'Label_Binary', 'severeness_label', 'split']]

# 6. Perform the Merge (Inner join on Video_ID)
print("Merging datasets...")
final_df = pd.merge(features_df, labels_subset, on='Video_ID', how='inner')

# Drop any rows where the label is missing (NaN)
final_df = final_df.dropna(subset=['Label_Binary'])

# Save the master dataset
final_df.to_csv('master_training_data.csv', index=False)
print(f"✅ Success! Merged dataset saved to 'master_training_data.csv'")
print(f"Total labeled samples ready for AI training: {len(final_df)}")