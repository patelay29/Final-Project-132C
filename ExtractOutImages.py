
# %%
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


file_path = 'disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.img'
img = nib.load(file_path)

data = img.get_fdata()

# select a slice
slice_idx = data.shape[2] // 2 + 10
slice_data = data[:, :, slice_idx, 0]

# plot the slice
plt.imshow(slice_data.T, cmap='gray', origin='lower')
plt.axis('off')

#%%
full_brain = data[:,:,:,0]
# loop through the slices of patient 1
for slice in full_brain:
    plt.imshow(slice.T, cmap='gray', origin='lower')
    plt.axis('off')
    plt.show()


#%%
# loop through all patients

# Step 1: Read file names from Excel spreadsheet
excel_file = 'oasis_cross-sectionalcopycopy.xlsx'  # Replace with your Excel file
df = pd.read_excel(excel_file)
desired_file_names = df['ID'].tolist()

# Get list of files in the disc1 folder
file_list = os.listdir('disc1')

all_images = []
description = []

for patientID in range(1,len(file_list)+1):
    file_path = f'disc1/OAS1_{str(patientID).zfill(4)}_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_{str(patientID).zfill(4)}_MR1_mpr_n4_anon_sbj_111.img'
    
    if file_list[patientID-1] in desired_file_names and os.path.exists(file_path):
        print(file_list[patientID-1])
        print("File has data")
        img = nib.load(file_path)
        data = img.get_fdata()
        slice_idx = data.shape[2] // 2
        slice_data = data[:, :, slice_idx, 0]
        all_images.append(slice_data)
            # Find the corresponding row in the Excel spreadsheet
        row = df.loc[df['ID'] == file_list[patientID-1]]
        
        # Extract the descriptors
        age = row['Age'].values[0]
        gender = row['M/F'].values[0]
        education = row['Educ'].values[0]
        cdr = row['CDR'].values[0]
        
        # Append the descriptors to the corresponding image
        # (Assuming all_images and desired_file_names have the same order)
        description.append([age, gender, education, cdr])

# Convert the list of images to a numpy array
all_images = np.array(all_images)

reshaped_array = all_images.reshape(-1, 65536)
# Save the numpy array to a file
np.save('all_images.npy', all_images)

scalar = StandardScaler()
Reshaped_scaled = scalar.fit_transform(reshaped_array)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(Reshaped_scaled)
cdr = np.array([df['CDR'].values]).T
ages = np.array([df['Age'].values]).T
education = np.array([df['Educ'].values]).T
# Convert the list of descriptors to a numpy array
description = np.array(description)

# Save the numpy array to a file
np.save('description.npy', description)

        

        




#%%
#Good one 


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Assuming you already have your MRI scans stored in an array named mri_scans
# Shape of mri_scans: (18, 65536)

# Preprocess the data (e.g., scaling)
scaler = StandardScaler()
mri_scans_scaled = scaler.fit_transform(reshaped_array)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
mri_scans_pca = pca.fit_transform(mri_scans_scaled)

# Apply k-means clustering
n_clusters = 3  # Change this to your desired number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(mri_scans_scaled)

# Convert cluster labels to match the ground truth labels
# For example, if cluster 0 corresponds to label 0.0, cluster 1 corresponds to label 0.5, and cluster 2 corresponds to label 1.0
# You may need to adjust this mapping based on the actual clustering results
cluster_mapping = {0: 0.0, 1: 0.5, 2: 1.0}
predicted_labels = np.array([cluster_mapping[cluster] for cluster in clusters])

# Visualize the clusters using PCA
plt.figure(figsize=(8, 6))
plt.scatter(mri_scans_pca[:, 0], mri_scans_pca[:, 1], c=predicted_labels, cmap='viridis', edgecolor='k', vmin=0, vmax=1)
plt.title('k-means clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Predicted CDR')
plt.show()


# Alternatively, you can use t-SNE for visualization with reduced perplexity
tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # Adjust the perplexity value as needed
mri_scans_tsne = tsne.fit_transform(mri_scans_scaled)


# Visualize the clusters using t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(mri_scans_tsne[:, 0], mri_scans_tsne[:, 1], c=predicted_labels, cmap='viridis', edgecolor='k', vmin=0, vmax=1)
plt.title('k-means clustering with t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Predicted CDR')
plt.show()

from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix

# Assuming you have a list of 18 values indicating whether each MRI scan has a disease or not
# Replace this with your actual list
ground_truth_labels = [entry[3] for entry in description]
ground_truth_labels = np.array(ground_truth_labels).astype(float)



# Define thresholds for discretization
thresholds = [0.25, 0.75]

# Convert continuous values to discrete classes
ground_truth_classes = np.digitize(ground_truth_labels, thresholds)
predicted_classes = np.digitize(predicted_labels, thresholds)

# Compute accuracy
accuracy = accuracy_score(ground_truth_classes, predicted_classes)
print("Accuracy:", accuracy)

# print a ground truth labels as they compare to the predicted labels
# Format the ground truth and predicted labels into pairs
label_pairs = [f"Ground Truth: {gt:.1f}\tPredicted: {pred:.1f}" for gt, pred in zip(ground_truth_labels.astype(float), predicted_labels.astype(float))]

# Join the pairs together with newline character
output = "\n".join(label_pairs)

# Print the output
print(output)

# %%
