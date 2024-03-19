
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

# Plotting 
plt.scatter(principal_components[:,0], principal_components[:,1], c = cdr, cmap = 'viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of MRI images')
        

        

    

# %%
import os
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Step 1: Read file names from Excel spreadsheet
excel_file = 'oasis_cross-sectionalcopycopy.xlsx'  # Replace with your Excel file
df = pd.read_excel(excel_file)
desired_file_names = df['ID'].tolist()

# Get list of files in the disc1 folder
file_list = os.listdir('disc1')

pixel_data = []  # List to store flattened pixel data

for file_name in file_list:
    patientID = int(file_name.split('_')[1])
    if file_name[:-4] in desired_file_names:  # Remove the file extension before checking
        file_path = os.path.join('disc1', file_name, 'PROCESSED', 'MPRAGE', 'SUBJ_111',
                                 f'{file_name}_mpr_n4_anon_sbj_111.img')

        if os.path.exists(file_path):
            img = nib.load(file_path)
            data = img.get_fdata()
            slice_idx = data.shape[2] // 2
            slice_data = data[:, :, slice_idx, 0]
            
            # Flatten the 2D pixel array
            flattened_data = slice_data.flatten()
            pixel_data.append(flattened_data)





















# %%
import os
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# Flatten the images and standardize the data
reshaped_array = all_images.reshape(-1, 65536)
scalar = StandardScaler()
reshaped_scaled = scalar.fit_transform(reshaped_array)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(reshaped_scaled)

# Extract CDR values for coloring
cdr_values = df['CDR'].values

# Plotting
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=cdr_values, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of MRI images (Colored by CDR)')
plt.colorbar(label='CDR')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(reshaped_array)

# Example: Identify regions of interest (dummy data)
# Let's say we have a binary mask indicating regions of disease (1) and healthy regions (0)
# Replace this with your actual method for identifying regions of interest
mask = np.random.randint(0, 2, size=(18, 256, 256))

# Map PCA components back to image space
reconstructed_images = pca.inverse_transform(principal_components)
reconstructed_images = reconstructed_images.reshape(-1, 256, 256)

# Color coding based on regions of interest
colors = np.zeros((18, 256, 256, 3))  # Initialize array for colored images
colors[mask == 1] = [255, 0, 0]  # Color regions of disease as red

# Plot color-coded images
plt.figure(figsize=(10, 6))
for i in range(18):
    plt.subplot(3, 6, i + 1)
    plt.imshow(colors[i].astype(np.uint8))
    plt.title(f'Image {i + 1}')
    plt.axis('off')
plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(reshaped_array)

# Example: Threshold CDR values to identify images with disease
cdr_values = [entry[3] for entry in description] # Assuming CDR values are stored in the 4th column of the description array

# Threshold CDR values to identify images with disease
disease_mask = np.array(cdr_values) > 0  # Binary mask indicating images with disease
# Example: Generate a mask for important pixels in images with disease
# You need to replace this with your actual method for identifying important pixels
# For demonstration purposes, let's create a random mask
important_pixels_mask = np.random.randint(0, 2, size=(18, 256, 256))

# Color code the important pixels in images with disease
colors = np.zeros((18, 256, 256, 3))  # Initialize array for colored images
colors[disease_mask] = [0, 0, 255]  # Color pixels in images with disease as blue
colors[disease_mask & (important_pixels_mask == 1)] = [255, 0, 0]  # Color important pixels in blue images as red

# Plot color-coded images
plt.figure(figsize=(10, 6))
for i in range(18):
    plt.subplot(3, 6, i + 1)
    plt.imshow(colors[i].astype(np.uint8))
    plt.title(f'Image {i + 1}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load your MRI scan data
# Replace this with your actual MRI scan data loading code
# Your data should be in the form of a matrix where each row represents a scan and each column represents a feature (e.g., voxel intensity)
# X = load_mri_data()

# Preprocess the data (e.g., scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality for visualization (optional)
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X_scaled)

# Determine the optimal number of clusters (optional)
# Use methods like the elbow method or silhouette score to find the optimal k
# Then, replace n_clusters with your chosen number of clusters
# optimal_k = find_optimal_k(X_scaled)

# Apply k-means clustering
n_clusters = 2  # Change this to your desired number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualize the clusters (optional)
# Replace X_pca or X_tsne with the reduced dimensionality data if you performed dimensionality reduction
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
# plt.title('k-means clustering')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.colorbar()
# plt.show()

# Alternatively, you can further analyze the clusters to identify the diseased and non-diseased brain regions
# For example, if you have ground truth labels for some of your data, you can compare the clusters to the ground truth labels
# Or you can perform further analysis on the cluster centroids to interpret the clusters

# Output the cluster assignments or use them for further analysis
# print(clusters)


#%%

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

# Visualize the clusters using PCA
plt.figure(figsize=(8, 6))
plt.scatter(mri_scans_pca[:, 0], mri_scans_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k')
plt.title('k-means clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()


# Alternatively, you can use t-SNE for visualization with reduced perplexity
tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # Adjust the perplexity value as needed
mri_scans_tsne = tsne.fit_transform(mri_scans_scaled)


# Visualize the clusters using t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(mri_scans_tsne[:, 0], mri_scans_tsne[:, 1], c=clusters, cmap='viridis', edgecolor='k')
plt.title('k-means clustering with t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Cluster')
plt.show()

from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix

# Assuming you have a list of 18 values indicating whether each MRI scan has a disease or not
# Replace this with your actual list
ground_truth_labels = [entry[3] for entry in description]
ground_truth_labels = np.array(ground_truth_labels).astype(float)

# Convert cluster labels to match the ground truth labels
# For example, if cluster 0 corresponds to label 0.0, cluster 1 corresponds to label 0.5, and cluster 2 corresponds to label 1.0
# You may need to adjust this mapping based on the actual clustering results
cluster_mapping = {0: 0.0, 1: 0.5, 2: 1.0}
predicted_labels = np.array([cluster_mapping[cluster] for cluster in clusters])

# Define thresholds for discretization
thresholds = [0.25, 0.75]

# Convert continuous values to discrete classes
ground_truth_classes = np.digitize(ground_truth_labels, thresholds)
predicted_classes = np.digitize(predicted_labels, thresholds)

# Compute accuracy
accuracy = accuracy_score(ground_truth_classes, predicted_classes)
print("Accuracy:", accuracy)


# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score



# Extract ground truth labels from the description list
ground_truth_labels = np.array([entry[3] for entry in description], dtype=float)

# Preprocess the data (e.g., scaling)
scaler = StandardScaler()
mri_scans_scaled = scaler.fit_transform(reshaped_array)

# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
mri_scans_pca = pca.fit_transform(mri_scans_scaled)

# Apply k-means clustering
n_clusters = 3  # Change this to your desired number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(mri_scans_scaled)

# Convert cluster labels to match the ground truth labels
# Replace this with a proper mapping based on your clustering results
# For simplicity, let's assume cluster 0 corresponds to label 0.0, cluster 1 corresponds to label 0.5, and cluster 2 corresponds to label 1.0
cluster_mapping = {0: 0.0, 1: 0.5, 2: 1.0}
predicted_labels = np.array([cluster_mapping[cluster] for cluster in clusters])

# Define thresholds for discretization
thresholds = [0.25, 0.75]

# Convert continuous values to discrete classes
ground_truth_classes = np.digitize(ground_truth_labels, thresholds)
predicted_classes = np.digitize(predicted_labels, thresholds)

# Compute accuracy
accuracy = accuracy_score(ground_truth_classes, predicted_classes)
print("Accuracy:", accuracy)

# Get the minimum and maximum ground truth labels for setting color bar limits
cmin = np.min(ground_truth_labels)
cmax = np.max(ground_truth_labels)

# Visualize the clusters using PCA
plt.figure(figsize=(8, 6))
plt.scatter(mri_scans_pca[:, 0], mri_scans_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k')
plt.title('k-means clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster', ticks=np.arange(n_clusters))
plt.clim(-0.5, n_clusters - 0.5)  # Set color bar limits
plt.show()

# Alternatively, you can use t-SNE for visualization with reduced perplexity
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
mri_scans_tsne = tsne.fit_transform(mri_scans_scaled)

# Visualize the clusters using t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(mri_scans_tsne[:, 0], mri_scans_tsne[:, 1], c=clusters, cmap='viridis', edgecolor='k')
plt.title('k-means clustering with t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Cluster', ticks=np.arange(n_clusters))
plt.clim(0.5, n_clusters - 0.5)  # Set color bar limits
plt.show()

# %%
