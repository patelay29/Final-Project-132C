
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

np.save('reshaped_array.npy', reshaped_array)

        

        




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


# Assuming you have a list of 18 values indicating whether each MRI scan has a disease or not
# Replace this with your actual list
ground_truth_labels_age = [entry[0] for entry in description]
ground_truth_labels_age = np.array(ground_truth_labels_age).astype(float)



# Visualize the ground truth using PCA
plt.figure(figsize=(8, 6))
plt.scatter(mri_scans_pca[:, 0], mri_scans_pca[:, 1], c=ground_truth_labels_age, cmap='viridis', edgecolor='k')
plt.title('ground truth differentiation with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Age')
plt.show()

# Reduce dimensionality for visualization
pca10 = PCA(n_components=6)
mri_scans_pca10 = pca10.fit_transform(mri_scans_scaled)
# Alternatively, you can use t-SNE for visualization with reduced perplexity
tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # Adjust the perplexity value as needed
mri_scans_tsne = tsne.fit_transform(mri_scans_pca10)




# Visualize the ground truth using t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(mri_scans_tsne[:, 0], mri_scans_tsne[:, 1], c=ground_truth_labels_age, cmap='viridis', edgecolor='k')
plt.title('ground truth differentiation with t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Age')
plt.show()

# Plot the variance explained by each principal component
plt.figure(figsize=(8, 6))
plt.bar(range(1, pca10.n_components_ + 1), pca10.explained_variance_ratio_)
plt.title('Variance explained by each principal component')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()





# %%
# CDR 
ground_truth_labels_cdr = [entry[3] for entry in description]
ground_truth_labels_cdr = np.array(ground_truth_labels_cdr).astype(float)

# Visualize the ground truth using PCA
plt.figure(figsize=(8, 6))
for cluster_label in np.unique(ground_truth_labels_cdr):
    indices = np.where(ground_truth_labels_cdr == cluster_label)
    plt.scatter(mri_scans_pca[indices, 0], mri_scans_pca[indices, 1], label=f'CDR {cluster_label}', cmap='paired', edgecolor='k')
plt.title('ground truth differentiation with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
for cluster_label in np.unique(ground_truth_labels_cdr):
    indices = np.where(ground_truth_labels_cdr == cluster_label)
    plt.scatter(mri_scans_tsne[indices, 0], mri_scans_tsne[indices, 1], label=f'CDR {cluster_label}', cmap='tab10', edgecolor='k')
plt.title('ground truth differentiation with t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.show()


# %%
