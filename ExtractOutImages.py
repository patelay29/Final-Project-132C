
# %%
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 


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

# Save the numpy array to a file
np.save('all_images.npy', all_images)

# Convert the list of descriptors to a numpy array
description = np.array(description)

# Save the numpy array to a file
np.save('description.npy', description)
        

        

    

# %%





















