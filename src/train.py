import pandas as pd

# Load the CSV file
partition_df = pd.read_csv('data/processed/list_eval_partition.csv', header=None)

# Rename columns for clarity
partition_df.columns = ['image_id', 'partition']

# Split the data into train, validation, and test sets
train_set = partition_df[partition_df['partition'] == 0]
validation_set = partition_df[partition_df['partition'] == 1]
test_set = partition_df[partition_df['partition'] == 2]

# Print out the sizes of each set
print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(validation_set)}")
print(f"Test set size: {len(test_set)}")

# import os
# import shutil

# # Define the base directories
# image_dir = 'path/to/img_align_celeba'
# train_dir = 'path/to/data/processed/train'
# validation_dir = 'path/to/data/processed/validation'
# test_dir = 'path/to/data/processed/test'

# # Create directories if they don't exist
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(validation_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)

# # Function to copy images to respective directories
# def copy_images(set_df, target_dir):
#     for img_id in set_df['image_id']:
#         src_path = os.path.join(image_dir, img_id)
#         dst_path = os.path.join(target_dir, img_id)
#         shutil.copy(src_path, dst_path)

# # Copy images to the corresponding directories
# copy_images(train_set, train_dir)
# copy_images(validation_set, validation_dir)
# copy_images(test_set, test_dir)
