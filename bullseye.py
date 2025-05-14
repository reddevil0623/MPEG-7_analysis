import sys
sys.path.append('eucalc_directory')
import eucalc as ec
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import label

# Use the current directory for images.
datafolder = "."

# List all files with the .gif extension (case-insensitive)
all_files = os.listdir(datafolder)
names = [file for file in all_files if file.lower().endswith('.gif')]

# First pass: determine the maximum dimension among all images.
max_dim = 0
for file in names:
    file_path = os.path.join(datafolder, file)
    with Image.open(file_path) as img:
        width, height = img.size  # PIL returns (width, height)
        max_dim = max(max_dim, width, height)
target_length = max_dim + 10
print("Global target length for padding:", target_length)

# Group files by category and extract the number from each filename.
# Expected filename format: "xxx-n.gif" (where "xxx" is the category and "n" is the image number)
files_by_category = {}
for file in names:
    parts = file.split('-')
    if len(parts) < 2:
        continue  # Skip files not matching the expected naming convention
    category = parts[0]
    num_str = parts[1].split('.')[0]
    try:
        number = int(num_str)
    except ValueError:
        number = num_str
    files_by_category.setdefault(category, []).append((number, file))

# Sort files within each category by their number and sort the categories alphabetically.
for cat, file_list in files_by_category.items():
    file_list.sort(key=lambda x: x[0])
sorted_categories = sorted(files_by_category.keys())

# Parameters for the ECT computation.
k = 360*4
xinterval = (-1.5, 1.5)
xpoints = 3000

#--------------------------------------------------------------------------
# Function to keep only the largest connected component in a binary image.
def filter_to_largest_cc(img_array):
    """
    Given a binary image array (with white pixels having values > 0),
    find the largest connected component and return a new image in which
    only the largest connected component is white (1) and all other pixels
    (including other white regions) are set to 0.
    """
    # Create a boolean mask for white pixels.
    mask = img_array > 0
    
    # Label connected components using 8-connectivity (structure of ones).
    labeled_array, num_features = label(mask, structure=np.ones((3, 3)))
    
    # If no connected component is found, return an image of zeros.
    if num_features == 0:
        return np.zeros_like(img_array)
    
    # Calculate the size of each component.
    component_sizes = np.bincount(labeled_array.ravel())
    
    # Exclude the background (index 0) and find the largest component.
    if len(component_sizes) > 1:
        largest_component = np.argmax(component_sizes[1:]) + 1  # +1 because index 0 is background
    else:
        return np.zeros_like(img_array)
    
    # Create a mask keeping only the largest connected component.
    filtered_mask = (labeled_array == largest_component)
    return filtered_mask.astype(img_array.dtype)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Function to pad a binary image to a square of target_length.
def pad_image_to_square(img_array, target_length):
    """
    Pad the given binary image (values 0 or 1) with black pixels so that 
    it becomes a square of side target_length. The image is shifted so that
    the centroid (average coordinate) of all white pixels (value > 0) is centered.
    """
    # Get original dimensions.
    orig_height, orig_width = img_array.shape[0:2]
    
    # Compute indices of white pixels.
    white_pixels = np.argwhere(img_array > 0)
    if white_pixels.size == 0:
        # If no white pixels, default to the center of the original image.
        centroid = np.array([orig_height / 2.0, orig_width / 2.0])
    else:
        centroid = white_pixels.mean(axis=0)
    
    # Determine the center coordinate of the target image.
    target_center = np.array([target_length / 2.0, target_length / 2.0])
    
    # Compute the offset needed to center the white pixel centroid.
    offset = np.round(target_center - centroid).astype(int)
    
    # Create a new square image filled with zeros (black).
    new_img = np.zeros((target_length, target_length), dtype=img_array.dtype)
    
    # Compute destination indices for pasting.
    paste_row = offset[0]
    paste_col = offset[1]
    
    dest_row_start = max(0, paste_row)
    dest_col_start = max(0, paste_col)
    dest_row_end = min(target_length, paste_row + orig_height)
    dest_col_end = min(target_length, paste_col + orig_width)
    
    # Compute corresponding source indices.
    src_row_start = max(0, -paste_row)
    src_col_start = max(0, -paste_col)
    src_row_end = src_row_start + (dest_row_end - dest_row_start)
    src_col_end = src_col_start + (dest_col_end - dest_col_start)
    
    # Paste the original image into the new image.
    new_img[dest_row_start:dest_row_end, dest_col_start:dest_col_end] = \
        img_array[src_row_start:src_row_end, src_col_start:src_col_end]
    
    return new_img

class EctImg:
    def __init__(self, nm, img, k=20, xinterval=(-1., 1.), xpoints=100):
        self.nm = nm
        self.xinterval = xinterval
        self.xpoints = xpoints
        # Compute the ECT image at initialization.
        self.image = self.compute(img, k, xinterval, xpoints)
    
    def compute(self, img, k, xinterval, xpoints):
        # Create an embedded complex from the image using eucalc.
        cplx = ec.EmbeddedComplex(img)
        cplx.preproc_ect()
        thetas = np.random.uniform(0, 2 * np.pi, k + 1)
        ect1 = np.empty((k, xpoints), dtype=float)
        for i in range(k):
            theta = thetas[i]
            direction = np.array((np.sin(theta), np.cos(theta)))
            ect_dir = cplx.compute_euler_characteristic_transform(direction)
            T = np.linspace(xinterval[0], xinterval[1], xpoints)
            ect1[i] = [ect_dir.evaluate(t) for t in T]
        return ect1

# Lists to store computed ECT images and corresponding labels.
ECT_all = []
labels_all = []

# Process images category by category.
for category in sorted_categories:
    for number, file in files_by_category[category]:
        file_path = os.path.join(datafolder, file)
        with Image.open(file_path) as img:
            try:
                # For animated images, use the first frame.
                img.seek(0)
            except EOFError:
                pass
            img_array = np.array(img)
        
        # First, filter the image to keep only the largest connected component.
        filtered_img = filter_to_largest_cc(img_array)
        
        # Then, pad the filtered binary image to a square with target_length.
        padded_img = pad_image_to_square(filtered_img, target_length)
        
        # Compute the ECT image for the padded image.
        ect_instance = EctImg(file, padded_img, k, xinterval, xpoints)
        ECT_all.append(ect_instance.image)
        labels_all.append((category, number))

# Optionally, flatten the ECT images for subsequent classification tasks.
flattened_all = [img.flatten() for img in ECT_all]

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# ---------- Provided Distance Functions ----------
def wasserstein_distance(empirical1, empirical2, delta_x=1.0):
    """
    Compute the Wasserstein distance between two empirical measures,
    each represented as a k x N array (each row is a function sampled at N points),
    incorporating the delta_x factor for uniformly sampled points.
    """
    cost_matrix = cdist(empirical1, empirical2, metric='minkowski', p=1) * (delta_x )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    avg_cost_p = np.mean(cost_matrix[row_ind, col_ind])
    return avg_cost_p

def compute_row(i, measures, delta_x=1.0):
    """
    Computes the i-th row of the distance matrix.
    """
    n = len(measures)
    row = np.zeros(n)
    for j in range(i, n):
        d = wasserstein_distance(measures[i], measures[j], delta_x=delta_x)
        row[j] = d
    return i, row

def compute_distance_matrix_parallel(measures, delta_x=1.0):
    """
    Computes the symmetric pairwise distance matrix in parallel.
    """
    n = len(measures)
    results = Parallel(n_jobs=-1)(delayed(compute_row)(i, measures, delta_x) for i in range(n))
    distance_matrix = np.zeros((n, n))
    for i, row in results:
        distance_matrix[i, i:] = row[i:]
        distance_matrix[i:, i] = row[i:]
    return distance_matrix

# ---------- Data and Labels Setup ----------
# Assume that ECT_all and labels_all have been computed previously.
# For this example, we use a subset of 200 entries.
subset = ECT_all              # list of k x N arrays (ECT features)
labels_subset = labels_all     # list of tuples (category, number)

# Extract the category (first element of each tuple) as target labels.
categories = np.array([label[0] for label in labels_subset])

distance_matrix = compute_distance_matrix_parallel(subset, delta_x=1.0)

# --- Compute the Bull's‐Eye Score ---
# Extract class labels (first element of each label tuple)
categories = np.array([lbl[0] for lbl in labels_subset])

N = len(categories)
# Determine number of samples per class (assumes uniform)
_, counts = np.unique(categories, return_counts=True)
M = counts[0]

top_k = 40
total_hits = 0

for i in range(N):
    # sort distances ascending, include the query itself at index 0
    nearest = np.argsort(distance_matrix[i])[:top_k]
    # count how many in top_k share the same category as query i
    total_hits += np.sum(categories[nearest] == categories[i])

# maximum possible hits = N * M (including self)
bulls_eye_rate = total_hits / (N * M)

print(f"Bull's‐Eye Retrieval Rate: {bulls_eye_rate:.4f} ({bulls_eye_rate*100:.2f}%)")
