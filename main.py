import os
import glob
import cv2
import argparse
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.feature import hog

# Define a function to compute binned color features
def bin_spatial(img, size=32):
    # Resize the image and use ravel() to flat the array
    features = cv2.resize(img, (size,size)).ravel()
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient=9, pixels_per_cell=(8,8), cells_per_block=(2,2), vis=False, feature_vec=False):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pixels_per_cell,pixels_per_cell), cells_per_block=(cells_per_block,cells_per_block), transform_sqrt=True, visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pixels_per_cell,pixels_per_cell), cells_per_block=(cells_per_block,cells_per_block), transform_sqrt=True, visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=32, hist_bins=32, hist_range=(0, 256), orient=9, pixels_per_cell=8, cells_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        img = mpimg.imread(file)
        # Apply color conversion if other than 'RGB'
        if cspace == 'HSV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'HSL':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSL)
        elif cspace == 'YUV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'LUV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'YCrCb':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_img = np.copy(img)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_img, size=spatial_size)
        file_features.append(spatial_features)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(img, nbins=hist_bins, bins_range=hist_range)
        file_features.append(hist_features)
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == -1: # means all channels
            hog_features = []
            for channel in range(img.shape[-1]):
                hog_features.append(get_hog_features(img[:,:,channel], orient, pixels_per_cell, cells_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(img[:,:,hog_channel], orient, pixels_per_cell, cells_per_block, vis=False, feature_vec=True)
        file_features.append(hog_features)
        # Append the new feature vector to the features list
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def sliding_windows(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1-xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1-xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Loop through finding x and y window positions
    windows = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            x1 = xs*nx_pix_per_step + x_start_stop[0]
            y1 = ys*ny_pix_per_step + y_start_stop[0]
            x2 = x1 + xy_window[0]
            y2 = y1 + xy_window[1]
            # Append window position to list
            windows.append(((x1,y1),(x2,y2)))
    # Return the list of windows
    return windows

def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
    box_img = np.copy(img)
    for box in bboxes:
        box_img = cv2.rectangle(img, box[0], box[1], color, thick)
    return box_img

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect vehicles on images/videos', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cspace', default='RGB', help='color space used to convert input image.')
    parser.add_argument('--spatial_size', default=32, type=int, help='size used to resize input image.')
    parser.add_argument('--hist_bins', default=32, type=int, help='number of bins used for retrieve histograms of input image.')
    parser.add_argument('--orient', default=9, type=int, help='orientation for HOG.')
    parser.add_argument('--pixels_per_cell', default=8, type=int, help='number of pixels per cell for HOG.')
    parser.add_argument('--cells_per_block', default=2, type=int, help='number of cells per block for HOG.')
    parser.add_argument('--hog_channel', default=0, type=int, choices=[0, 1, 2, -1], help='channels to use for HOG. -1 means all channels.')
    args = parser.parse_args()

    # Read in car and non-car images
    vehicles = []
    non_vehicles = []
    for file in glob.glob('vehicles/**/*.png'):
        vehicles.append(file)
    for file in glob.glob('non-vehicles/**/*.png'):
        non_vehicles.append(file)

    # Extract features from vehicles and non vehicles dataset
    print("Extract features from datasets with:\n- cspace={}\n- spatial_size={}\n- hist_bins={}\n- orient={}\n- pixels_per_cell={}\n- cells_per_block={}\n- hog_channel={}\n".format(args.cspace, args.spatial_size, args.hist_bins, args.orient, args.pixels_per_cell, args.cells_per_block, args.hog_channel))
    vehicles_features = extract_features(vehicles, cspace=args.cspace, spatial_size=args.spatial_size, hist_bins=args.hist_bins, orient=args.orient, pixels_per_cell=args.pixels_per_cell, cells_per_block=args.cells_per_block, hog_channel=args.hog_channel)
    non_vehicles_features = extract_features(non_vehicles, cspace=args.cspace, spatial_size=args.spatial_size, hist_bins=args.hist_bins, orient=args.orient, pixels_per_cell=args.pixels_per_cell, cells_per_block=args.cells_per_block, hog_channel=args.hog_channel)

    # Create an array stack of feature vectors
    X = np.vstack((vehicles_features, non_vehicles_features)).astype(np.float64)
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(vehicles)), np.zeros(len(non_vehicles))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=rand_state)
    # Use a linear SVC as classifier
    clf = SVC()
    t = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print("{0:.2f} seconds to train SVC".format(t2-t))
    print("Test Accuracy of SVC = {0:.3f}".format(clf.score(X_test, y_test)))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print("My SVC predictions: ", clf.predict(X_test[:n_predict]))
    print("The right labels: ", y_test[:n_predict])
    t2 = time.time()
    print("{0:.2f} seconds to predict {1:} samples".format(t2-t, n_predict))

    for file in glob.glob('test_images/*.jpg'):
        # Read image
        print("Finding vehicles on {}".format(file))
        img = mpimg.imread(file)
        # Retrieve sliding windows from image
        windows = sliding_windows(img, x_start_stop=[None,None], y_start_stop=[400,None], xy_window=(128,128), xy_overlap=(0.5,0.5))
        # Search for car in windows using our classifier
        # TODO
        # Draw bounding boxes around detected cars
        vehicles_img = draw_boxes(img, windows)
        plt.imshow(vehicles_img)
        plt.show()
