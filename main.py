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
from sklearn.externals import joblib
from skimage.feature import hog
from scipy.ndimage.measurements import label

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
    for img in imgs:
        file_features = []
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

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0,0,255), thick=2):
    box_img = np.copy(img)
    choose_random_color = False
    for box in bboxes:
        if color == 'random' or choose_random_color:
            choose_random_color = True
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        box_img = cv2.rectangle(box_img, box[0], box[1], color, thick)
    return box_img

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_vehicles_in_windows(img, windows, scaler, clf, cspace='RGB', spatial_size=32, hist_bins=32, orient=9, pixels_per_cell=8, cells_per_block=2, hog_channel=0):
    hot_windows = []
    # Iterate throug each window
    for window in windows:
        # Extract features from window
        window_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64,64)) # resize img as the ones used for training are 64x64
        features = extract_features([window_img], cspace=cspace, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, hog_channel=hog_channel)
        # Scale features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict vehicle detection in window using our classifier
        prediction = clf.predict(test_features)
        # If a vehicle has been detected, then append window to the list
        if prediction == 1:
            hot_windows.append(window)
    # Return windows list containing detected vehicles
    return hot_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def save_figure(img, path, name):
    # Create output directory if doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    # Save image
    plt.clf()
    plt.imshow(img)
    plt.savefig(path+'/'+name)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect vehicles on images/videos', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cspace', default='RGB', help='color space used to convert input image.')
    parser.add_argument('--spatial_size', default=32, type=int, help='size used to resize input image.')
    parser.add_argument('--hist_bins', default=32, type=int, help='number of bins used for retrieve histograms of input image.')
    parser.add_argument('--orient', default=9, type=int, help='orientation for HOG.')
    parser.add_argument('--pixels_per_cell', default=8, type=int, help='number of pixels per cell for HOG.')
    parser.add_argument('--cells_per_block', default=2, type=int, help='number of cells per block for HOG.')
    parser.add_argument('--hog_channel', default=-1, type=int, choices=[0, 1, 2, -1], help='channels to use for HOG. -1 means all channels.')
    args = parser.parse_args()

    if os.path.exists('classifier.pkl') and os.path.exists('scaler.pkl'):
        # Retrieve the saved model that has already been trained
        print('Load classifier from disk...')
        clf = joblib.load('classifier.pkl')
        scaler = joblib.load('scaler.pkl')
    else:
        # Read in car and non-car images
        print("Read files in vehicles and non vehicles datasets...")
        vehicles = []
        non_vehicles = []
        for file in glob.glob('vehicles/**/*.png'):
            vehicles.append(mpimg.imread(file))
        for file in glob.glob('non-vehicles/**/*.png'):
            non_vehicles.append(mpimg.imread(file))

        # Extract features from vehicles and non vehicles dataset
        print("Extract features from datasets with:\n- cspace={}\n- spatial_size={}\n- hist_bins={}\n- orient={}\n- pixels_per_cell={}\n- cells_per_block={}\n- hog_channel={}\n".format(args.cspace, args.spatial_size, args.hist_bins, args.orient, args.pixels_per_cell, args.cells_per_block, args.hog_channel))
        vehicles_features = extract_features(vehicles, cspace=args.cspace, spatial_size=args.spatial_size, hist_bins=args.hist_bins, orient=args.orient, pixels_per_cell=args.pixels_per_cell, cells_per_block=args.cells_per_block, hog_channel=args.hog_channel)
        non_vehicles_features = extract_features(non_vehicles, cspace=args.cspace, spatial_size=args.spatial_size, hist_bins=args.hist_bins, orient=args.orient, pixels_per_cell=args.pixels_per_cell, cells_per_block=args.cells_per_block, hog_channel=args.hog_channel)

        # Create an array stack of feature vectors
        X = np.vstack((vehicles_features, non_vehicles_features)).astype(np.float64)
        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, 'scaler.pkl') # save to reuse it later
        X_scaled = scaler.transform(X)
        # Define the labels vector
        y = np.hstack((np.ones(len(vehicles_features)), np.zeros(len(non_vehicles_features))))
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=rand_state)
        # Use a linear SVC as classifier
        clf = SVC()
        t = time.time()
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'classifier.pkl') # save to reuse it later
        t2 = time.time()
        print("{0:.2f} seconds to train SVC".format(t2-t))
        print("Test Accuracy of SVC = {0:.3f}".format(clf.score(X_test, y_test)))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print("My SVC predictions: ", clf.predict(X_test[:n_predict]))
        print("The right labels: ", y_test[:n_predict])
        t2 = time.time()
        print("{0:.2f} seconds to predict {1:} samples\n".format(t2-t, n_predict))

    # Run the pipeline on each test image
    for file in glob.glob('test_images/*.jpg'):
        # Read image
        print("Finding vehicles on {}".format(file))
        img = mpimg.imread(file)

        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        img = img.astype(np.float32)/255

        # Retrieve sliding windows from image (and save it)
        windows = []
        sub_directory = file.split('/')[-1].split('.')[0]

        windows_small = sliding_windows(img, x_start_stop=[None,None], y_start_stop=[400,500], xy_window=(64,64), xy_overlap=(0.5,0.6))
        windows.append(windows_small)
        save_figure(draw_boxes(img, windows_small, color='random'), './output_images/'+sub_directory, 'small_windows.jpg')

        windows_medium = sliding_windows(img, x_start_stop=[None,None], y_start_stop=[400,530], xy_window=(80,80), xy_overlap=(0.5,0.6))
        windows.append(windows_medium)
        save_figure(draw_boxes(img, windows_medium, color='random'), './output_images/'+sub_directory, 'medium_windows.jpg')

        windows_large = sliding_windows(img, x_start_stop=[None,None], y_start_stop=[400,580], xy_window=(128,128), xy_overlap=(0.5,0.6))
        windows.append(windows_large)
        save_figure(draw_boxes(img, windows_large, color='random'), './output_images/'+sub_directory, 'large_windows.jpg')

        windows_x_large = sliding_windows(img, x_start_stop=[None,None], y_start_stop=[400,660], xy_window=(160,160), xy_overlap=(0.5,0.4))
        windows.append(windows_x_large)
        save_figure(draw_boxes(img, windows_x_large, color='random'), './output_images/'+sub_directory, 'x_large_windows.jpg')

        windows = [item for sublist in windows for item in sublist]

        # Search for car in windows using our classifier
        hot_windows = search_vehicles_in_windows(img, windows, scaler, clf, cspace=args.cspace, spatial_size=args.spatial_size, hist_bins=args.hist_bins, orient=args.orient, pixels_per_cell=args.pixels_per_cell, cells_per_block=args.cells_per_block, hog_channel=args.hog_channel)

        # Build a heat map from the detected boxes
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat, hot_windows)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 3)
        heatmap = np.clip(heat, 0, 255) # clip values from 0 to 255

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        result = draw_labeled_bboxes(np.copy(img), labels)
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(result)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.savefig('./output_images/'+sub_directory+'/result.jpg')
