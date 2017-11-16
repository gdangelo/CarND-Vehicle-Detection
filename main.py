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
from moviepy.editor import VideoFileClip

# Define a function to compute binned color features
def bin_spatial(img, size=32):
    # Resize the image and use ravel() to flat the array
    color1 = cv2.resize(img[:,:,0], (size,size)).ravel()
    color2 = cv2.resize(img[:,:,1], (size,size)).ravel()
    color3 = cv2.resize(img[:,:,2], (size,size)).ravel()
    return np.hstack((color1, color2, color3))

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
def extract_features(imgs, cspace='RGB', spatial_size=32, hist_bins=32, hist_range=(0, 256), orient=9, pixels_per_cell=8, cells_per_block=2, hog_channel=-1):
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
            for channel in range(feature_img.shape[2]):
                hog_features.extend(get_hog_features(feature_img[:,:,channel], orient, pixels_per_cell, cells_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_img[:,:,hog_channel], orient, pixels_per_cell, cells_per_block, vis=False, feature_vec=True)
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
def search_vehicles_in_windows(img, windows, scaler, clf, cspace='RGB', spatial_size=32, hist_bins=32, orient=9, pixels_per_cell=8, cells_per_block=2, hog_channel=-1):
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

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, scaler, clf, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=32, hist_bins=32, cspace='RGB', hog_channel=-1):

    # Array of bounding boxes where cars were detected
    bboxes = []

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]

    # Apply color conversion if other than 'RGB'
    if cspace == 'HSV':
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
    elif cspace == 'HSL':
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSL)
    elif cspace == 'YUV':
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
    elif cspace == 'LUV':
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
    elif cspace == 'YCrCb':
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else:
        ctrans_tosearch = np.copy(img_tosearch)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    if hog_channel == -1:
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    else:
        ch1 = ctrans_tosearch[:,:,hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == -1:
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if hog_channel == -1:
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = clf.predict(test_features)
            #test_confidence = clf.decision_function(test_features)
        
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return bboxes

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
    heatmap[heatmap < threshold] = 0
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

def process_img(img, scaler, clf, cspace, spatial_size, hist_bins, orient, pixels_per_cell, cells_per_block, hog_channel):
    # Retrieve windows where cars have been detected
    windows = []
    windows.append(sliding_windows(img, x_start_stop=[None, None], y_start_stop=[400, 640], xy_window=(128, 128), xy_overlap=(0.5, 0.5)))
    windows.append(sliding_windows(img, x_start_stop=[None, None], y_start_stop=[400, 600], xy_window=(96, 96), xy_overlap=(0.5, 0.5)))
    windows.append(sliding_windows(img, x_start_stop=[None, None], y_start_stop=[390, 540], xy_window=(80, 80), xy_overlap=(0.5, 0.5)))
    windows = [item for sublist in windows for item in sublist]

    hot_windows = search_vehicles_in_windows(img, windows, scaler, clf, cspace=cspace, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, hog_channel=hog_channel)

    # Build a heat map from the detected boxes
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255) # clip values from 0 to 255

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    result = draw_labeled_bboxes(np.copy(img), labels)
    return result

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect vehicles on images/videos', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cspace', default='YCrCb', help='color space used to convert input image.')
    parser.add_argument('--spatial_size', default=32, type=int, help='size used to resize input image.')
    parser.add_argument('--hist_bins', default=32, type=int, help='number of bins used for retrieve histograms of input image.')
    parser.add_argument('--orient', default=9, type=int, help='orientation for HOG.')
    parser.add_argument('--pixels_per_cell', default=8, type=int, help='number of pixels per cell for HOG.')
    parser.add_argument('--cells_per_block', default=2, type=int, help='number of cells per block for HOG.')
    parser.add_argument('--hog_channel', default=-1, type=int, choices=[0, 1, 2, -1], help='channels to use for HOG. -1 means all channels.')
    args = parser.parse_args()

    if os.path.exists('classifier.pkl') and os.path.exists('scaler.pkl'):
        # Retrieve the saved model that has already been trained
        print('Load classifier from disk...\n')
        clf = joblib.load('classifier.pkl')
        scaler = joblib.load('scaler.pkl')

    else:
        # Read in car and non-car images
        print("Read files in vehicles and non vehicles datasets...")
        vehicles = []
        non_vehicles = []
        for file in glob.glob('vehicles/**/*.png', recursive=True):
            vehicles.append(mpimg.imread(file))
        for file in glob.glob('non-vehicles/**/*.png', recursive=True):
            non_vehicles.append(mpimg.imread(file))

        print("{} car images".format(len(vehicles)))
        print("{} non-car images".format(len(non_vehicles)))

        # Extract features from vehicles and non vehicles dataset
        print("\nExtract features from datasets with:\n- cspace={}\n- spatial_size={}\n- hist_bins={}\n- orient={}\n- pixels_per_cell={}\n- cells_per_block={}\n- hog_channel={}\n".format(args.cspace, args.spatial_size, args.hist_bins, args.orient, args.pixels_per_cell, args.cells_per_block, args.hog_channel))
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
    test_image_dir = './test_images/'
    output_image_dir = './output_images/'
    if not os.path.isdir(output_image_dir):
        os.makedirs(output_image_dir)

    for file in os.listdir(test_image_dir):
        # Read image
        print("Finding vehicles on {}".format(file))
        img = mpimg.imread(test_image_dir + file)

        # Retrieve windows where cars have been detected
        windows = []
        windows.append(find_cars(img, 400, 480, 0.5, scaler, clf, args.orient, args.pixels_per_cell, args.cells_per_block, args.spatial_size, args.hist_bins, args.cspace, args.hog_channel))
        windows.append(find_cars(img, 400, 560, 1.0, scaler, clf, args.orient, args.pixels_per_cell, args.cells_per_block, args.spatial_size, args.hist_bins, args.cspace, args.hog_channel))
        windows.append(find_cars(img, 400, 660, 2.0, scaler, clf, args.orient, args.pixels_per_cell, args.cells_per_block, args.spatial_size, args.hist_bins, args.cspace, args.hog_channel))
        windows = [item for sublist in windows for item in sublist]

        img_bboxes = draw_boxes(img, windows, color='random')

        # Build a heat map from the detected boxes
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat, windows)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 2)
        heatmap = np.clip(heat, 0, 255) # clip values from 0 to 255

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        result = draw_labeled_bboxes(np.copy(img), labels)

        # Display detected boxes and heatmap
        plt.axis('off')
        plt.imshow(img_bboxes)
        plt.savefig(output_image_dir+file.split('.')[0]+'_detected_windows.jpg')
        plt.imshow(result)
        plt.savefig(output_image_dir+file.split('.')[0]+'_car_position.jpg')
        plt.imshow(heatmap, cmap='hot')
        plt.savefig(output_image_dir+file.split('.')[0]+'_heatmap.jpg')

    # Run the pipeline on each test video
    test_video_dir = './test_videos/'
    output_video_dir = './output_videos/'
    if not os.path.isdir(output_video_dir):
        os.makedirs(output_video_dir)

    '''for file_name in os.listdir(test_video_dir):
        print("\nRun pipeline for '" + file_name + "'...")
        video_input = VideoFileClip(test_video_dir + file_name)
        processed_video = video_input.fl_image(lambda x: process_img(x, scaler, clf, args.cspace, args.spatial_size, args.hist_bins, args.orient, args.pixels_per_cell, args.cells_per_block, args.hog_channel))
        processed_video.write_videofile(output_video_dir + file_name, audio=False)'''