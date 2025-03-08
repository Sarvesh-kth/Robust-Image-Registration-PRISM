import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from scipy.ndimage import zoom
import time

# record start time
# Hyper Parameters
brlt = 200
dayThreshold = 100
motion_rate = 251
hyper_path = "output_"
decay_factor = 0.05
blur_kernel_size = 3

DEVICE = 'cuda'


def load_image(imfile):
    image = cv2.imread(imfile)
    img = np.array(image.astype(np.uint8))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def optical_flow_filter(optical_flow, contours):
    height, width, _ = optical_flow.shape
    # Initialize an array to store the filtered optical flow
    filtered_optical_flow = np.zeros((height, width, 2))

    # overall
    overall_motion_magnitude = np.linalg.norm(filtered_optical_flow, axis=2)
    overAllThres = np.mean(overall_motion_magnitude) + np.std(overall_motion_magnitude)
    overAllLowerThres = np.mean(overall_motion_magnitude)
    # Iterate over sections
    for k in contours:
        j, i, width, height = cv2.boundingRect(k)
        section_size = (height, width)

        section = optical_flow[i:i + section_size[0], j:j + section_size[1]]

        # Calculate the motion magnitude for the current section
        motion_magnitude = np.linalg.norm(section, axis=2)

        # Calculate the dynamic threshold for the section as the median of motion magnitudes
        section_threshold = 0

        # Create a mask of valid motion within the section based on the local threshold
        valid_motion_mask = np.logical_and(motion_magnitude > overAllLowerThres,
                                           np.logical_or(motion_magnitude >= section_threshold,
                                                         motion_magnitude > overAllThres))

        # Apply the mask to keep only the valid motion vectors within the section
        filtered_section = section.copy()
        filtered_section[~valid_motion_mask] = 0

        # Store the filtered section back into the output array
        filtered_optical_flow[i:i + section_size[0], j:j + section_size[1]] = filtered_section

    return filtered_optical_flow


def averagingFlow(frames, magnitudes, isDay):
    sum_array = np.zeros(frames[0].shape, dtype=np.float32)
    count_array = np.zeros(frames[0].shape, dtype=np.float32)

    # initialize empty max image
    max_image = np.zeros(frames[0].shape, dtype=np.float32)
    height, width = frames[0].shape[:2]

    # initialize accumulated optical flow
    accof = np.zeros((height, width), dtype=np.float32)

    # Iterate through frames and magnitudes
    nonzero_mask = np.any(magnitudes[0] == 0, axis=2)
    for i, (frame, magnitude) in enumerate(zip(frames, magnitudes)):
        # find maximum pixel values across all frames
        max_image = np.maximum(max_image, frame)

        grayfr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # makes an accumulated mask of bright light
        _, bright = cv2.threshold(grayfr, brlt, 255, cv2.THRESH_BINARY)
        accof = accof + bright

        # only has 0 and 255
        accof = np.clip(accof, 0, 255)

        #  makes an accumulated mask of optical flow
        nonzero_mask = np.logical_or(nonzero_mask, np.max(magnitude < motion_rate, axis=2))

        # Update sum_array and count_array only for non-zero magnitude pixels
        blur_kernel_size = 7
        blur_frame = frame.copy()
        blur_frame = cv2.GaussianBlur(blur_frame, (blur_kernel_size, blur_kernel_size), 0)

        # for decay averaging
        # sum_array = (sum_array * (1 - decay_factor)) + (frame * nonzero_mask[:, :, np.newaxis])
        # count_array = (count_array * (1 - decay_factor)) + nonzero_mask[:, :, np.newaxis]

        # for normal averaging
        sum_array += blur_frame * nonzero_mask[:, :, np.newaxis]
        count_array += nonzero_mask[:, :, np.newaxis]

        # Avoid division by zero
    count_array[count_array == 0] = 1

    # to dilate the bright light mask to make more prominent
    kernel = np.ones((3, 3), np.uint8)
    # accof = cv2.dilate(accof, kernel, iterations=1)

    # reduce the mask to 1 or 0
    accof = 255 - accof
    accof = accof / 255.0

    # inverse mask
    accofInv = 1 - accof

    # Calculate the average
    average_image = sum_array / count_array
    # average_image = sum_array
    zeroMask = 1 - nonzero_mask

    # applies the mask for moving and not moving , not moving takes the last frame as reference frame
    average_image = average_image * nonzero_mask[:, :, np.newaxis] + frames[len(frames) - 1] * zeroMask[:, :,
                                                                                               np.newaxis]

    # mask for all 3 channels
    accofMask = np.repeat(accof[:, :, np.newaxis], 3, axis=2)
    accofInvMask = np.repeat(accofInv[:, :, np.newaxis], 3, axis=2)

    # NIGHT TIME

    # add the bright light mask
    if not isDay:
        average_image = average_image * accofMask + max_image * accofInvMask

    # adds the optical flow mask
    average_image = average_image * nonzero_mask[:, :, np.newaxis] + frames[len(frames) - 1] * zeroMask[:, :,
                                                                                               np.newaxis]

    # converts to np.uint8 from np.float32
    average_image = average_image[:, :, [2, 1, 0]] / 255.0
    average_image = average_image * 255.0
    average_image = average_image.astype(np.uint8)
    return average_image


def viz(img, flo, contours):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # flo = cv2.medianBlur(flo, ksize=5)
    flo = optical_flow_filter(flo, contours)

    flo = flow_viz.flow_to_image(flo)
    img_flo = np.ascontiguousarray(img, dtype=np.uint8)

    return img_flo, flo


def demo(args, bounding_BOX, isDay):
    #Load the torch model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    pp = 1
    with torch.no_grad():
        downimages = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        origimages = glob.glob(os.path.join(args.curpath, '*.png')) + \
                     glob.glob(os.path.join(args.curpath, '*.jpg'))
        i = 0
        arrayImg = []
        arrayFlo = []
        #get both original and down scaled images
        for imfile1, imfile2, imfile3 in zip(downimages[:-1], downimages[1:],origimages[:-1]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            curimage = cv2.imread(imfile3)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            #Run the raft model
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            #Visualize the output
            imgOut, floOut = viz(image1, flow_up, contours=bounding_BOX[i])

            #Upscale the output flow to fit original image
            h = int(curimage.shape[0])
            w = int(curimage.shape[1])
            flo1 = floOut[:, :, 0]
            flo2 = floOut[:, :, 1]
            flo3 = floOut[:, :, 2]
            zf = (2, 2)  # Upscale by a factor of 2 in both dimensions
            flo1 = zoom(flo1, zf, order=1)
            flo2 = zoom(flo2, zf, order=1)
            flo3 = zoom(flo3, zf, order=1)

            wdiff = int((flo1.shape[1] - curimage.shape[1])/2)
            hdiff = int((flo1.shape[0] - curimage.shape[0]) / 2)
            nflo1 = np.zeros((h,w))

            nflo1 = flo1[hdiff:flo1.shape[0]-hdiff, wdiff:flo1.shape[1]-wdiff]
            nflo2 = np.zeros((h,w))
            nflo2 = flo2[hdiff:flo1.shape[0]-hdiff, wdiff:flo1.shape[1]-wdiff]
            nflo3 = np.zeros((h,w))
            nflo3 = flo3[hdiff:flo1.shape[0]-hdiff, wdiff:flo1.shape[1]-wdiff]

            curimage = curimage[:, :, [2, 1, 0]]

            newOut = np.stack((nflo1, nflo2, nflo3), axis=-1)
            arrayImg.append(curimage)
            arrayFlo.append(newOut)
            i += 1


        averageFrame = averagingFlow(arrayImg, arrayFlo,isDay)
        

        # Save the image
        cv2.imwrite(hyper_path+"Averageframe.png", averageFrame)




def getContours():
    images = glob.glob(os.path.join(args.path, '*.png')) + \
             glob.glob(os.path.join(args.path, '*.jpg'))
    images = sorted(images)
    
    frames = []
    for i in images:
        img = cv2.imread(i)
        frames.append(img)
    
    #Median Frame is found  and converted to grayscale
    frame_median = np.median(frames, axis=0).astype(dtype=np.uint8)
    gray_frame_median = cv2.cvtColor(frame_median, cv2.COLOR_BGR2GRAY)

    #Depending on the brightness values, we decide if its day or night shot
    average_brightness = cv2.mean(gray_frame_median)[0]
    isDay = True
    if average_brightness > dayThreshold:
        isDay = True
    else:
        isDay = False    
    return gray_frame_median, isDay


def getBounding(gray_frame_median):
    images = glob.glob(os.path.join(args.path, '*.png')) + \
             glob.glob(os.path.join(args.path, '*.jpg'))
    images = sorted(images)
    frames = []
    for i in images:
        frames.append(cv2.imread(i))
    countoursAll = []
    for frame in frames:
        #Using the median frame as reference we see how much a area has moved and draw bounding boxes
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dframe = cv2.absdiff(gray_frame, gray_frame_median)
        blur_frame = cv2.GaussianBlur(dframe, (11, 11), 0)
        ret, threshold_frame = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        (contours, _) = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        countoursAll.append(contours)
    return countoursAll

def deletefiles(directory):
    try:      # Iterate through all files in the directory and delete them
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print("An error occurred: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default="models/raft-things.pth")
    parser.add_argument('--curpath', help="dataset for evaluation", default="demo_frames")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--path', help="directory for downscaled images", default="new_frames")

    args = parser.parse_args()

    #downscaling
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    else:
        deletefiles(args.path)

    # List all files in the path directory

    files = os.listdir(args.curpath)

    for file in files:
        # Check if the file is an image (you can add more extensions as needed)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Load the image
            image_path = os.path.join(args.curpath, file)
            imaging = Image.open(image_path)
            img = imaging.copy()
            os.remove(image_path)
            width, height = img.size
            target_dimension = 720

            #Based on the smaller dimension we reduce it by half
            if height > width:
                temp = width
                width = target_dimension
                height = int(height * float(target_dimension / temp))
                if height % 2 == 1:
                    if ((height + 1)/2) % 2 == 1:
                        height = height+3
                    else:
                        height = height + 1
            else:
                temp = height
                height = target_dimension
                width = int(width * float(target_dimension / temp))
                if width % 2 == 1:
                    if ((width + 1) / 2) % 2 == 1:
                        width = width + 3
                    else:
                        width = width + 1

            img = img.resize((width, height))

            img.save(image_path)

            new_width = int(width / 2)  # Adjust the scaling factor as needed
            new_height = int(height / 2)
            downscaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Save the downscaled image to newpath
            new_image_path = os.path.join(args.path, file)
            downscaled_img.save(new_image_path)
    print("Starting execution ")
    start = time.time()

    gray_frame_median, isDay = getContours()
    bounding_BOX = getBounding(gray_frame_median)
    demo(args, bounding_BOX,isDay)
    end = time.time()
    print("The time of execution of above program is :",
          (end - start) * 10 ** 3, "ms")

