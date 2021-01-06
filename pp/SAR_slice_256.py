# import the necessary packages
import os
import cv2
import time
import argparse
import numpy as np
import xml.etree.ElementTree as Et

from pascal_voc_writer import Writer
from PIL import Image
from PIL import ImageDraw


def load_xml(xml_path, draw):
    xml = open(xml_path, "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    bounding_box =[]

    objects = root.findall("object")
    print("Objects Description")
    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        object_box = [(int(xmin), int(ymin)), (int(xmax), int(ymax)), name]

        bounding_box.append(object_box)

        # draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red")

    return bounding_box


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def make_new_xml(new_img_dir, rectangle, box_list, cropped_xml_name):
    writer = Writer(new_img_dir, 160, 160)
    
    for box in box_list:
        new_xmin = box[0][0] - rectangle[0][0]
        new_ymin = box[0][1] - rectangle[0][1]
        new_xmax = new_xmin + box[1][0] - box[0][0]
        new_ymax = new_ymin + box[1][1] - box[0][1]
        writer.addObject(box[2], new_xmin, new_ymin, new_xmax, new_ymax)
    cropped_xml_name = cropped_xml_name + ".xml"
    writer.save(os.path.join(xml_save_dir, cropped_xml_name))
    # 저장 경로 수정!



def slice_image(image_dir, xml_dir, image_list, is_train=True):
    (winW, winH) = (args["window"], args["window"])

    for i in image_list:
        # image path
        i_path = os.path.join(image_dir, i)
        # find the annotation file using image name
        xml_file = ann_list[ann_list.index(".".join([i.split(".")[0], "xml"]))]

        pil_image = Image.open(i_path)
        draw = ImageDraw.Draw(pil_image)

        # draw bounding boxes from xml file
        bounding_box = load_xml(os.path.join(xml_dir, xml_file), draw)

        # convert PIL image to cv2 image
        image = np.array(pil_image)
        # load the image and define the window width and height
        # image = cv2.imread(i_path)
        id_num = 0

        for (x, y, window) in sliding_window(image, stepSize=args["step"], windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != 160 or window.shape[1] != 160:
              continue
                
            # since we do not have a classifier, we'll just draw the window
            clone = image.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

            rectangle = [(x, y), (x + winW, y + winH)]
            # if bounding_box is in sliding window, save image
            box_list = []
            for box in bounding_box:
                if (box[0][0] >= rectangle[0][0])\
                        and (box[0][1] >= rectangle[0][1])\
                        and (box[1][0] <= rectangle[1][0])\
                        and (box[1][1] <= rectangle[1][1]):
                    box_list.append(box)
            
            if is_train:
                if len(box_list) > 0:
                    cropped_image = window
                    id_num += 1
                    cropped_img_name = i.split(".")[0] + "_" + str(id_num)
                    cropped_img_dir = os.path.join(img_save_dir, (cropped_img_name + ".jpg"))
                    make_new_xml(cropped_img_dir, rectangle, box_list, cropped_img_name)
                    print("New Annotation saved!")
                    crop_image = Image.fromarray(cropped_image)
                    crop_image.save(cropped_img_dir)
                    print("New image saved!")
            else:
                cropped_image = window
                id_num += 1
                cropped_img_name = i.split(".")[0] + "_" + str(id_num)
                cropped_img_dir = os.path.join(img_save_dir, (cropped_img_name + ".jpg"))
                make_new_xml(cropped_img_dir, rectangle, box_list, cropped_img_name)
                print("New Annotation saved!")
                crop_image = Image.fromarray(cropped_image)
                crop_image.save(cropped_img_dir)
                print("New image saved!")


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, type=str, help="Directory of the dataset")
ap.add_argument("-w", "--window", required=True, type=int, help="Input of window size to slice image")
ap.add_argument("-s", "--step", required=True, type=int, help="Input of step size to slice image")
args = vars(ap.parse_args())

# get root directory, image list directory
dataset_path = args["dataset"]

IMAGE_FOLDER = "img"
ANNOTATIONS_FOLDER = "label"

ann_dir, ann_flist, ann_list = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))
img_dir, img_flist, img_list = next(os.walk(os.path.join(dataset_path, IMAGE_FOLDER)))

# make a new directory to save preprocessed data
save_dir = os.path.join(dataset_path, str(args["window"]))
img_save_dir = os.path.join(save_dir, IMAGE_FOLDER)
xml_save_dir = os.path.join(save_dir, ANNOTATIONS_FOLDER)

if not(os.path.isdir(save_dir)):
    os.makedirs(save_dir)
if not(os.path.isdir(img_save_dir)):
    os.makedirs(img_save_dir)
if not(os.path.isdir(xml_save_dir)):
    os.makedirs(xml_save_dir)

slice_image(img_dir, ann_dir, img_list)
