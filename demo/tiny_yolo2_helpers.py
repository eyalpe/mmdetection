import cv2
import numpy as np
from typing import List, Tuple

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS =  [(254.0, 254.0, 254), (239.88888888888889, 211.66666666666669, 127),
           (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
           (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
           (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
           (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254), 
           (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
           (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
           (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254), 
           (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
           (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127)]
CLASS_COLOR = {name:color for name, color in zip(CLASSES, COLORS)}

model='/host_data/models/tiny_yolov2/tinyyolov2-7.onnx'
img_path='/host_data/code_projects/mmdetection/demo/dog.jpg'
img_out_path='/host_data/code_projects/mmdetection/demo/dog_out.jpg'

expected_model_input_shape = (1, 3, 416, 416)
expected_model_dsize_h = expected_model_input_shape[2]
expected_model_dsize_w = expected_model_input_shape[3]
expected_model_dsize = (expected_model_dsize_h, expected_model_dsize_w)

iou_threshold = 0.3
score_threshold = 0.5

def preprocess(img_mat):
  data_blob = cv2.resize(img_mat, expected_model_dsize, cv2.INTER_CUBIC)
  data_blob = data_blob.transpose([2,0,1])
  data_blob = np.expand_dims(data_blob, axis=0)
  return data_blob.astype(np.float32)


# sigmoid helper
def sigmoid(x):
  return 1. / (1. + np.exp(-x))


# softmax header
def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def iou(boxA, boxB):
  # boxA = boxB = [x1,y1,x2,y2]

  # Determine the coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  # Compute the area of intersection
  intersection_area = (xB - xA + 1) * (yB - yA + 1)
 
  # Compute the area of both rectangles
  boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
  # Compute the IOU
  iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

  return iou


def non_maximal_suppression(thresholded_predictions, iou_threshold):
  nms_predictions = []

  # Add the best B-Box because it will never be deleted
  nms_predictions.append(thresholded_predictions[0])

  # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
  # thresholded_predictions[i][0] = [x1,y1,x2,y2]
  i = 1
  while i < len(thresholded_predictions):
    n_boxes_to_check = len(nms_predictions)
    #print('N boxes to check = {}'.format(n_boxes_to_check))
    to_delete = False

    j = 0
    while j < n_boxes_to_check:
        curr_iou = iou(thresholded_predictions[i][0],nms_predictions[j][0])
        if(curr_iou > iou_threshold):
            to_delete = True
        #print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
        j = j+1

    if to_delete == False:
        nms_predictions.append(thresholded_predictions[i])
    i = i+1

  return nms_predictions


# expoected grid.shape is (25,13,13)
def postprocess(grid):
  numClasses = len(CLASSES)
  thresholded_predictions = []
  tcs = []
  anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
  for row in range(13):
    for col in range(13):
      for b in range(5):
        channel = b * (numClasses + 5)
        tx = grid[channel    , row, col]
        ty = grid[channel + 1, row, col]
        tw = grid[channel + 2, row, col]
        th = grid[channel + 3, row, col]
        tc = grid[channel + 4, row, col]
        #print(f'[cy, cx, b] = {[cy, cx, b]} => [tx, ty, tw, th, tc] = {[tx,ty, tw, th, tc]}')
        center_x = (float(col) + sigmoid(tx)) * 32.0
        center_y = (float(row) + sigmoid(ty)) * 32.0
        roi_w = np.exp(tw) * anchors[2*b + 0] * 32.0
        roi_h = np.exp(th) * anchors[2*b + 1] * 32.0
        final_confidence = sigmoid(tc)
        tcs.append(final_confidence)
        # Find best class
        class_predictions = grid[channel+5:channel+5+numClasses,row, col]
        class_predictions = softmax(class_predictions)
        class_predictions = tuple(class_predictions)
        best_class_index = class_predictions.index(max(class_predictions))
        best_class_score = class_predictions[best_class_index]
        # Compute the final coordinates on both axes
        left   = int(center_x - (roi_w/2.))
        right  = int(center_x + (roi_w/2.))
        top    = int(center_y - (roi_h/2.))
        bottom = int(center_y + (roi_h/2.))
        
        final_class_score = final_confidence * best_class_score
        if(final_class_score > score_threshold):
          thresholded_predictions.append([[left,top,right,bottom], final_class_score, CLASSES[best_class_index]])

  # Sort the B-boxes by their final score
  thresholded_predictions.sort(key=lambda tup: tup[1],reverse=True)
  nms_predictions = non_maximal_suppression(thresholded_predictions, iou_threshold)
  return nms_predictions


def print_results(yolo2_results):
  print('Printing {} B-boxes results ...'.format(len(yolo2_results)))
  for i, result in enumerate(yolo2_results):
      print('  B-Box {} : {}'.format(i+1, result))


def rescale_results(yolo2_results, img_h, img_w):
  rescaled_results=[]
  for result in yolo2_results:
    bbox, score, class_name = result
    x1, y1, x2, y2 = bbox
    scale_w = img_w / expected_model_dsize_w
    scale_h = img_h / expected_model_dsize_h
    rescaled_bbox = [x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h]
    rescaled_results.append([rescaled_bbox, score, class_name])
  return rescaled_results



if __name__ == '__main__':
    print("=== Start. ===")
    print(model)
    print("=== End ===")
