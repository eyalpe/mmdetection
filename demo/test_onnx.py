import cv2
import numpy as np
import onnxruntime
import time
from typing import *

def timing_stats_str(durations: List[float]) -> str:
    durs = np.asarray(durations)
    N=len(durations)
    avg = durs.mean()
    std = durs.std()
    return f'#{N} samples, AVG: {avg:.03f}s. STD: {std:.03f}s. min: {durs.min():.03f}s. max: {durs.max():.03f}s.'


NUM_WARMUP_ITERATIONS = 1
NUM_TEST_ITERATIONS = 10

# model attributes:
model_dir = "/mmdetection/checkpoints"
model = '/host_data/models/faster_mmdet/mmdeploy_dynamic/end2end.onnx'
#'~/models/faster_mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.onnx'
img_path = '/host_data/code_projects/mmdetection/demo/demo.jpg'
img_out_path = '/host_data/code_projects/mmdetection/demo/demo_out.jpg'

mean_rgb = [123.675, 116.28, 103.53]
std_rgb = [58.395, 57.12, 57.375]

# Load and Preprocess the image
img = cv2.imread(img_path)
dsize_w = 1920  # img.shape[1] #608
dsize_h = 1080  # img.shape[0] #608
dsize = (dsize_w, dsize_h)
data_blob = cv2.dnn.blobFromImage(image=img, scalefactor = 1/np.mean(std_rgb), size=dsize, mean=mean_rgb, swapRB=True)

# run inference
warmup_duration_samples = []
inference_duration_samples = []
session_providers = ['TensorrtExecutionProvider',
                     ('CUDAExecutionProvider',
                      {
                          'device_id': 1,
                          'cudnn_conv_algo_search': 'HEURISTIC',
                      }
                     ),
                     'CPUExecutionProvider']
session = onnxruntime.InferenceSession(model, providers=session_providers)

for i in range(NUM_WARMUP_ITERATIONS + NUM_TEST_ITERATIONS):
    t_start = time.time()
    dets, labels = session.run(['dets', 'labels'], {'input': data_blob})
    t_end = time.time()
    current_measurment_list = warmup_duration_samples if i < NUM_WARMUP_ITERATIONS else inference_duration_samples
    current_measurment_list.append(t_end - t_start)

scores = dets[0,...,-1]

# scale bbs to original image dims:
img_h, img_w, _ = img.shape
scale_x = img_w / dsize_w
scale_y = img_h / dsize_h
threshold = 0.5

lefts =   dets[0,...,0] * scale_x
tops =    dets[0,...,1] * scale_y
rights =  dets[0,...,2] * scale_x
bottoms = dets[0,...,3] * scale_y

im = img.copy()
for i, score in enumerate(scores):
    if score < threshold:
        break
    l = int(lefts[i])
    t = int(tops[i])
    r = int(rights[i])
    b = int(bottoms[i])
    tl = (l, t)
    br = (r, b)
    
    # label to rgb:
    label = labels[0,i]
    rgb = int((2 ** 24 - 1) * label / labels.max())
    clr = [(rgb&0xff0000) >> 16, (rgb&0xff00) >> 8, (rgb&0xff)]
    cv2.rectangle(im, tl, br, clr, 1, cv2.LINE_8)
    cv2.putText(im, f'{score:.2f}#{label}', (l,t), cv2.FONT_HERSHEY_DUPLEX, 0.6, clr, 1, cv2.LINE_AA)
    print(f'{i}: class-id: {label}, score: {score:.4f}, bb:[tl:{tl}, br:{br}]')
    
cv2.imwrite(img_out_path, im)

# print timing results:
heading = f'Timing statistics for an input data blob of shape: {data_blob.shape}'
print()
print(heading)
print(len(heading)*'=')
print(f'Warmap iterations stats:  {timing_stats_str(warmup_duration_samples)}')
print(f'Inference stats:         {timing_stats_str(inference_duration_samples)}')
