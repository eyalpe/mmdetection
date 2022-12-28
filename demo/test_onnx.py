import os
import cv2
import time
import tabulate
import numpy as np
import onnxruntime
from typing import List, Tuple
#from mmcls.datasets.imagenet import ImageNet
from mmdet.datasets.coco import CocoDataset
import torch
from torchvision import transforms
import demo.tiny_yolo2_helpers as tiny_yolo2_helpers

class VerySimpleStopWatch():
    class WatchPoint():
        def __init__(self, name: str, t_prev_sample:float=time.time(), t_sample:float=time.time()) -> None:
            self.name = name
            self.prev_sample = t_prev_sample
            self.sample = t_sample
        
        def __str__(self) -> str:
            return f'{self.name}: t_sample={self.sample:0.3f}s, t_prev_sample={self.prev_sample:0.3f}s'
        
        def __repr__(self) -> str:
            return self.__str__()
    # end of class WatchPoint

    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        self.samples = [VerySimpleStopWatch.WatchPoint('t0')]
        self.t0 = self.samples[-1].sample
        self.t_last = self.samples[-1].sample
    
    def sample(self, name:str)->None:
        t = VerySimpleStopWatch.WatchPoint(name, self.t_last, time.time())
        self.t_last = t.sample
        self.samples.append(t)

    def get_samples(self)-> List[WatchPoint]:
        return self.samples 

    def get_diffs(self)-> List[Tuple[str, float]]:
        return [(t.name, t.sample-t.prev_sample) for t in self.samples]
    
    def print_watch_points(self) -> None:
        for t in self.get_samples():
            print(t)

    def print_diffs(self) -> None:
        data=[]
        for d in self.get_diffs():
            data.append((d[0], d[1]))
        print(tabulate.tabulate(data, headers=['name', 'duration'], floatfmt="0.3f"))
#end of class VerySimpleStopWatch



# model attributes:
stopwatch = VerySimpleStopWatch()
model_dir ="/mmdetection/checkpoints"
#model='/host_data/models/efficient_net/efficientnet-b8_3rdparty_8xb32-aa-advprop_in1k_20220119-297ce1b7_simplified.onnx'
#model='/host_data/models/efficient_net_b5/efficientnet-b5_3rdparty_8xb32-aa-advprop_in1k_20220119-f57a895a.onnx'
#model='/host_data/models/faster_mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.onnx'
#model='/host_data/models/test_retinanet_resnet101/retinanet-9.onnx'
#model='/host_data/models/retinanet_r50/mmdeploy/end2end.onnx'
#model='/host_data/models/tiny_yolov2/tinyyolov2-7.onnx'
model='/host_data/models/faster_mmdet/mmdeploy_dynamic/end2end.onnx'

img_path='/host_data/code_projects/mmdetection/demo/demo.jpg'
img_out_path='/host_data/code_projects/mmdetection/demo/demo_out.jpg'

session = onnxruntime.InferenceSession(model, None)
stopwatch.sample('load_model')

ort_inputs  = [{ "name": x.name, "shape": x.shape } for x in session.get_inputs()]
print(f'model inputs: {ort_inputs}')
ort_outputs = [{ "name": x.name, "shape": x.shape } for x in session.get_outputs()]
print(f'model outputs: {ort_outputs}')

ort_input = ort_inputs[0]
mean_rgb = [ 123.675, 116.28, 103.53 ]
std_rgb =  [ 58.395,  57.12,  57.375 ]
to_rgb=True

scalefactor = 1/(np.mean(std_rgb))

# Load and Preprocess the image
img = cv2.imread(img_path)
data_blob = cv2.dnn.blobFromImage(image=img, scalefactor = scalefactor, mean=mean_rgb, swapRB=to_rgb, crop=False)

# run inference
#dets, labels = session.run(['dets', 'labels'], {'input': data_blob})
random_blob = np.random.uniform(size=data_blob.shape).astype(np.float32)
#probs = session.run(output_names=['dets'], input_feed={'input': random_blob})[0]
_ = session.run(None, {ort_input["name"]: random_blob})
stopwatch.sample('warmup')
dets, labels = session.run(None, {ort_input["name"]: data_blob})
stopwatch.sample('inference')

# postprocess:

scores = dets[0,...,-1]

# scale bbs to original image dims:
img_h, img_w, _ = img.shape
scale_x = 1 # img_w / dsize_w
scale_y = 1 # img_h / dsize_h
threshold = 0.7

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
    label_name = CocoDataset.CLASSES[label]
    clr = CocoDataset.PALETTE[label]
    #rgb = int((2 ** 24 - 1) * label / labels.max())
    #clr = [(rgb&0xff0000) >> 16, (rgb&0xff00) >> 8, (rgb&0xff)]
    cv2.rectangle(im, tl, br, clr, 2, cv2.LINE_8)
    cv2.putText(im, f'{score:.2f} {label_name}', (l,t), cv2.FONT_HERSHEY_TRIPLEX, 0.6, clr, 1, cv2.LINE_AA)
    print(f'{i}: class: {label_name}(id: {label}), score: {score:.4f}, bb:[tl:{tl}, br:{br}]')
stopwatch.sample('postprocess')    
cv2.imwrite(img_out_path, im)
stopwatch.sample('save_image1')

### same thing but with cropped image:

# Load and Preprocess the image
prev_h, prev_w, _ = img.shape
new_h = int(prev_h / 2)
new_w = int(prev_w / 2)
img = img[:new_h, :new_w, :]
data_blob = cv2.dnn.blobFromImage(image=img, scalefactor = scalefactor, mean=mean_rgb, swapRB=to_rgb, crop=False)
stopwatch.sample('crop_and_preprocess')

# run inference
dets, labels = session.run(None, {ort_input["name"]: data_blob})
stopwatch.sample('inference_cropped')

# postprocess:
scores = dets[0,...,-1]

# scale bbs to original image dims:
img_h, img_w, _ = img.shape
scale_x = 1 # img_w / dsize_w
scale_y = 1 # img_h / dsize_h
threshold = 0.7

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
    label_name = CocoDataset.CLASSES[label]
    clr = CocoDataset.PALETTE[label]
    #rgb = int((2 ** 24 - 1) * label / labels.max())
    #clr = [(rgb&0xff0000) >> 16, (rgb&0xff00) >> 8, (rgb&0xff)]
    cv2.rectangle(im, tl, br, clr, 2, cv2.LINE_8)
    cv2.putText(im, f'{score:.2f} {label_name}', (l,t), cv2.FONT_HERSHEY_TRIPLEX, 0.6, clr, 1, cv2.LINE_AA)
    print(f'{i}: class: {label_name}(id: {label}), score: {score:.4f}, bb:[tl:{tl}, br:{br}]')
    
stopwatch.sample('postprocess_cropped')
p,ext = os.path.splitext(img_out_path)
img_out_path = f'{p}_cropped{ext}'
cv2.imwrite(img_out_path, im)
stopwatch.sample('save_image_cropped')

stopwatch.print_diffs()
