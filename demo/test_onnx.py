import cv2
import numpy as np
import onnxruntime


# model attributes:
model_dir ="/mmdetection/checkpoints"
model='/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.onnx'
img_path='/mmdetection/demo/demo.jpg'
img_out_path='/host_data/code_projects/mmdetection/demo/demo_out.jpg'

dsize_w = 608
dsize_h = 608
dsize =(dsize_w, dsize_h)
mean_rgb=[123.675, 116.28, 103.53]
std_rgb=[58.395, 57.12, 57.375]

# Load and Preprocess the image
img = cv2.imread(img_path)
data_blob = cv2.dnn.blobFromImage(image=img, scalefactor = 1/np.mean(std_rgb), size=dsize, mean=mean_rgb, swapRB=True)

# run inference
session = onnxruntime.InferenceSession(model, None)
dets, labels = session.run(['dets', 'labels'], {'input': data_blob})
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
