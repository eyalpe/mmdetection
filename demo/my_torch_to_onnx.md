# torch to onnx
1. download pretrained torch model from mmdetection's zoo
```shell
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest . 
```

2. Run the conversion script:
```shell
python tools/deployment/pytorch2onnx.py     configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py     checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth     --output-file checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.onnx     --input-img demo/demo.jpg     --test-img tests/data/color.jpg     --shape 608 608     --show     --verify
```
> ### Note:
> ---
> If the conversion script fails, you may need to run the following steps to install protobuf and its libs:
> ```shell
> apt update
> apt install protobuf
> apt install libprotobuf
> apt install protobuf-compiler
> ```

3. Create an environment to run the test script:
```shell
python3 -m venv onnx_env
source onnx_env/bin/activate
python3 -m pip install --upgrade pip
pip install onnx onnxruntime opencv-python
```

4. Run the test script:
```shell
python demo/test_onnx.py
```

5. check the `demo/demo_out.jpg` image to get impressed by the result
