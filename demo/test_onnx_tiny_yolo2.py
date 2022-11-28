import cv2
import time
import tabulate
import numpy as np
import onnxruntime
from typing import List, Tuple
import tiny_yolo2_helpers

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
        diffs = [(t.name, t.sample-t.prev_sample) for t in self.samples]
        diffs.append(('t_total', time.time() - self.t0))
        return diffs
    
    def print_watch_points(self) -> None:
        for t in self.get_samples():
            print(t)

    def print_diffs(self) -> None:
        data=[]
        for d in self.get_diffs():
            data.append((d[0], d[1]))
        print(tabulate.tabulate(data, headers=['name', 'duration'], floatfmt="0.3f"))
#end of class VerySimpleStopWatch

if __name__ == '__main__':
  stopwatch = VerySimpleStopWatch()

  model=tiny_yolo2_helpers.model
  img_path=tiny_yolo2_helpers.img_path
  img_out_path=tiny_yolo2_helpers.img_out_path

  # Load model and verify its properties:
  session = onnxruntime.InferenceSession(model, None)
  ort_input = { "name": session.get_inputs()[0].name, "shape": session.get_inputs()[0].shape }
  print(f'model input: {ort_input}')
  assert [x for x in ort_input["shape"][1:]] == [x for x in tiny_yolo2_helpers.expected_model_input_shape[1:]]
  stopwatch.sample('load_model')

  # Warmup
  random_blob = np.random.uniform(size=tiny_yolo2_helpers.expected_model_input_shape).astype(np.float32)
  _ = session.run(None, {ort_input["name"]: random_blob})
  stopwatch.sample('warmup')

  # Load and pre-process the image
  img = cv2.imread(img_path)
  data_blob = tiny_yolo2_helpers.preprocess(img)
  blob_fname = '/tmp/myblob_{}.raw'.format('x'.join([str(x) for x in data_blob.shape]))
  data_blob.astype(np.float32).tofile(blob_fname)
  stopwatch.sample("preprocess")

  # Inference
  predictions = session.run(None, {ort_input["name"]: data_blob})
  stopwatch.sample('inference')
  
  # Post-process:
  yolo_results = tiny_yolo2_helpers.postprocess(predictions[0][0])
  yolo_rescaled_results = tiny_yolo2_helpers.rescale_results(yolo_results, img_h=img.shape[0], img_w=img.shape[1])
  stopwatch.sample('postprocess')

  # save the results
  tiny_yolo2_helpers.print_results(yolo_results)
  tiny_yolo2_helpers.print_results(yolo_rescaled_results)

  im = img.copy()
  for yolo_res in yolo_rescaled_results:
    bbox, score, class_name = yolo_res
    l, t, r, b = [int(coord) for coord in bbox]
    tl = (l, t)
    br = (r, b)
    
    # label to rgb:
    clr = tiny_yolo2_helpers.CLASS_COLOR[class_name]
    cv2.rectangle(im, tl, br, clr, 2, cv2.LINE_8)
    cv2.putText(im, f'{score:.2f} {class_name}', (l,t), cv2.FONT_HERSHEY_TRIPLEX, 0.6, clr, 1, cv2.LINE_AA)    
  cv2.imwrite(img_out_path, im)
  stopwatch.sample("dump_results")
  stopwatch.print_diffs()
