import sys
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

def main():
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
  blob_fname = '{}_{}.raw'.format(img_path,'x'.join([str(x) for x in data_blob.shape]))
  #data_blob.astype(np.float32).tofile(blob_fname)
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


def read_dlc_results_from_file(dlc_res_file):
  expected_grid_shape = [13,13,125]
  dlc_out = np.fromfile(dlc_res_file, dtype=np.float32)
  dlc_out = dlc_out.reshape(expected_grid_shape).transpose([2,0,1])
  return dlc_out


def postprocess_dlc_output(dlc_result_file):
    dlc_out = read_dlc_results_from_file(dlc_res_file=dlc_result_file)
    return tiny_yolo2_helpers.postprocess(dlc_out)


def draw_yolo_results_on_image(yolo_results, image_copy):
    im = image_copy
    for yolo_res in yolo_results:
        bbox, score, class_name = yolo_res
        l, t, r, b = [int(coord) for coord in bbox]
        tl = (l, t)
        br = (r, b)
        clr = tiny_yolo2_helpers.CLASS_COLOR[class_name]  # class id to rgb
        cv2.rectangle(im, tl, br, clr, 2, cv2.LINE_8)
        cv2.putText(im, f'{score:.2f} {class_name}', tl, cv2.FONT_HERSHEY_TRIPLEX, 0.6, clr, 1, cv2.LINE_AA)
    return im


if __name__ == "__main__":
    # main()
    # sys.exit(0)
    img = cv2.imread(tiny_yolo2_helpers.img_path)
    img_h = img.shape[0]
    img_w = img.shape[1]
    resdir = '/host_data/models/tiny_yolov2/results/with_aip_supported_dlc'
    snpe_backends = ['aip', 'cpu', 'dsp', 'gpu']
    #snpe_backends = ['x86']
    dlc_results_unscaled = {}
    dlc_results_rescaled = {}
    for snpe_backend in snpe_backends:
        dlc_res_file = f'{resdir}/out_{snpe_backend}/Result_0/grid.raw'
        unscaled_res = postprocess_dlc_output(dlc_res_file)
        dlc_results_unscaled[snpe_backend] = unscaled_res
        # tiny_yolo2_helpers.print_results(unscaled_res)
        rescaled_res = tiny_yolo2_helpers.rescale_results(unscaled_res, img_h=img_h, img_w=img_w)
        dlc_results_rescaled[snpe_backend] = rescaled_res
        print(f'=== {snpe_backend} backend results: ===')
        tiny_yolo2_helpers.print_results(rescaled_res)
        im = draw_yolo_results_on_image(rescaled_res, img.copy())
        cv2.imwrite(f'{resdir}/{snpe_backend}_results.jpg', im)
    # resdir = '/host_data/models/tiny_yolov2'
    # snpe_backends = ['x86']
    # for snpe_backend in snpe_backends:
    #     dlc_res_file = f'{resdir}/out_{snpe_backend}/Result_0/grid.raw'
    #     unscaled_res = postprocess_dlc_output(dlc_res_file)
    #     dlc_results_unscaled[snpe_backend] = unscaled_res
    #     # tiny_yolo2_helpers.print_results(unscaled_res)
    #     rescaled_res = tiny_yolo2_helpers.rescale_results(unscaled_res, img_h=img_h, img_w=img_w)
    #     dlc_results_rescaled[snpe_backend] = rescaled_res
    #     print(f'=== {snpe_backend} backend results: ===')
    #     tiny_yolo2_helpers.print_results(rescaled_res)
    #     im = draw_yolo_results_on_image(rescaled_res, img.copy())
    #     cv2.imwrite(f'{resdir}/{snpe_backend}_results.jpg', im)


