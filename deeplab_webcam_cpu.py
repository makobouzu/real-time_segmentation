import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
import cv2
import time

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.
    Args:
      image: A PIL.Image object, raw input image.
    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    resized_image = image
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.
  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.
  Args:
    label: A 2D array with integer type, storing the segmentation label.
  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.
  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  
  result = cv2.add(image, seg_image)
  cv2.imshow("camera window", result)

def mask_segmentation(image, seg_map):
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  mask = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
  retval, result = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
  
  return result

def mask_human(image, mask):
  image[mask==255] = [255, 255, 255]
  return image

def rectangle(image, mask):
  contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for i in range(0, len(contours)):
    if len(contours[i]) > 0:
      if cv2.contourArea(contours[i]) < 800:
        continue

      rect = contours[i]
      x, y, w, h = cv2.boundingRect(rect)
      print("num: " + str(i) +  " x: " + str(x) + " y: " + str(y) + " w: " + str(w) + " h: " + str(h))
      cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)

def inpaint_segmentation(image, mask):
  result = cv2.inpaint(image, mask, 10, cv2.INPAINT_TELEA)
  result = cv2.resize(result, dsize=(960, 540))
  cv2.imshow("camera window", result)

#------------------------------------------------------------------------
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

MODEL_NAME = 'mobilenetv2_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'

model_dir = tempfile.mkdtemp()
tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                   download_path)
print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')

capure = cv2.VideoCapture(0)

def run_visualization():
  while(True):
    ret, frame = capure.read()
    original_im = cv2.resize(frame,(480,320))

    start_time = time.time()

    resized_im, seg_map = MODEL.run(original_im)
    mask_img = mask_segmentation(resized_im, seg_map)
    
    output = mask_human(resized_im, mask_img)
    rectangle(output, mask_img)
    cv2.imshow("camera window", output)
        
#    vis_segmentation(resized_im, seg_map)
#    inpaint_segmentation(resized_im, mask_img)

    elapsed_time = time.time() - start_time
#    print(elapsed_time)

    if cv2.waitKey(1) == 27:
      break
  capure.release()
  cv2.destroyAllWindows()

run_visualization()
