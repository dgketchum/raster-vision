import json
import os

import matplotlib.pyplot as plt
from PIL import Image
from descartes import PolygonPatch
from matplotlib import collections as cplt
from shapely.geometry import Polygon
import rastervision as rv
from pycocotools import mask
from pycocotools.coco import COCO

if 'home' in os.getcwd():
    home = os.path.expanduser('~')
    ROOT_URI = os.path.join(home, 'field_extraction', 'training_data')
    PROCESSED_URI = os.path.join(ROOT_URI, 'example', 'COCO')
    TMP = os.environ['TMPDIR'] = os.path.join(ROOT_URI, 'tmp')
    os.environ['TORCH_HOME'] = os.path.join(home, 'field_extraction', 'torch-cache')
    os.environ['GDAL_DATA'] = os.path.join(home,
                                           'miniconda2/envs/vision/lib/python3.7/site-packages/rasterio/gdal_data')
else:
    ROOT_URI = '/opt/data/training_data'
    PROCESSED_URI = os.path.join(ROOT_URI, 'example', 'COCO')

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

MODEL_URI = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',


class InstanceSegmentationExperiments(rv.ExperimentSet):
    def exp_main(self):
        # train_scene_info = get_scene_info('train')
        val_scene_info = get_scene_info('val')

        exp_id = 'coco-inseg'
        classes = COCO_INSTANCE_CATEGORY_NAMES

        debug = True
        num_epochs = 100
        batch_size = 8

        task = rv.TaskConfig.builder(rv.INSTANCE_SEGMENTATION) \
            .with_chip_size(300) \
            .with_chip_options(chips_per_scene=50) \
            .with_classes(classes) \
            .build()

        backend = rv.BackendConfig.builder(rv.PYTORCH_INSTANCE_SEGMENTATION) \
            .with_task(task) \
            .with_pretrained_uri(MODEL_URI)\
            .with_train_options(
            batch_size=batch_size,
            lr=1e-4,
            num_epochs=num_epochs,
            model_arch='resnet50',
            debug=debug) \
            .build()

        # train_scenes = [make_scene(x, y, task) for x, y in train_scene_info]
        val_scenes = [make_scene(x, y, task) for x, y in val_scene_info[:100]]

        dataset = rv.DatasetConfig.builder() \
            .with_validation_scenes(val_scenes) \
            .build()

        experiment = rv.ExperimentConfig.builder() \
            .with_id(exp_id) \
            .with_root_uri(ROOT_URI) \
            .with_task(task) \
            .with_backend(backend) \
            .with_dataset(dataset) \
            .with_stats_analyzer() \
            .build()

        return experiment


def make_scene(raster_uri, label_uri, task):
    _id = os.path.splitext(os.path.basename(raster_uri))[0]

    label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
        .with_uri(label_uri) \
        .build()

    label_source = rv.LabelSourceConfig.builder(rv.INSTANCE_SEGMENTATION) \
        .with_raster_source(label_raster_source) \
        .build()

    return rv.SceneConfig.builder() \
        .with_task(task) \
        .with_id(_id) \
        .with_raster_source(raster_uri, channel_order=[0, 1, 2]) \
        .with_label_source(label_source) \
        .build()


def get_scene_info(_type='train'):

    f = open('/home/dgketchum/Downloads/annotations/instances_val2017_collated.json', 'r')
    meta = json.load(f)
    f.close()

    img_dir_train = os.path.join(PROCESSED_URI, 'image_{}'.format(_type))
    img_uris_train = [os.path.join(img_dir_train, x) for x in os.listdir(img_dir_train) if x.endswith('.jpg')]

    labels_train = os.path.join(PROCESSED_URI, 'label_{}'.format(_type))

    for k, v in meta.items():

        ann = v['annotations']
        seg = ann['segmentation']
        h, w = v['height'], v['width']
        b = ann['bbox']

        x = [[x for x in s[1::2]] for s in seg]
        y = [[y for y in s[::2]] for s in seg]
        vectors = [Polygon(zip(x, y)) for x, y in zip(x, y)]

        # x, y, bh, bw = b[0], h - b[1], b[2], b[3]
        # b = Box(y - bh, x, y, x + bw)
        # vectors.append(b.to_shapely())

        im = Image.open(v['uri'], 'r')
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        fig, ax = plt.subplots()
        plt.imshow(im, cmap='viridis')
        patches = [PolygonPatch(feature, edgecolor="red", facecolor="none",
                                linewidth=1.) for feature in vectors]
        ax.add_collection(cplt.PatchCollection(patches, match_original=True))
        ax.set_xlim(0, im.size[0])
        ax.set_ylim(0, im.size[1])
        fig_name = os.path.join(labels_train, '{}'.format(v['file_name']))
        plt.savefig(fig_name)
        plt.close(fig)

    labels_uris_train = [os.path.join(labels_train, x) for x in os.listdir(img_dir_train) if x.endswith('.jpg')]
    return [(x, y) for x, y in zip(img_uris_train, labels_uris_train)]


def collate_annotations(img_dir_train):

    f = open('/home/dgketchum/Downloads/annotations/instances_val2017.json', 'r')
    j = json.load(f)
    f.close()

    meta = {}
    for x in j['images']:
        try:
            x['uri'] = os.path.join(img_dir_train, x['file_name'])
            if os.path.exists(x['uri']):
                for a in j['annotations']:
                    if a['id'] == x['id']:
                        x['annotations'] = a
                        meta[a['id']] = x
                        break
        except ValueError:
            pass

    with open('/home/dgketchum/Downloads/annotations/instances_val2017_collated.json', 'w') as fp:
        json.dump(meta, fp)


if __name__ == '__main__':
    # i = InstanceSegmentationExperiments().exp_main()
    # rv.cli.main.run(['local', '--tempdir', '{}'.format(TMP)])
    # rv.main()
    # collate_annotations(os.path.join(PROCESSED_URI, 'image_val'))
    get_scene_info(_type='val')
# ====================================== EOF =================================================================
