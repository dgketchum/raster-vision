import json
import os

import rasterio
from rasterio.dtypes import int16
from numpy import zeros
# from pycocotools.coco import COCO

import rastervision as rv

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
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier'
]

MODEL_URI = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',


class InstanceSegmentationExperiments(rv.ExperimentSet):
    def exp_main(self):

        train_scene_info = get_scene_info('train')
        val_scene_info = get_scene_info('val')

        exp_id = 'coco-inseg'
        classes = COCO_INSTANCE_CATEGORY_NAMES

        debug = True
        num_epochs = 100
        batch_size = 5

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

        train_scenes = [make_scene(x, y, task) for x, y in train_scene_info]
        val_scenes = [make_scene(x, y, task) for x, y in val_scene_info]

        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
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

    coco = COCO('/home/dgketchum/Downloads/annotations/instances_val2017.json')

    images = []

    for k, v in meta.items():
        id_ = v['id']
        h, w = v['height'], v['width']

        img_ann = [a for a in coco.anns.values() if a['image_id'] == id_]
        label_map = zeros((len(img_ann), h, w), dtype=int16)

        for i, a in enumerate(img_ann, start=0):
            label_mask = coco.annToMask(img_ann[i]) == 1
            new_label = img_ann[i]['category_id']
            label_map[i, label_mask] = new_label

        fig_name = os.path.join(labels_train, '{}'.format(v['file_name'].replace('.jpg', '.tif')))
        meta = {'driver': 'GTiff',
                'height': h,
                'width': w,
                'count': len(img_ann),
                'dtype': int16}
        with rasterio.open(fig_name, 'w', **meta) as out:
            out.write(label_map)

        images.append((v['uri'], fig_name))

    return images


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

    # cmd = '/home/dgketchum/field_extraction/training_data/chip/coco-inseg/command-config-0.json'
    # rv.runner.CommandRunner.run(cmd)

    cmd = '/home/dgketchum/field_extraction/training_data/train/coco-inseg/command-config-0.json'
    rv.runner.CommandRunner.run(cmd)

# ====================================== EOF =================================================================
