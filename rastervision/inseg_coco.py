import json
import os

import rasterio
from rasterio.dtypes import int16
from numpy import zeros
from PIL.ImageColor import colormap
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

COCO_INSTANCE_CATEGORY_NAMES = [str(x) for x in range(91)]
colors = [k for k in colormap.keys()]

MODEL_URI = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',


def get_images(t):
    l_dir_ = os.path.join(PROCESSED_URI, 'label_{}'.format(t))
    l_l = [os.path.join(l_dir_, x) for x in os.listdir(l_dir_)]
    i_l = [x.replace('label_{}'.format(t), 'image_{}'.format(t)) for x in l_l]
    i_l = [x.replace('.tif', '.jpg') for x in i_l]
    return list(zip(i_l, l_l))


class InstanceSegmentationExperiments(rv.ExperimentSet):
    def exp_main(self):

        try:
            train_scene_info = get_images('train')
            val_scene_info = get_images('val')
            test_scene_info = get_images('test')

        except:
            train_scene_info = get_scene_info('train')
            val_scene_info = get_scene_info('val')
            test_scene_info = get_scene_info('test')

        exp_id = 'coco-inseg'
        classes = {k: (v, colors[v]) for (v, k) in enumerate(COCO_INSTANCE_CATEGORY_NAMES)}

        debug = True
        num_epochs = 1
        batch_size = 5

        task = rv.TaskConfig.builder(rv.INSTANCE_SEGMENTATION) \
            .with_chip_size(300) \
            .with_chip_options(chips_per_scene=50,
                               window_method='sliding',
                               debug_chip_probability=1.0) \
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

        train_scenes = [make_scene(raster=x, label=y, task=task, mode='train') for x, y in train_scene_info]
        val_scenes = [make_scene(raster=x, label=y, task=task, mode='val') for x, y in val_scene_info]
        test_scenes = [make_scene(raster=x, label=y, task=task, mode='test') for x, y in test_scene_info]

        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(val_scenes) \
            .with_test_scenes(test_scenes) \
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


def make_scene(raster, label, task, mode):

    _id = os.path.splitext(os.path.basename(raster))[0]

    label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
        .with_uri(label) \
        .build()

    label_source = rv.LabelSourceConfig.builder(rv.INSTANCE_SEGMENTATION) \
        .with_raster_source(label_raster_source) \
        .build()

    uri = label.replace('label_{}'.format(mode), 'pred')
    label_store = rv.LabelStoreConfig.builder(rv.INSTANCE_SEGMENTATION_RASTER) \
        .with_uri(uri) \
        .with_rgb(True) \
        .build()

    return rv.SceneConfig.builder() \
        .with_task(task) \
        .with_id(_id) \
        .with_raster_source(raster, channel_order=[0, 1, 2]) \
        .with_label_store(label_store) \
        .with_label_source(label_source) \
        .build()


def get_scene_info(type_='train'):

    print(type_)

    img_dir = os.path.join(PROCESSED_URI, 'image_{}'.format(type_))
    label_dir = os.path.join(PROCESSED_URI, 'label_{}'.format(type_))
    pred_dir = os.path.join(PROCESSED_URI, 'pred_{}'.format(type_))

    meta = collate_annotations(img_dir)

    coco = COCO('/home/dgketchum/Downloads/annotations/instances_val2017.json')

    images = []

    for k, v in meta.items():
        id_ = v['id']
        h, w = v['height'], v['width']

        img_ann = [a for a in coco.anns.values() if a['image_id'] == id_]

        if len(img_ann) < 1:
            print(v['file_name'], 'has no matching images')

        else:
            label_map = zeros((len(img_ann), h, w), dtype=int16)

            for i, a in enumerate(img_ann, start=0):
                label_mask = coco.annToMask(img_ann[i]) == 1
                new_label = img_ann[i]['category_id']
                label_map[i, label_mask] = new_label

            name = v['file_name'].replace('.jpg', '.tif')
            label_name = os.path.join(label_dir, name)

            meta = {'driver': 'GTiff',
                    'height': h,
                    'width': w,
                    'count': len(img_ann),
                    'dtype': int16}

            with rasterio.open(label_name, 'w', **meta) as out:
                out.write(label_map)

            pred = os.path.join(pred_dir, name)

            images.append((v['uri'], label_name, pred))

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

    return meta


if __name__ == '__main__':
    # i = InstanceSegmentationExperiments().exp_main()
    # rv.cli.main.run(['local', '--tempdir', '{}'.format(TMP)])

    # cmd = '/home/dgketchum/field_extraction/training_data/train/coco-inseg/command-config-0.json'
    # rv.runner.CommandRunner.run(cmd)

    cmd = '/home/dgketchum/field_extraction/training_data/predict/coco-inseg/command-config-0.json'
    rv.runner.CommandRunner.run(cmd)

# ====================================== EOF =================================================================
