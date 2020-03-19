import os

import rastervision as rv

if 'home' in os.getcwd():
    home = os.path.expanduser('~')
    ROOT_URI = os.path.join(home, 'field_extraction', 'WA')
    PROCESSED_URI = os.path.join(ROOT_URI, 'data')
    TMP = os.environ['TMPDIR'] = os.path.join(ROOT_URI, 'tmp')
    os.environ['TORCH_HOME'] = os.path.join(home, 'field_extraction', 'torch-cache')
    os.environ['GDAL_DATA'] = os.path.join(home, 'miniconda2/envs/vision/lib/python3.7',
                                           'site-packages/rasterio/gdal_data')
else:
    ROOT_URI = '/opt/data/training_data'
    PROCESSED_URI = os.path.join(ROOT_URI, 'example')


class InstanceSegmentationExperiments(rv.ExperimentSet):
    def exp_main(self):
        train_scene_info = get_scene_info('train')
        val_scene_info = get_scene_info('val')

        exp_id = 'washington-inseg'
        classes = {'field': (1, 'red'), 'background': (2, 'green')}

        debug = True
        num_epochs = 40
        batch_size = 2

        task = rv.TaskConfig.builder(rv.INSTANCE_SEGMENTATION) \
            .with_chip_size(300) \
            .with_chip_options(window_method='sliding',
                               target_classes=[1],
                               debug_chip_probability=0.25,
                               negative_survival_probability=1.0,
                               stride=100) \
            .with_classes(classes) \
            .build()

        backend = rv.BackendConfig.builder(rv.PYTORCH_INSTANCE_SEGMENTATION) \
            .with_task(task) \
            .with_train_options(
            batch_size=batch_size,
            lr=0.0025,
            num_epochs=num_epochs,
            model_arch='resnet50',
            debug=debug) \
            .build()

        train_scenes = [make_scene(raster=x, label=y, task=task, mode='train') for x, y in train_scene_info]
        val_scenes = [make_scene(raster=x, label=y, task=task, mode='val') for x, y in val_scene_info]

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


def get_scene_info(_type='train'):
    img_dir_train = os.path.join(PROCESSED_URI, 'image_{}'.format(_type))
    img_uris_train = [os.path.join(img_dir_train, x) for x in os.listdir(img_dir_train) if x.endswith('.tif')]
    labels_train = img_dir_train.replace('image_{}'.format(_type), 'label_{}'.format(_type))
    labels_uris_train = [os.path.join(labels_train, x) for x in os.listdir(img_dir_train) if x.endswith('.tif')]
    return [(x, y) for x, y in zip(img_uris_train, labels_uris_train)]


if __name__ == '__main__':
    # i = InstanceSegmentationExperiments().exp_main()
    # rv.cli.main.run(['local', '--tempdir', '{}'.format(TMP)])
    # rv.main()

    cmd = '/home/dgketchum/field_extraction/WA/train/washington-inseg/command-config-0.json'
    rv.runner.CommandRunner.run(cmd)

# ====================================== EOF =================================================================

