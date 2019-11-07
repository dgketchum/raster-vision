import os

import rastervision as rv

if 'home' in os.getcwd():
    home = os.path.expanduser('~')
    ROOT_URI = os.path.join(home, 'field_extraction', 'training_data')
    PROCESSED_URI = os.path.join(ROOT_URI, 'example')
    TMP = os.environ['TMPDIR'] = os.path.join(ROOT_URI, 'tmp')
    os.environ['TORCH_HOME'] = os.path.join(home, 'field_extraction', 'torche-cache')
else:
    ROOT_URI = '/opt/data/training_data'
    PROCESSED_URI = os.path.join(ROOT_URI, 'example')


class InstanceSegmentationExperiments(rv.ExperimentSet):
    def exp_main(self, test=True):

        train_scene_info = get_scene_info('train')
        val_scene_info = get_scene_info('val')

        exp_id = 'washington-inseg'
        classes = {'field': (1, 'green'), 'background': (0, 'white')}

        if test:
            exp_id += '-test'
            debug = True
            num_epochs = 1
            batch_size = 2
            train_scene_info = train_scene_info[0:1]
            val_scene_info = val_scene_info[0:1]
        else:
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
        .with_uri(label_uri)\
        .build()

    label_source = rv.LabelSourceConfig.builder(rv.INSTANCE_SEGMENTATION) \
        .with_raster_source(label_raster_source) \
        .build()

    return rv.SceneConfig.builder() \
        .with_task(task) \
        .with_id(_id) \
        .with_raster_source(raster_uri, channel_order=[0, 1, 2, 3]) \
        .with_label_source(label_source) \
        .build()


def get_scene_info(_type='train'):
    img_dir_train = os.path.join(PROCESSED_URI, 'image_{}'.format(_type))
    img_uris_train = [os.path.join(img_dir_train, x) for x in os.listdir(img_dir_train) if x.endswith('.tif')]
    labels_train = img_dir_train.replace('image_{}'.format(_type), 'label_{}'.format(_type))
    labels_uris_train = [os.path.join(labels_train, x) for x in os.listdir(img_dir_train) if x.endswith('.tif')]
    return [(x, y) for x, y in zip(img_uris_train, labels_uris_train)]


if __name__ == '__main__':
    # i = InstanceSegmentationExperiments().exp_main(test=True)
    # rv.cli.main.run(['local', '--tempdir', '{}'.format(TMP)])
    # rv.main()
    cmd = '/home/dgketchum/field_extraction/training_data/train/washington-inseg-test/command-config-0.json'

    rv.runner.CommandRunner.run(cmd)

# ====================================== EOF =================================================================

