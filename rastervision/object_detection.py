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


class CowcObjectDetectionExperiments(rv.ExperimentSet):
    def exp_main(self, test=True):

        exp_id = 'washington-object-detection-test'
        train_scene_info = get_scene_info('train')
        val_scene_info = get_scene_info('val')

        debug = True
        classes = {'field': (1, 'green'), 'background': (0, 'white')}

        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
            .with_classes(classes) \
            .with_chip_options(neg_ratio=5.0, ioa_thresh=0.9).with_predict_options(merge_thresh=0.5,
                                                                                   score_thresh=0.9).build()

        batch_size = 16
        num_epochs = 10
        if test:
            batch_size = 2
            num_epochs = 2

        backend = rv.BackendConfig.builder(rv.PYTORCH_OBJECT_DETECTION).with_task(task).with_train_options(
            lr=1e-4,
            one_cycle=True,
            batch_size=batch_size,
            num_epochs=num_epochs,
            model_arch='resnet18',
            debug=debug).build()

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
    img_dir_train = os.path.join(PROCESSED_URI, 'image_{}'.format(_type))
    img_uris_train = [os.path.join(img_dir_train, x) for x in os.listdir(img_dir_train) if x.endswith('.tif')]
    labels_train = img_dir_train.replace('image_{}'.format(_type), 'label_{}'.format(_type))
    labels_uris_train = [os.path.join(labels_train, x) for x in os.listdir(img_dir_train) if x.endswith('.tif')]
    return [(x, y) for x, y in zip(img_uris_train, labels_uris_train)]


if __name__ == '__main__':
    i = CowcObjectDetectionExperiments().exp_main(test=True)
    rv.cli.main.run(['local', '--tempdir', '{}'.format(TMP)])
    # rv.main()

    # cmd = '/home/dgketchum/field_extraction/WA/train/washington-object-detection-test/command-config-0.json'
    # rv.runner.CommandRunner.run(cmd)

# ========================= EOF ====================================================================
