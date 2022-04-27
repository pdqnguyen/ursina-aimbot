## First-person shooter in Ursina engine

## Real-time object detection

Mostly following steps from
[this neptune.ai tutorial](https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api).
Can I just say this tutorial is superbly written, explaining all the subtlies most would neglect to point out??
More tutorials need to do this;
most issues with following programming tutorials arise from confusion over file organization and script parameters:
the little things!

Directory structure, roughly mimicking the above tutorial's suggestions, with explanations for everything:

```
fps/
|__ models/                           (cloned from tensorflow github)
    |__ ...
    |__ research/
        |__ ...
        |__ object_detection/
|__ workspace/                        (this is where all machine learning work goes)
    |__ images/
        |__ raw/                      (screenshots taken from the video game, includes empty images to be ignored)
        |__ labeled/                  (output of image labeling via labelImg: xml files in "Pascal VOC" format)
        |__ split/                    (files from raw AND labeled (not including raw images w/o labels))
            |__ train/                (60% of the files in labeled, for model training)
            |__ validation/           (20% of the files in labeled, for model evaluation during training)
            |__ test/                 (20% of the files in labeled, for model testing after training)
    |__ data/
        |__ label_map.pbtxt           (just a single-class label map, copying the format the COCO dataset label map)
        |__ train.record              (converted from workspace/images/split/train/ using generate_tfrecord.py)
        |__ validation.record         (converted from workspace/images/split/validation/ using generate_tfrecord.py)
        |__ test.record               (converted from workspace/images/split/test/ using generate_tfrecord.py)
    |__ pre_trained_models/           (this is where I download TAR files from the TF model zoo and unzip them)
        |__ <model name>/             (unzipped model directory, e.g. "ssd_mobilenet_v2_320x320_coco17_tpu-8")
            |__ checkpoint/           (point to this when you want to use the pre-trained version)
            |__ saved_model/          (contains the actual model architecture)
            |__ pipeline.config       (copy this to a new directory: this is how you customize your own version)
        |__ ...                       (more model name folders if trying other models)
    |__ models/                       (output model directories go in here)
        |__ <model name>/             (mirrors pre-trained model, e.g. "ssd_mobilenet_v2_320x320_coco17_tpu-8")
            |__ <version>/            (e.g. v1, v2, etc., this is officially the "model_dir" fed to a lot of scripts)
                |__ train/            (TF makes this folder automatically when you train your model)
                |__ eval/             (TF makes this folder automatically when you evaluate your model)
                |__ <checkpoints>     (a bunch of auto-generated checkpoint files get saved here)
                |__ pipeline.config   (copied from pre_trained_models and customized to our liking)
                |__ train_command.txt (I like to save the command used for training so I remember how I ran them)
                |__ eval_command.txt  (same as above, but for validation set; this can be run at the same time)
            |__ ...                   (more version folders if have multiple runs)
        |__ ...                       (more model name folders if trying other models)
    |__ model_main_tf2.py             (provided by TF repository in /models/research/object_detection/)
|__ fps.py                            (the game itself, made using the ursina engine)
|__ my_fpc.py                         (custom version of ursina's FirstPersonController class, for custom crosshair)
|__ record.py                         (screen capture for collecting training data set)
|__ generate_tfrecord.py              (data conversion; copied from sglvladi's TensorFlowObjectDetectionTutorial)
|__ detection.py                      (object detection functions; includes tests on images, webcam, and monitor input)
|__ main.py                           (script that runs the detection model and makes mouse commands accordingly)
```

## Simple bot with pre-trained model

## Screen recording for training data collection

## Label images and create tf record files for training/testing

Used [labelImg](https://github.com/tzutalin/labelImg)
and the `generate_tfrecord.py` script (copied from
[here](https://github.com/sglvladi/TensorFlowObjectDetectionTutorial/blob/master/docs/source/scripts/generate_tfrecord.py))

## Training

Trained the "ssd_mobilenet_v2_320x320_coco17_tpu-8" model with the following command:
```
python .\model_main_tf2.py\
 --pipeline_config_path=E:/Projects/fps/workspace/models/ssd_mobilenet_v2_320x320_coco17_tpu-8/v3/pipeline.config \
  --model_dir=E:/Projects/fps/workspace/models/ssd_mobilenet_v2_320x320_coco17_tpu-8/v3/ \
  --checkpoint_every_n=1000 \
  --n_workers=6 \
  --alsologtostderr
```
Evaluated on validation data:
```
python .\model_main_tf2.py \
  --pipeline_config_path=E:/Projects/fps/workspace/models/ssd_mobilenet_v2_320x320_coco17_tpu-8/v3/pipeline.config \
  --model_dir=E:/Projects/fps/workspace/models/ssd_mobilenet_v2_320x320_coco17_tpu-8/v3/ \
  --checkpoint_dir=E:/Projects/fps/workspace/models/ssd_mobilenet_v2_320x320_coco17_tpu-8/v3/
```
Visualized progress using `tensorboard`:
```
tensorboard --logdir=E:/Projects/fps/workspace/models/ssd_mobilenet_v2_320x320_coco17_tpu-8/v3/
```

## Final aimbot


