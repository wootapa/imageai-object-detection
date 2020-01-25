import os
import sys
import glob
import shutil
from imageai.Detection.Custom import DetectionModelTrainer

# This is where our training data is found
DATA_PATH = '/opt/training'
# This is the base-model we're transferlearning from
MODEL_PATH = '/opt/model/model.h5'
# This is the type of objects we want to learn (bird,dog,cat,person)
OBJECT_NAMES = os.environ.get('OBJECT_NAMES', '').split(',')
# How many passes we will do of ALL the images
EXPIREMENTSCOUNT = 5
# How many images will be fed into every pass. Higher requries more memory.
BATCHCOUNT = 2

DOTRAIN = os.environ.get('DOTRAIN', 0) == '1'
DOEVALUATE = os.environ.get('DOEVALUATE', 0) == '1'
DORESUMELAST = os.environ.get('DORESUMELAST', 0) == '1'
DOTRANSFERLEARN = os.environ.get('DOTRANSFERLEARN', 0) == '1'


def main():
    # Will we train?
    if DOTRAIN:
        # Use last trained model as base for transferlearning?
        last_model = '' # Ohmy. This means None.
        if DOTRANSFERLEARN:
            last_model = MODEL_PATH

            if DORESUMELAST:
                last_model = get_last_model()
        else:
            if os.path.exists(os.path.join(DATA_PATH, 'cache')):
                shutil.rmtree(os.path.join(DATA_PATH, 'cache'))
                shutil.rmtree(os.path.join(DATA_PATH, 'json'))
                shutil.rmtree(os.path.join(DATA_PATH, 'logs'))
                shutil.rmtree(os.path.join(DATA_PATH, 'models'))

        trainer = DetectionModelTrainer()
        trainer.setModelTypeAsYOLOv3()
        trainer.setDataDirectory(data_directory=DATA_PATH)

        trainer.setTrainConfig(
            object_names_array=OBJECT_NAMES,
            batch_size=BATCHCOUNT,
            num_experiments=EXPIREMENTSCOUNT,
            train_from_pretrained_model=last_model
        )
        trainer.trainModel()

    # Will we evaluate?
    if DOEVALUATE:
        # Show evaluations for the last model
        metrics = trainer.evaluateModel(
            model_path=get_last_model(),
            json_path=os.path.join(DATA_PATH, 'json', 'detection_config.json'),
            iou_threshold=.5,
            object_threshold=.3,
            nms_threshold=.5
        )
        print(metrics)


def get_last_model():
    models = glob.glob(os.path.join(DATA_PATH, 'models', '*'))
    if models:
        return max(models, key=os.path.getctime)
    return MODEL_PATH


if __name__ == '__main__':
    main()
