import segmentation_models as sm
import os
from datastructure.dataloader import Dataloder
from datastructure.dataset import Dataset
from utils.image_utils import visualize, denormalize, get_training_augmentation, get_preprocessing, \
    get_validation_augmentation
import keras
import numpy as np

SKIN_MODEL_V1 = './skin_segmentation_model_v1.h5'
ZWI_MODEL_PATH = './best_model.h5'
DATA_DIR = './dataset/'

BACKBONE = 'efficientnetb3'
BATCH_SIZE = 8
CLASSES = ['skin']
LR = 0.0001
EPOCHS = 20

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'


def train_model():

    os.environ["SM_FRAMEWORK"] = "tf.keras"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    preprocess_input = sm.get_preprocessing(BACKBONE)

    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # Dataset for train images
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint(ZWI_MODEL_PATH, save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    # train model
    history = model.fit_generator(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )


def test_model(model_path=ZWI_MODEL_PATH):

    print('viewing model', model_path)

    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    preprocess_input = sm.get_preprocessing(BACKBONE)

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        classes=CLASSES,
        preprocessing=get_preprocessing(preprocess_input),
    )

    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    optim = keras.optimizers.Adam(LR)

    model.compile(optim, total_loss, metrics)

    model.load_weights(model_path)

    scores = model.evaluate_generator(test_dataloader)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))

    n = 5
    ids = np.random.choice(np.arange(len(test_dataset)), size=n)

    for i in ids:
        image, gt_mask = test_dataset[i]
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image).round()

        visualize(
            image=denormalize(image.squeeze()),
            gt_mask=gt_mask[..., 0].squeeze(),
            pr_mask=pr_mask[..., 0].squeeze(),
        )


if __name__ == "__main__":
    train_model()
    test_model(SKIN_MODEL_V1)  # Change here to view other models
