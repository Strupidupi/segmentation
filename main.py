

def main():
    import os
    from datastructure.dataloader import Dataloder
    from datastructure.dataset import Dataset
    from utils.image_utils import visualize, denormalize, get_training_augmentation, get_preprocessing, \
        get_validation_augmentation
    import matplotlib.pyplot as plt
    import cv2
    import keras
    import numpy as np

    os.environ["SM_FRAMEWORK"] = "tf.keras"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    DATA_DIR = './data/CamVid/'

    # load repo with data if it is not exists
    if not os.path.exists(DATA_DIR):
        print('Loading data...')
        os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
        print('Done!')

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')


    # dataset = Dataset(x_train_dir, y_train_dir, classes=['car', 'pedestrian'])
    #
    # image, mask = dataset[5] # get some sample
    # visualize(
    #     image=image,
    #     cars_mask=mask[..., 0].squeeze(),
    #     sky_mask=mask[..., 1].squeeze(),
    #     background_mask=mask[..., 2].squeeze(),
    # )

    # Lets look at augmented data we have
    # dataset = Dataset(x_train_dir, y_train_dir, classes=['car', 'sky'], augmentation=get_training_augmentation())
    #
    # image, mask = dataset[12] # get some sample
    # visualize(
    #     image=image,
    #     cars_mask=mask[..., 0].squeeze(),
    #     sky_mask=mask[..., 1].squeeze(),
    #     background_mask=mask[..., 2].squeeze(),
    # )

    import segmentation_models as sm

    BACKBONE = 'efficientnetb3'
    BATCH_SIZE = 8
    CLASSES = ['car']
    LR = 0.0001
    EPOCHS = 2

    preprocess_input = sm.get_preprocessing(BACKBONE)

    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

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

    # check shapes for errors
    assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
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

    # Plot training & validation iou_score values
    # plt.figure(figsize=(30, 5))
    # plt.subplot(121)
    # plt.plot(history.history['iou_score'])
    # plt.plot(history.history['val_iou_score'])
    # plt.title('Model iou_score')
    # plt.ylabel('iou_score')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    # plt.subplot(122)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()


    ### Model evaluation

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)


    # load best weights
    model.load_weights('best_model.h5')


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
    main()