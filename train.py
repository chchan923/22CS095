# Import necessary modules
import tensorflow as tf

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Rescaling, concatenate
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2
import os
import argparse

def ssim_loss(y_true, y_pred):
    alpha = 0.4

    loss_ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0, filter_size = 11))
    loss_L1 = tf.losses.mean_absolute_error(y_true, y_pred)
    return loss_ssim * alpha + loss_L1 * (1-alpha)

def cnn_model():
    inputs = Input(shape=(None, None, 1))

    conv1 = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    conv1 = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(pool3)
    conv4 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(conv4)

    up1 = Conv2D(filters=32, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(conv4))
    merge1 = concatenate([conv3, up1], axis = 3)
    conv5 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(merge1)
    conv5 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(conv5)

    up2 = Conv2D(filters=16, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(conv5))
    merge2 = concatenate([conv2, up2], axis = 3)
    conv6 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(merge2)
    conv6 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(conv6)

    up3 = Conv2D(filters=8, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(conv6))
    merge3 = concatenate([conv1, up3], axis = 3)
    conv7 = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(merge3)
    conv7 = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(conv7)

    final = Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='relu')(conv7)

    model = Model(inputs=inputs, outputs=final)
    return model

def train(config):

    # training dataset config
    data_gen_args = dict(
        rescale= 1./255,  # Normalize pixel values
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode= "nearest",
    )

    # create dataset generator
    x_train_datagen = ImageDataGenerator(**data_gen_args)
    x_test_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = 1000
    x_train_generator = x_train_datagen.flow_from_directory(
        config.lowlight_train_images_path,
        batch_size = config.train_batch_size,
        class_mode = None,
        target_size= (384,384),
        color_mode='grayscale',
        seed = seed
    )

    x_test_generator = x_test_datagen.flow_from_directory(
        config.result_train_images_path,
        batch_size=config.train_batch_size,
        class_mode = None,
        target_size= (384,384),
        color_mode='grayscale',
        seed = seed
    )

    # validation dataset
    y_train_datagen = ImageDataGenerator(rescale= 1./255)
    y_test_datagen = ImageDataGenerator(rescale= 1./255)

    y_train_generator = y_train_datagen.flow_from_directory(
        config.lowlight_test_images_path,
        batch_size=config.test_batch_size,
        class_mode = None,
        color_mode='grayscale',
        target_size= (384,384),
        seed = seed
    )

    y_test_generator = y_test_datagen.flow_from_directory(
        config.result_test_images_path,
        batch_size=config.test_batch_size,
        class_mode = None,
        color_mode='grayscale',
        target_size= (384,384),
        seed = seed
    )

    # Compile the model
    model = cnn_model()
    # print(model.summary())
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = ssim_loss, 
        metrics = ["mae"]
    )

    train_generator = zip(x_train_generator, x_test_generator)
    validation_generator = zip(y_train_generator, y_test_generator) 

    model.fit(
        train_generator,
        validation_data = validation_generator,
        steps_per_epoch = (x_train_generator.samples / config.train_batch_size),
        epochs = config.epochs,
        batch_size= config.train_batch_size,
        validation_steps = (y_test_generator.samples / config.test_batch_size),
        validation_batch_size= config.test_batch_size,
        verbose = 1
    )

    model.save(config.model_loc)

def predict(config):
    model = load_model(config.model_loc, custom_objects={'ssim_loss': ssim_loss})

    for filename in os.listdir(config.predict_images_input_path):
        image_bgr = cv2.imread(os.path.join(config.predict_images_input_path,filename))

        if image_bgr is not None:

            # prevent maxpool error
            w, h, _ = image_bgr.shape
            offset_w = w % 8 // 2
            offset_h = h % 8 // 2
            end_w = w - (w % 8 - offset_w)
            end_h = h - (h % 8 - offset_h)
            image_bgr = image_bgr[offset_w:end_w, offset_h:end_h, :]

            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCR_CB)
            image = image / 255.0

            Y = np.array([image[:, :, 0]])
            Cr = image[:, :, 1]
            Cb = image[:, :, 2]

            new_Y = model.predict(Y)[0]
            sat = (new_Y[:, :, 0] - Y) / (image[:, :, 0] + 0.04)

            new_Cr = (Cr - 0.5) * sat + 0.5
            new_Cb = (Cb - 0.5) * sat + 0.5

            image[:, :, 0] = new_Y[:, :, 0]
            image[:, :, 1] = new_Cr
            image[:, :, 2] = new_Cb

            image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

            
            result = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
            cv2.imwrite(os.path.join(config.predict_images_output_path, filename), result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Parameters

    # Train + Predict
    parser.add_argument('--model_loc', type=str, default="./Model/")
    parser.add_argument('--mode', type=str, default="predict")

    # Train
    parser.add_argument('--lowlight_train_images_path', type=str, default="./Dataset/Train/Low")
    parser.add_argument('--result_train_images_path', type=str, default="./Dataset/Train/Normal")
    
    parser.add_argument('--lowlight_test_images_path', type=str, default="./Dataset/Test/Low")
    parser.add_argument('--result_test_images_path', type=str, default="./Dataset/Test/Normal")

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=16)

    # Predict
    parser.add_argument('--predict_images_input_path', type=str, default="./Predict/Input")
    parser.add_argument('--predict_images_output_path', type=str, default="./Predict/Output")

    config = parser.parse_known_args()[0]

    if(config.mode == "train"):
        train(config)
    elif(config.mode == "predict"):
        predict(config)