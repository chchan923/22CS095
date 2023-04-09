import tensorflow as tf
from keras.utils import image_dataset_from_directory
import argparse

def test(config):
    check_dataset = image_dataset_from_directory(
        directory= config.check_images_path,
        labels=None,
        label_mode=None,
        batch_size=100,
        image_size=(600, 400))

    true_dataset = image_dataset_from_directory(
        directory= config.ground_truth_images_path,
        labels=None,
        label_mode=None,
        batch_size=100,
        image_size=(600, 400))
    
    psnr_val = 0
    ssim_val = 0
    mae_val = 0

    for check, true in zip(check_dataset, true_dataset):
        for index in range(100):
            psnr_val += tf.reduce_mean(tf.image.psnr(check[index], true[index], 255.0))
            ssim_val += tf.reduce_mean(tf.image.ssim(check[index], true[index], 255.0))
            mae_val +=  tf.reduce_mean(tf.abs((check[index] / 255.0) - (true[index] / 255.0)))

    print("PSNR:", psnr_val / 100.0)
    print("SSIM:", ssim_val / 100.0)
    print("MAE:", mae_val / 100.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_images_path', type=str, default="./MethodResult/MyModel")
    parser.add_argument('--ground_truth_images_path', type=str, default="./MethodResult/GroundTruth")
    config = parser.parse_known_args()[0]
    test(config)

