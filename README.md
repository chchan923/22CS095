# Final Year Project

This repo contains my final year project: Deep-Learning Model For Image Enhancement

# Requirement
It is recommended that: \
Python >= 3.9
Tensorflow >= 2.10

## Run
To test the model:

```
python train.py --mode=predict \
    --predict_images_input_path='./Predict/Input' \
    --predict_images_output_path='./Predict/Output'
    
```

To train the model:

You need to have a dataset before training the model, and you can download the <a href="https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view">LOL-v2 dataset here</a> from Yang et al.

After downloading the dataset, put the Real_captured dataset and Synthetic dataset together and run:
```
python train.py --mode=train \
    --lowlight_train_images_path='./Dataset/Train/Low' \
    --result_train_images_path='./Dataset/Train/Normal' \
    --lowlight_test_images_path='./Dataset/Test/Low' \
    --result_test_images_path='./Dataset/Test/Normal' \
```
You can also adjust the batch size and epochs by command, please refer to the program for infos.

To check the model result with metrics:
```
python testresult.py \
    --check_images_path="./Predict/Output/" \
    --ground_truth_images_path="./Dataset/Test/Normal"
```


