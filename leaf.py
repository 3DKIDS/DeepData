import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageOps
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import shutil
import random
import cv2
import matplotlib.pyplot as plt

# GPU 설정 확인
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

# Load data
data_path = '/kaggle/input/plant-pathology-2020-fgvc7/'

train_df = pd.read_csv(data_path + 'train.csv')
test_df = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')

# 데이터를 타깃값별로 추출
healthy = train_df.loc[train_df['healthy'] == 1]
multiple_diseases = train_df.loc[train_df['multiple_diseases'] == 1]
rust = train_df.loc[train_df['rust'] == 1]
scab = train_df.loc[train_df['scab'] == 1]

plt.figure(figsize=(7, 7))

label = ['healthy', 'multiple diseases', 'rust', 'scab']
plt.pie([len(healthy), len(multiple_diseases), len(rust), len(scab)],
        labels=label,
        autopct='%.1f%%')

# 이미 구성 경로
def construct_image_path(image_id, folder_path):
    image_path = os.path.join(folder_path, image_id + ".jpg")
    return image_path

# 이미지 저장 폴더
images_folder_path = "/kaggle/input/plant-pathology-2020-fgvc7/images"

train_df['image_path'] = train_df['image_id'].apply(lambda x: construct_image_path(x, images_folder_path))

def copy_images_by_column(train_df, column_name):
    filtered_df = train_df[train_df[column_name] == 1]
    output_folder = os.path.join("output", column_name)
    os.makedirs(output_folder, exist_ok=True)
    for _, row in filtered_df.iterrows():
        image_path = row['image_path']
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, image_name)
        shutil.copyfile(image_path, output_path)
    print(f"Images for {column_name} copied to {output_folder} folder successfully.")

copy_images_by_column(train_df, 'healthy')
copy_images_by_column(train_df, 'rust')
copy_images_by_column(train_df, 'scab')
copy_images_by_column(train_df, 'multiple_diseases')

# 증강 기능
def augment_images(image_path, target_count):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'. Skipping...")
        return

    filename = os.path.basename(image_path)
    filename_no_ext, ext = os.path.splitext(filename)
    output_dir = os.path.dirname(image_path)

    num_augmented_images = target_count - len(os.listdir(output_dir))

    for i in range(num_augmented_images):
        angle = random.randint(0, 360)
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.path.join(output_dir, f"{filename_no_ext}_rotated_{angle}_{i}{ext}"), rotated)

    print(f"{num_augmented_images} augmented images created for {image_path}")

# 각 폴더에 대해 증강 진행
for class_folder in ['healthy', 'rust', 'scab', 'multiple_diseases']:
    folder_path = os.path.join("/kaggle/working/output", class_folder)
    image_files = os.listdir(folder_path)
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        augment_images(image_path, 400)

print("Augmentation completed.")

# 데이터 폴더 설정
dataset_folder = "/kaggle/working/output"
train_percent = 0.7
test_percent = 0.2
val_percent = 0.1

final_output_folder = "/kaggle/working/final_output"
os.makedirs(final_output_folder, exist_ok=True)

train_folder = os.path.join(final_output_folder, "train")
test_folder = os.path.join(final_output_folder, "test")
val_folder = os.path.join(final_output_folder, "val")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

for class_folder in os.listdir(dataset_folder):
    class_path = os.path.join(dataset_folder, class_folder)
    if os.path.isdir(class_path):
        train_class_path = os.path.join(train_folder, class_folder)
        test_class_path = os.path.join(test_folder, class_folder)
        val_class_path = os.path.join(val_folder, class_folder)

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)

        images = os.listdir(class_path)
        random.shuffle(images)

        num_train = int(len(images) * train_percent)
        num_test = int(len(images) * test_percent)
        num_val = len(images) - num_train - num_test

        for img in images[:num_train]:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_path, img))

        for img in images[num_train:num_train + num_test]:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_path, img))

        for img in images[num_train + num_test:]:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_class_path, img))

# 데이터 생성기 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 경로 재설정
train_folder = '/kaggle/working/final_output/train'
test_folder = '/kaggle/working/final_output/test'
val_folder = '/kaggle/working/final_output/val'

# 폴더가 존재하는지 확인하고, 없으면 생성
for folder in [train_folder, test_folder, val_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

try:
    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_folder,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # 모델 정의 함수
    def create_baseline_model(input_shape):
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # 모델 생성 및 요약 출력
    input_shape = (224, 224, 3)
    baseline_model = create_baseline_model(input_shape)
    baseline_model.summary()

    # 모델 학습
    history = baseline_model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        verbose=1
    )

    # 모델 평가
    test_loss, test_acc = baseline_model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.2f}")

    # 학습 결과 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
