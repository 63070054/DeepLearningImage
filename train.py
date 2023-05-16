from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to your dataset directory containing the image folders
dataset_dir = './images'

# Set the number of classes
num_classes = 5

# Set the input image size
input_shape = (128, 128, 3)

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Splitting the data into training and validation sets
)

# Load and preprocess the images from the dataset directory
image_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Use the training subset of the data
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Use the validation subset of the data
)

# Build the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    image_generator,
    steps_per_epoch=image_generator.samples // image_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)