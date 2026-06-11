import os
# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def build_model(num_classes=6):
    print("Building MobileNetV2 transfer learning model...")
    # Load MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base layers initially
    base_model.trainable = False
    
    # Create the model head
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, base_model

def main():
    data_dir = 'Garbage_Augmented'
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found. Please run prepare_data.py first.")
        return

    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # 1. Collect file paths and labels
    print("Collecting dataset file paths...")
    data = []
    for c in classes:
        class_dir = os.path.join(data_dir, c)
        if not os.path.exists(class_dir):
            continue
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append({
                    'filename': os.path.join(class_dir, f),
                    'class': c
                })
                
    df = pd.DataFrame(data)
    print(f"Total images collected: {len(df)}")
    print("Class distribution:\n", df['class'].value_counts())
    
    # 2. Stratified split (70% Train, 15% Val, 15% Test)
    print("Splitting dataset into Train (70%), Val (15%), and Test (15%)...")
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        stratify=df['class'], 
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['class'], 
        random_state=42
    )
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # 3. Data generators with Augmentation
    print("Setting up Data Generators...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # Batch size
    BATCH_SIZE = 32
    
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_gen = val_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # 4. Build Model
    model, base_model = build_model(num_classes=len(classes))
    model.summary()

    # 5. Compile and train Stage 1 (Warmup classification head)
    print("\n--- STAGE 1: Training Classification Head (Base layers frozen) ---")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    checkpoint = ModelCheckpoint(
        'model.h5', 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min', 
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=8, 
        restore_best_weights=True, 
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=4, 
        min_lr=1e-6, 
        verbose=1
    )

    epochs_stage1 = 6
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_stage1,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # 6. Compile and train Stage 2 (Fine-tuning top layers of base model)
    print("\n--- STAGE 2: Fine-Tuning Top Base Model Layers (Unfreezing layer 100+) ---")
    base_model.trainable = True
    
    # Freeze layers up to 100
    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Very low learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    epochs_stage2 = 8
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_stage2,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # 7. Evaluate on Test Dataset
    print("\n--- EVALUATION: Testing Model Performance ---")
    # Load best saved weights
    model = keras.models.load_model('model.h5')
    
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")

    # Predictions
    test_gen.reset()
    predictions = model.predict(test_gen)
    pred_indices = np.argmax(predictions, axis=1)
    true_indices = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_indices, pred_indices, target_names=class_labels))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_indices, pred_indices)
    print(cm)

    # 8. Plot Accuracy & Loss History
    print("\nGenerating training history plots...")
    # Concatenate history curves
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#3b82f6', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#10b981', linewidth=2)
    plt.axvline(x=len(history1.history['accuracy']), color='gray', linestyle='--', label='Fine-tuning Started')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#ef4444', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#f59e0b', linewidth=2)
    plt.axvline(x=len(history1.history['accuracy']), color='gray', linestyle='--', label='Fine-tuning Started')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Saved training history plot as 'training_history.png'")
    plt.close()
    
    print("\nTraining workflow fully completed and best model saved to 'model.h5'.")

if __name__ == '__main__':
    main()
