import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'paddy_disease_model.h5')
TFLITE_PATH = os.path.join(BASE_DIR, 'model', 'paddy_disease_model.tflite')

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find model at {MODEL_PATH}")
        return

    print("Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Converting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable optimizations for size and speed
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    print("Saving TFLite model...")
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Success! Model successfully converted and saved to {TFLITE_PATH}")

if __name__ == "__main__":
    main()
