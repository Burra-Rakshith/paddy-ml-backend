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
    
    print("Converting model to TFLite format (Float32 for max compatibility)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # We purposefully AVOID optimizations (like INT8 quantization) 
    # because they force newer opcode versions (like FULLY_CONNECTED v9) 
    # which break on older or specific TensorFlow runtime builds.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Enable standard ops.
        tf.lite.OpsSet.SELECT_TF_OPS    # Fallback to TF ops if needed.
    ]
    
    tflite_model = converter.convert()
    
    print("Saving highly-compatible TFLite model...")
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Success! Model successfully converted and saved to {TFLITE_PATH}")

if __name__ == "__main__":
    main()
