import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Modeli yükle
model = tf.keras.models.load_model("pet_classifier_model.h5")

# Sınıf adları (Klasörlerden otomatik bulabiliriz)
class_names = sorted(os.listdir("pet_data"))

# Tahmin fonksiyonu
def predict(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    preds = model.predict(img)[0]
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))
    return f"{pred_class} ({confidence:.2%} güven)"

# Arayüz
demo = gr.Interface(fn=predict, 
                    inputs=gr.Image(type="pil"), 
                    outputs="text",
                    title="🐶 Kedi/Köpek Irk Tahmini")

# Yayına aç
demo.launch()