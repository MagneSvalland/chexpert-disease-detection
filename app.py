import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import keras
from keras import layers
import gradio as gr
from PIL import Image

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
MODEL_PATH = "results/baseline_cnn_best.keras"

model = keras.models.load_model(MODEL_PATH)

last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, layers.Conv2D):
        last_conv_layer_name = layer.name
        break


def make_gradcam_heatmap(img_array, pred_index):
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def superimpose_heatmap(img_array, heatmap, alpha=0.4):
    heatmap_resized = np.uint8(255 * heatmap)
    jet_colors = matplotlib.colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed = jet_heatmap * alpha + np.uint8(img_array * 255)
    return keras.utils.array_to_img(superimposed)


def predict(image):
    # Preprocess
    img = Image.fromarray(image).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_expanded, verbose=0)[0]
    scores = {label: float(preds[i]) for i, label in enumerate(LABELS)}

    # Grad-CAM for the highest-scoring condition
    top_idx = int(np.argmax(preds))
    heatmap = make_gradcam_heatmap(img_expanded, top_idx)
    gradcam_img = superimpose_heatmap(img_array, heatmap)

    return scores, gradcam_img


with gr.Blocks(title="CheXpert Disease Detection") as demo:
    gr.Markdown(
        "# Chest X-Ray Disease Detection\n"
        "Upload a frontal chest X-ray to get predictions for 5 conditions. "
        "The Grad-CAM heatmap shows which regions influenced the top prediction."
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload chest X-ray", type="numpy")
            submit_btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            label_output = gr.Label(label="Prediction scores", num_top_classes=5)
            gradcam_output = gr.Image(label="Grad-CAM (top condition)")

    submit_btn.click(fn=predict, inputs=image_input, outputs=[label_output, gradcam_output])

    gr.Markdown(
        "**Model:** Baseline CNN trained on CheXpert (mean AUC 0.865)  \n"
        "**Conditions:** Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion  \n"
        "**Note:** This is a research prototype and not a clinical tool."
    )

if __name__ == "__main__":
    demo.launch()
