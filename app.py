import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import tensorflow as tf
import keras
from keras import layers
import gradio as gr
from PIL import Image

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

models = {
    "Baseline CNN (AUC 0.865)": keras.models.load_model("results/baseline_cnn_best.keras"),
    "DenseNet121 (AUC 0.857)": keras.models.load_model("results/densenet_best.keras"),
}


def get_last_conv_layer(model):
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            last_conv = layer
        elif hasattr(layer, 'layers'):
            inner = get_last_conv_layer(layer)
            if inner:
                last_conv = inner
    return last_conv


def make_gradcam_heatmap(img_array, model, conv_layer, pred_index):
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
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


def predict(image, model_name):
    model = models[model_name]

    img = Image.fromarray(image).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_expanded, verbose=0)[0]
    scores = {label: float(preds[i]) for i, label in enumerate(LABELS)}

    top_idx = int(np.argmax(preds))
    try:
        conv_layer = get_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(img_expanded, model, conv_layer, top_idx)
        gradcam_img = superimpose_heatmap(img_array, heatmap)
    except Exception:
        gradcam_img = Image.fromarray(np.uint8(img_array * 255))

    return scores, gradcam_img


with gr.Blocks(title="CheXpert Disease Detection") as demo:
    gr.Markdown(
        "# Chest X-Ray Disease Detection\n"
        "Upload a frontal chest X-ray to get predictions for 5 conditions. "
        "The Grad-CAM heatmap shows which regions of the image influenced the top prediction."
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload chest X-ray", type="numpy")
            model_selector = gr.Radio(
                choices=list(models.keys()),
                value="Baseline CNN (AUC 0.865)",
                label="Select model"
            )
            submit_btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            label_output = gr.Label(label="Prediction scores", num_top_classes=5)
            gradcam_output = gr.Image(label="Grad-CAM (top condition)")

    gr.Examples(
        examples=[
            ["examples/edema.jpg", "Baseline CNN (AUC 0.865)"],
            ["examples/pleural_effusion.jpg", "Baseline CNN (AUC 0.865)"],
            ["examples/cardiomegaly.jpg", "Baseline CNN (AUC 0.865)"],
        ],
        inputs=[image_input, model_selector],
        label="Example X-rays (click to load)"
    )

    submit_btn.click(fn=predict, inputs=[image_input, model_selector], outputs=[label_output, gradcam_output])

    gr.Markdown(
        "**Conditions:** Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion  \n"
        "**Note:** This is a research prototype and not a clinical diagnostic tool."
    )

if __name__ == "__main__":
    demo.launch()
