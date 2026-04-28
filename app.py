import numpy as np
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import keras
from keras import layers
import gradio as gr
from PIL import Image

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

model = keras.models.load_model("results/baseline_cnn_best.keras")

last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, layers.Conv2D):
        last_conv_layer = layer
        break


def make_gradcam_heatmap(img_array, pred_index):
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
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


def predict(image, threshold):
    img = Image.fromarray(image).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_expanded, verbose=0)[0]

    # Build result table with threshold-based classification
    rows = []
    for i, label in enumerate(LABELS):
        score = float(preds[i])
        detected = "Yes" if score >= threshold else "No"
        if score >= 0.6:
            indicator = "🔴"
        elif score >= threshold:
            indicator = "🟡"
        else:
            indicator = "🟢"
        rows.append([indicator, label, f"{score:.1%}", detected])

    result_md = "| | Condition | Score | Detected |\n|---|---|---|---|\n"
    result_md += "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows)

    top_idx = int(np.argmax(preds))
    heatmap = make_gradcam_heatmap(img_expanded, top_idx)
    gradcam_img = superimpose_heatmap(img_array, heatmap)

    top_label = LABELS[top_idx]
    top_score = float(preds[top_idx])

    return result_md, gradcam_img, f"**Top prediction:** {top_label} ({top_score:.1%})"


with gr.Blocks(
    title="CheXpert Disease Detection",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
) as demo:
    gr.Markdown(
        "# Chest X-Ray Disease Detection\n"
        "Upload a frontal chest X-ray to receive predictions for 5 conditions. "
        "The Grad-CAM heatmap highlights the regions that most influenced the prediction."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload chest X-ray", type="numpy")
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label="Detection threshold",
                info="Lower = more sensitive, higher = more specific"
            )
            submit_btn = gr.Button("Analyse", variant="primary", size="lg")

        with gr.Column(scale=1):
            top_pred_output = gr.Markdown()
            result_table = gr.Markdown(label="Results")
            gradcam_output = gr.Image(label="Grad-CAM heatmap (top condition)")

    gr.Examples(
        examples=[
            ["examples/edema.jpg", 0.5],
            ["examples/pleural_effusion.jpg", 0.5],
            ["examples/cardiomegaly.jpg", 0.5],
        ],
        inputs=[image_input, threshold_slider],
        label="Example X-rays (click to load)"
    )

    submit_btn.click(
        fn=predict,
        inputs=[image_input, threshold_slider],
        outputs=[result_table, gradcam_output, top_pred_output]
    )

    gr.Markdown(
        "---\n"
        "**Model:** Baseline CNN trained on CheXpert (mean AUROC: 0.865) | "
        "**Conditions:** Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion  \n"
        "⚠️ This is a research prototype and not a clinical diagnostic tool."
    )

if __name__ == "__main__":
    demo.launch()
