import gradio as gr

from captioner import generate_caption
from batch import caption_folder


def process_folder(folder):
    """Process a folder and create a CSV of captions."""

    if not folder:
        return None

    try:
        return caption_folder(folder)

    except FileNotFoundError:
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None


with gr.Blocks() as interface:

    gr.Markdown("# ReCaption 🖼️🤖")

    gr.Markdown(
        "Generate AI-powered captions for images."
    )

    gr.Markdown("## Single Image")

    image_input = gr.Image(
        label="Upload an image"
    )

    caption_output = gr.Textbox(
        label="Generated Caption"
    )

    caption_button = gr.Button(
        "Generate Caption"
    )

    caption_button.click(
        fn=generate_caption,
        inputs=image_input,
        outputs=caption_output
    )

    gr.Markdown("## Batch Captioning")

    folder_input = gr.Textbox(
        label="Image folder path",
        placeholder="Example: images"
    )

    batch_button = gr.Button(
        "Generate Batch Captions"
    )

    csv_output = gr.File(
        label="Download captions.csv"
    )

    batch_button.click(
        fn=process_folder,
        inputs=folder_input,
        outputs=csv_output
    )


interface.launch()