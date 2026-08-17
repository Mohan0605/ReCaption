import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def generate_caption(image):
    """Generate a caption for an uploaded image."""

    if image is None:
        return "Please upload an image."

    try:
        image = Image.fromarray(image).convert("RGB")

        prompt = "a photo of"

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )

        output = model.generate(**inputs)

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption

    except Exception as e:
        return f"Error generating caption: {e}"


# Create the Gradio interface
interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(),
    outputs=gr.Textbox(label="Generated Caption"),
    title="ReCaption 🖼️🤖",
    description="Upload an image and let AI generate a caption."
)


# Launch the application
interface.launch()