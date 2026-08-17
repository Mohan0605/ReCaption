from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def generate_caption(image):
    """Generate a caption for an image."""

    if image is None:
        return "Please upload an image."

    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

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