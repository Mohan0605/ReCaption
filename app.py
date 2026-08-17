from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def generate_caption(image_path):
    """Generate a caption for an image."""

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print("Error: Image file not found.")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    inputs = processor(images=image, return_tensors="pt")

    output = model.generate(**inputs)

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption


image_path = input("Enter the path to your image: ")

caption = generate_caption(image_path)

if caption:
    print("\nGenerated Caption:")
    print(caption)