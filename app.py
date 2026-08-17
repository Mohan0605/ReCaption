from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Ask the user for an image path
image_path = input("Enter the path to your image: ")

# Open the image
image = Image.open(image_path)

# Convert the image into model inputs
inputs = processor(images=image, return_tensors="pt")

# Generate a caption
output = model.generate(**inputs)

# Convert the generated tokens into text
caption = processor.decode(output[0], skip_special_tokens=True)

print("\nGenerated Caption:")
print(caption)