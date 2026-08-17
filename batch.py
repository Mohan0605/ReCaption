import csv
import os

from PIL import Image

from captioner import processor, model


SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


def caption_folder(folder_path, output_file="outputs/captions.csv"):
    """Generate captions for all images in a folder."""

    if not os.path.isdir(folder_path):
        raise FileNotFoundError("Image folder not found.")

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    results = []

    for filename in sorted(os.listdir(folder_path)):

        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        image_path = os.path.join(
            folder_path,
            filename
        )

        try:
            image = Image.open(image_path).convert("RGB")

            inputs = processor(
                images=image,
                text="a photo of",
                return_tensors="pt"
            )

            output = model.generate(**inputs)

            caption = processor.decode(
                output[0],
                skip_special_tokens=True
            )

            results.append([
                filename,
                caption
            ])

            print(f"{filename}: {caption}")

        except Exception as e:
            print(
                f"Error processing {filename}: {e}"
            )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "caption"
        ])

        writer.writerows(results)

    return output_file