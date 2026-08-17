# ReCaption 🖼️🤖

ReCaption is an AI-powered image captioning application that uses
Salesforce BLIP through Hugging Face Transformers to generate natural
language descriptions of images.

The application provides both single-image captioning and batch image
captioning with CSV output through a Gradio web interface.

## Features

- AI-powered image caption generation
- Single image captioning
- Batch captioning for multiple images
- CSV export for batch results
- Gradio web interface
- Conditional prompting
- Local inference using PyTorch
- Support for JPG, JPEG, PNG, and WebP images

## Architecture


                    ReCaption
                        │
                 ┌──────▼──────┐
                 │   Gradio    │
                 │     UI      │
                 └──────┬──────┘
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
       Single Image          Batch Images
              │                   │
              ▼                   ▼
        captioner.py           batch.py
              │                   │
              └─────────┬─────────┘
                        ▼
                 BLIP Model
                        │
                        ▼
                Generated Caption


## TECH STACK

Python
PyTorch
Hugging Face Transformers
Salesforce BLIP
Pillow
Gradio
Project Structure
ReCaption/
├── app.py
├── captioner.py
├── batch.py
├── images/
├── outputs/
├── README.md
├── requirements.txt
├── .gitignore
└── venv/

## HOW IT WORKS

The user uploads an image through the Gradio interface.
Pillow converts the image into a suitable RGB format.
The BLIP processor prepares the image and text prompt.
The pretrained BLIP model performs inference.
Generated tokens are decoded into natural-language text.
For batch processing, captions are written to a CSV file.

## INSTALLATION

Clone the repository:

git clone https://github.com/Mohan0605/ReCaption.git
cd ReCaption

Create a virtual environment:

python3 -m venv venv

Activate it:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Usage

Start ReCaption:

python app.py

Open the local Gradio URL shown in the terminal.

Single Image

Upload an image and click:

Generate Caption
Batch Captioning

Enter the path to a folder containing images, for example:

images

ReCaption processes the supported images and generates:

outputs/captions.csv

The CSV contains:

filename,caption
Example
Input

An image containing Spider-Man.

Generated Caption
spiderman in a spiderman suit


## What I Learned

This project helped me understand:

Pretrained vision-language models
Image preprocessing
PyTorch tensors
Hugging Face Transformers
BLIP image captioning
Model inference
Token generation and decoding
Conditional generation
Batch processing
CSV data handling
Python project structure
Gradio interfaces
Dependency management
Git and GitHub
Limitations
Caption quality depends on the pretrained BLIP model.
The model may incorrectly identify objects or people.
Generated captions may be generic.
The application currently performs local inference.
CPU inference can be relatively slow on supported hardware.


## Potential future improvements include:

Better captioning models
GPU optimization
Custom caption styles
More advanced visual-language models
Authentication and deployment
Cloud hosting
License

## This project is intended for educational and portfolio purposes.