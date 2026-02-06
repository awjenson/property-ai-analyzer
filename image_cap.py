import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import os

# 1. Setup device (Accelerates performance on your MacBook Pro)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# 2. Check if image exists
img_path = "image.png"
if not os.path.exists(img_path):
    print(f"Error: Could not find {img_path} in the current directory.")
else:
    # convert it into an RGB format 
    image = Image.open(img_path).convert('RGB')

    # 3. Prepare inputs and move them to the same device as the model
    text = "a photography of"
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    # Generate a caption for the image
    outputs = model.generate(**inputs, max_length=50)

    # Decode the generated tokens to text
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    print("-" * 20)
    print(f"Generated Caption: {caption}")
    print("-" * 20)