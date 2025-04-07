import torch
import torchvision.transforms as transforms
from PIL import Image

# Set the device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Load the model
model = torch.hub.load(repo_or_dir="miccunifi/QualiCLIP", source="github", model="QualiCLIP")
model.eval().to(device)

# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

# Load the image
img_path = "/home/naumov/code/general-pipeline/data/generated_images/stable-diffusion-v1-5-stable-diffusion-v1-5/3/image_0.png"
img = Image.open(img_path).convert("RGB")

# Preprocess the image
img = preprocess(img).unsqueeze(0).to(device)

# Compute the quality score
with torch.no_grad(), torch.cuda.amp.autocast():
    score = model(img)

print(f"Image quality score: {score.item()}")