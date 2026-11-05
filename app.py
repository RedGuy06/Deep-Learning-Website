from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

app = Flask(__name__)
CORS(app)  # Later: CORS(app, origins=["https://your-site.netlify.app"])

# ---------------------------------------------------------------------------
# Load the model ONCE at startup (not on every request)
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "resnet18_realcard_androom_combinedface.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

values = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
ranks = values["ranks"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(ranks))
model.load_state_dict(values["model_state_dict"])
model = model.to(device)
model.eval()

# Transforms to match how the model was trained
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def hand_total(hand):
    total = sum(1 if rank == "A" else int(rank) for rank in hand)  # Ace = 1 for now
    if "A" in hand and total + 10 <= 21:
        total += 10  # Make one ace 11 if it doesn't bust
    return total


@app.get("/")
def home():
    return send_file(Path(__file__).parent / "index.html")


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="No image was uploaded."), 400

    try:
        with Image.open(request.files["image"].stream) as photo:
            image = np.array(ImageOps.exif_transpose(photo).convert("RGB"))
    except Exception:
        return jsonify(error="Could not read that image."), 400

    # Separate cards from the table (cards should be lighter than the table)
    cutoff = 150
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    _, mask = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_images = []
    card_owners = []

    split_y = image.shape[0] / 2  # Top half = dealer, bottom half = player

    width, height = 224, 320
    target = np.float32([[0, 0], [width - 1, 0],
                         [width - 1, height - 1], [0, height - 1]])

    min_area = 0.01 * image.shape[0] * image.shape[1]

    for card_contour in contours:
        if cv2.contourArea(card_contour) < min_area:
            continue

        rect = cv2.minAreaRect(card_contour)
        points = cv2.boxPoints(rect)
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        points = points[np.argsort(angles)]

        # Start on a short edge so the card comes out portrait
        side_lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        if side_lengths[0] + side_lengths[2] > side_lengths[1] + side_lengths[3]:
            points = np.roll(points, -1, axis=0)

        matrix = cv2.getPerspectiveTransform(points, target)
        crop = cv2.warpPerspective(image, matrix, (width, height))
        card_images.append(Image.fromarray(crop))
        card_owners.append("Dealer" if center[1] < split_y else "Player")

    player_hand = []
    dealer_hand = []

    with torch.no_grad():
        for card_image, owner in zip(card_images, card_owners):
            inputs = transform(card_image).unsqueeze(0).to(device)
            prediction = model(inputs).argmax(dim=1).item()
            rank = ranks[prediction]
            (dealer_hand if owner == "Dealer" else player_hand).append(rank)

    return jsonify(
        player_total=hand_total(player_hand),
        dealer_total=hand_total(dealer_hand),
    )


if __name__ == "__main__":
    app.run(port=5000)
