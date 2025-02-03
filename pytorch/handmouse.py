import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import pyautogui
import numpy as np

# -----------------------------
# Define the Hand Gesture Model
# -----------------------------
class HandGestureModel(nn.Module):
    def __init__(self):
        super(HandGestureModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Assuming input images are resized to 64x64:
        # Calculate the flattened size after convolutions and pooling.
        # 64x64 -> Conv/Pool layers -> approximately 32 channels of 8x8 (adjust as needed)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 classes: "move", "click", "none"
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Instantiate the model
model = HandGestureModel()

# In a real scenario, load your trained model weights:
# model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
# For this demo, we use the random weights as-is.
model.eval()

# -----------------------------
# Define Preprocessing Transform
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # Adjust this size if your model expects a different input
    transforms.ToTensor(),
    # If you normalized during training, add normalization here:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Gesture Decoder Function
# -----------------------------
def decode_gesture(output_tensor):
    """
    Convert model output logits into a gesture label.
    """
    gestures = ["move", "click", "none"]
    _, predicted_idx = torch.max(output_tensor, 1)
    return gestures[predicted_idx.item()]

# -----------------------------
# Main Loop: Capture, Predict, Act
# -----------------------------
def main():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Optional: Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Preprocess the frame
        # For simplicity, we use the full frame here.
        # In practice, crop to the hand region if possible.
        input_frame = cv2.resize(frame, (64, 64))
        input_tensor = transform(input_frame)
        input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

        # Get gesture prediction from the model
        with torch.no_grad():
            output = model(input_tensor)
        gesture = decode_gesture(output)

        # Map the predicted gesture to a mouse action
        action_text = "No action"
        if gesture == "move":
            # For demo: move the mouse by a fixed offset.
            # In a real app, compute the hand position to control cursor movement.
            pyautogui.moveRel(10, 0, duration=0.1)
            action_text = "Moving Mouse"
        elif gesture == "click":
            pyautogui.click()
            action_text = "Mouse Click"
        # else: "none" does nothing

        # Display gesture and action on the frame
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, action_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
