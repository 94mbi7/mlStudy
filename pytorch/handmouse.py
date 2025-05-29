import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import pyautogui
import numpy as np

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

        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 3) 
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

model = HandGestureModel()

model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),

])

def decode_gesture(output_tensor):
    """
    Convert model output logits into a gesture label.
    """
    gestures = ["move", "click", "none"]
    _, predicted_idx = torch.max(output_tensor, 1)
    return gestures[predicted_idx.item()]

def main():
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

        frame = cv2.flip(frame, 1)

  
        input_frame = cv2.resize(frame, (64, 64))
        input_tensor = transform(input_frame)
        input_tensor = input_tensor.unsqueeze(0) 

        with torch.no_grad():
            output = model(input_tensor)
        gesture = decode_gesture(output)

        action_text = "No action"
        if gesture == "move":

            pyautogui.moveRel(10, 0, duration=0.1)
            action_text = "Moving Mouse"
        elif gesture == "click":
            pyautogui.click()
            action_text = "Mouse Click"

        cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, action_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
