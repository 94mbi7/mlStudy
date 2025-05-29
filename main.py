import cv2
import torch
import numpy as np
import time

print("Loading MiDaS model...")
model_type = "DPT_Hybrid"
try:
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
except Exception as e:
    print(f"Error loading MiDaS: {e}")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"MiDaS loaded on {device}")

print("Loading YOLOv5 model...")
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_model.to(device)
    print("YOLOv5 loaded")
except Exception as e:
    print(f"Error loading YOLOv5: {e}")
    exit(1)

REF_OBJECT_WIDTH_CM = 21.0

calibrated = False
pixel_per_cm = None
ref_depth = None

def get_depth_map(frame):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(rgb_frame).to(device)
        
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        depth_map = np.max(depth_map) - depth_map
        
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map
    except Exception as e:
        print(f"Error in depth estimation: {e}")
        return np.zeros(frame.shape[:2])

mouse_drawing = False
mouse_start_x, mouse_start_y = -1, -1
mouse_end_x, mouse_end_y = -1, -1
calibration_frame = None

def mouse_callback(event, x, y, flags, param):
    global mouse_drawing, mouse_start_x, mouse_start_y, mouse_end_x, mouse_end_y, calibration_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_drawing = True
        mouse_start_x, mouse_start_y = x, y
        mouse_end_x, mouse_end_y = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_drawing:
            mouse_end_x, mouse_end_y = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_drawing = False
        mouse_end_x, mouse_end_y = x, y

def calibrate(frame, depth_map):
    global pixel_per_cm, ref_depth, calibrated
    global mouse_drawing, mouse_start_x, mouse_start_y, mouse_end_x, mouse_end_y, calibration_frame
    
    print("\nCALIBRATION MODE")
    print("1. Place your reference object (A4 card/paper) in view")
    print("2. Click and drag to draw a bounding box around the object")
    print("3. Press SPACE to confirm selection")
    print("4. Press ESC to cancel")
    
    mouse_drawing = False
    mouse_start_x = mouse_start_y = mouse_end_x = mouse_end_y = -1
    calibration_frame = frame.copy()
    
    cv2.namedWindow("Calibration - Draw Box Around Reference Object", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Calibration - Draw Box Around Reference Object", mouse_callback)
    
    while True:
        display_frame = calibration_frame.copy()
        
        if mouse_start_x != -1 and mouse_start_y != -1:
            cv2.rectangle(display_frame, 
                         (mouse_start_x, mouse_start_y), 
                         (mouse_end_x, mouse_end_y), 
                         (0, 255, 0), 2)
            
            width = abs(mouse_end_x - mouse_start_x)
            height = abs(mouse_end_y - mouse_start_y)
            cv2.putText(display_frame, f"Size: {width}x{height} px", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display_frame, "Click & Drag to select | SPACE=Confirm | ESC=Cancel", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Calibration - Draw Box Around Reference Object", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            if mouse_start_x != -1 and mouse_start_y != -1 and mouse_end_x != -1 and mouse_end_y != -1:
                x1 = min(mouse_start_x, mouse_end_x)
                y1 = min(mouse_start_y, mouse_end_y)
                x2 = max(mouse_start_x, mouse_end_x)
                y2 = max(mouse_start_y, mouse_end_y)
                w = x2 - x1
                h = y2 - y1
                
                if w > 10 and h > 10:
                    cv2.destroyWindow("Calibration - Draw Box Around Reference Object")
                    
                    x, y = x1, y1
                    x, y = max(0, x), max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                    
                    depth_roi = depth_map[y:y+h, x:x+w]
                    if depth_roi.size == 0:
                        print("Invalid depth region. Calibration failed.")
                        return False
                    
                    avg_depth = np.mean(depth_roi)
                    pixel_per_cm = w / REF_OBJECT_WIDTH_CM
                    ref_depth = avg_depth
                    
                    print(f"Calibration successful!")
                    print(f"   Selected area: {w}x{h} pixels")
                    print(f"   Scale: {pixel_per_cm:.2f} pixels/cm")
                    print(f"   Reference depth: {ref_depth:.3f}")
                    print("   Starting real-time measurement...")
                    
                    calibrated = True
                    return True
                else:
                    print("Selection too small. Please select a larger area.")
            else:
                print("No selection made. Please draw a bounding box.")
                
        elif key == 27:
            cv2.destroyWindow("Calibration - Draw Box Around Reference Object")
            print("Calibration cancelled.")
            return False
    
    return False

def measure_objects(frame, depth_map, detections):
    if not calibrated or pixel_per_cm is None:
        return []
    
    results = []
    frame_height, frame_width = frame.shape[:2]
    
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(x1 + 1, min(x2, frame_width))
        y2 = max(y1 + 1, min(y2, frame_height))
        
        w, h = x2 - x1, y2 - y1
        
        depth_roi = depth_map[y1:y2, x1:x2]
        if depth_roi.size == 0:
            continue

        depth_roi = depth_roi.astype(np.float32)

        blurred_depth = cv2.GaussianBlur(depth_roi, (5, 5), 0)
            
        avg_depth = np.mean(blurred_depth)
        
        if ref_depth is not None and avg_depth > 0:
            depth_ratio = ref_depth / max(avg_depth, 0.001)
        else:
            depth_ratio = 1.0
        
        real_w_cm = (w / pixel_per_cm) * depth_ratio
        real_h_cm = (h / pixel_per_cm) * depth_ratio
        
        results.append({
            "bbox": (x1, y1, x2, y2),
            "size_cm": (real_w_cm, real_h_cm),
            "class": int(cls),
            "confidence": float(conf),
            "depth": avg_depth
        })
    
    return results

def draw_results(frame, results, class_names):
    for res in results:
        x1, y1, x2, y2 = res["bbox"]
        w_cm, h_cm = res["size_cm"]
        cls_id = res["class"]
        conf = res["confidence"]
        depth = res["depth"]
        
        if cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = f"Class_{cls_id}"
        
        label = f"{class_name} ({conf:.2f})"
        size_text = f"{w_cm:.1f}x{h_cm:.1f}cm"
        depth_text = f"d:{depth:.2f}"
        
        color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 45), (x1 + max(label_size[0], 150), y1), color, -1)
        
        cv2.putText(frame, label, (x1 + 2, y1 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, size_text, (x1 + 2, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def draw_calibration_status(frame):
    status_text = "calibrated" if calibrated else "Press 'c' to Calibrate"
    color = (0, 255, 0) if calibrated else (0, 0, 255)
    
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    if not calibrated:
        cv2.putText(frame, "Place A4 paper/card in view, then press 'c'", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    global calibrated
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera initialized")
    print("\nControls:")
    print("  'c' - Calibrate with reference object")
    print("  'r' - Reset calibration")
    print("  'esc' - Exit")
    print("\nCALIBRATION INSTRUCTIONS:")
    print("  1. Press 'c' to start calibration")
    print("  2. Click and drag to draw a box around your a4 paper/card")
    print("  3. Press space to confirm, esc to cancel")
    print("\nStarting video stream...")
    
    class_names = yolo_model.names
    
    fps_counter = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        frame = cv2.flip(frame, 1)
        
        draw_calibration_status(frame)
        
        if calibrated:
            try:
                depth_map = get_depth_map(frame)
                
                results = yolo_model(frame)
                detections = results.xyxy[0].cpu().numpy()
                
                detections = [det for det in detections if det[4] > 0.4]
                
                if len(detections) > 0:
                    measured = measure_objects(frame, depth_map, detections)
                    draw_results(frame, measured, class_names)
                
            except Exception as e:
                print(f"Error in processing: {e}")
        
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Real-Time Object Size Measurement", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('c') or key == ord('C'):
            depth_map = get_depth_map(frame)
            calibrate(frame, depth_map)
        elif key == ord('r') or key == ord('R'):
            calibrated = False
            pixel_per_cm = None
            ref_depth = None
            print("Calibration reset")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")

if __name__ == "__main__":
    main()