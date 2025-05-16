import cv2
import numpy as np
import argparse
import time
from ultralytics import YOLO
import torch

class YOLOShowcase:
    def __init__(self):
        self.current_task = "detect"  # Default task
        self.available_tasks = ["detect", "segment", "classify", "pose", "obb"]
        
        # Load YOLO models for different tasks
        print("Loading YOLO models...")
        self.models = {
            "detect": YOLO("yolo11n.pt"),
            "segment": YOLO("yolo11n-seg.pt"),
            "classify": YOLO("yolo11n-cls.pt"),
            "pose": YOLO("yolo11n-pose.pt"),
            "obb": YOLO("yolo11n-obb.pt")
        }
        print("Models loaded successfully!")
        
        # Colors for visualization
        self.task_colors = {
            "detect": (0, 255, 0),    # Green
            "segment": (0, 0, 255),   # Red
            "classify": (255, 0, 0),  # Blue
            "pose": (255, 255, 0),    # Cyan
            "obb": (255, 0, 255)      # Magenta
        }
        
        # Generate a distinctive color palette for segmentation objects
        self.segmentation_colors = self.generate_color_palette(80)  # COCO has 80 classes
        
        # For FPS calculation
        self.prev_time = 0
        self.curr_time = 0
    
    def generate_color_palette(self, num_colors):
        """Generate a list of distinct colors for visualization"""
        np.random.seed(42)  # For reproducibility
        
        # Generate visually distinct colors using HSV color space
        colors = []
        for i in range(num_colors):
            # Use golden ratio to spread hues evenly
            h = (i * 0.618033988749895) % 1.0
            s = 0.7 + 0.3 * ((i // 10) % 3) / 2
            v = 0.85 + 0.15 * ((i // 30) % 2)
            
            # Convert HSV to RGB
            h_i = int(h * 6)
            f = h * 6 - h_i
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            
            if h_i == 0:
                r, g, b = v, t, p
            elif h_i == 1:
                r, g, b = q, v, p
            elif h_i == 2:
                r, g, b = p, v, t
            elif h_i == 3:
                r, g, b = p, q, v
            elif h_i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
            
            # Convert to 0-255 range and add to colors list
            colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR format for OpenCV
        
        return colors

    def switch_task(self, new_task):
        """Switch between different YOLO tasks"""
        if new_task in self.available_tasks:
            self.current_task = new_task
            print(f"Switched to {self.current_task} task")
            return True
        else:
            print(f"Invalid task: {new_task}")
            print(f"Available tasks: {self.available_tasks}")
            return False

    def process_frame(self, frame):
        """Process a single frame with the current YOLO task"""
        # Calculate FPS
        self.curr_time = time.time()
        fps = 1 / (self.curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = self.curr_time
        
        # Create a copy of the frame for drawing
        result_frame = frame.copy()
        
                # Get current model
        model = self.models[self.current_task]
        color = self.task_colors[self.current_task]
        
        try:
            if self.current_task == "classify":
                # Classification task
                results = model(frame)
                result = results[0]
                
                # Get top prediction
                if len(result.probs.top5) > 0:
                    top_class_id = result.probs.top5[0]
                    top_class_name = result.names[top_class_id]
                    top_class_conf = result.probs.top5conf[0].item()
                    
                    # Draw classification result
                    label = f"{top_class_name}: {top_class_conf:.2f}"
                    cv2.putText(result_frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                # All other tasks use a similar API
                results = model.predict(frame, verbose=False)
                result = results[0]
                
                if self.current_task == "detect":
                    # Draw detection boxes
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = f"{result.names[cls_id]}: {conf:.2f}"
                        
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                elif self.current_task == "segment":
                    # Draw segmentation masks
                    if hasattr(result, 'masks') and result.masks is not None:
                        # Create a blank overlay for all masks
                        overlay = np.zeros_like(frame)
                        
                        for i, mask in enumerate(result.masks.data):
                            # Convert mask to numpy array
                            mask_np = mask.cpu().numpy().astype(np.uint8)
                            mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                            
                            # Get the class ID for color selection
                            cls_id = int(result.boxes[i].cls[0]) if i < len(result.boxes) else i % len(self.segmentation_colors)
                            
                            # Select a distinct color for this object
                            color = self.segmentation_colors[cls_id]
                            
                            # Apply color to this mask area in the overlay
                            overlay[mask_np > 0] = color
                            
                            # Draw bounding box and label for this object
                            if i < len(result.boxes):
                                box = result.boxes[i]
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                cls_id = int(box.cls[0])
                                class_name = result.names[cls_id]
                                label = f"{class_name}: {conf:.2f}"
                                
                                # Use the same color as the segmentation mask
                                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Add a filled rectangle behind text for better visibility
                                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                cv2.rectangle(result_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                                cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # Apply the combined overlay with alpha blending
                        alpha = 0.5
                        cv2.addWeighted(result_frame, 1 - alpha, overlay, alpha, 0, result_frame)
                
                elif self.current_task == "pose":
                    # Draw pose keypoints and connections
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        for kpt in result.keypoints.data:
                            # Set confidence threshold for keypoints
                            confidence_threshold = 0.5
                            
                            # Create keypoint visibility mask based on confidence
                            visible = kpt[:, 2] > confidence_threshold
                            
                            # Draw keypoints with sufficient confidence
                            for i, p in enumerate(kpt):
                                if visible[i]:  # Only draw if keypoint passes confidence check
                                    x, y = int(p[0]), int(p[1])
                                    cv2.circle(result_frame, (x, y), 3, color, -1)
                            
                            # Draw connections (simplified - actual connections depend on the specific pose model)
                            # COCO dataset connections (simplified version)
                            connections = [
                                (0, 1), (0, 2), (1, 3), (2, 4),  # Face and neck
                                (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                                (5, 6), (5, 11), (6, 12),  # Body
                                (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
                            ]
                            
                            # Only draw connections if both keypoints have sufficient confidence
                            for connection in connections:
                                p1_idx, p2_idx = connection
                                if p1_idx < len(kpt) and p2_idx < len(kpt):
                                    if visible[p1_idx] and visible[p2_idx]:
                                        p1 = (int(kpt[p1_idx][0]), int(kpt[p1_idx][1]))
                                        p2 = (int(kpt[p2_idx][0]), int(kpt[p2_idx][1]))
                                        
                                        # Additional sanity check - skip lines that are too long
                                        # (often indicates false connections)
                                        distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                                        max_reasonable_distance = frame.shape[1] * 0.7  # 70% of frame width
                                        
                                        if distance < max_reasonable_distance:
                                            cv2.line(result_frame, p1, p2, color, 2)
                
                elif self.current_task == "obb":
                    # Draw oriented bounding boxes
                    if hasattr(result, 'obb') and result.obb is not None:
                        for i, box in enumerate(result.obb.data):
                            # Get the 4 corners of the oriented box
                            points = box[:8].reshape(-1, 2).cpu().numpy().astype(np.int32)
                            # Draw the oriented box
                            cv2.polylines(result_frame, [points], isClosed=True, color=color, thickness=2)
                            
                            # Add label if available
                            if i < len(result.boxes):
                                box_info = result.boxes[i]
                                conf = float(box_info.conf[0])
                                cls_id = int(box_info.cls[0])
                                label = f"{result.names[cls_id]}: {conf:.2f}"
                                
                                # Place text at the first point of the OBB
                                text_point = (points[0][0], points[0][1] - 10)
                                cv2.putText(result_frame, label, text_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        except Exception as e:
            error_msg = f"Error processing frame: {str(e)}"
            print(error_msg)
            cv2.putText(result_frame, error_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
                # Add FPS and current task info
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Task: {self.current_task}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.task_colors[self.current_task], 2)
        cv2.putText(result_frame, "Press 1-5 to switch tasks, Q to quit", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="YOLO Showcase")
    parser.add_argument('--source', type=str, default='0', help='Source (webcam index, video file, or image file)')
    args = parser.parse_args()
    
    # Initialize YOLO showcase
    showcase = YOLOShowcase()
    
    # Open video source
    if args.source.isdigit():
        # Webcam
        cap = cv2.VideoCapture(int(args.source))
    else:
        # Video or image file
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Create window
    window_name = "YOLO Showcase"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame")
            break
        
        # Process frame with current task
        result_frame = showcase.process_frame(frame)
        
        # Show result
        cv2.imshow(window_name, result_frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit
            break
        elif key == ord('1'):
            # Switch to detection
            showcase.switch_task("detect")
        elif key == ord('2'):
            # Switch to segmentation
            showcase.switch_task("segment")
        elif key == ord('3'):
            # Switch to classification
            showcase.switch_task("classify")
        elif key == ord('4'):
            # Switch to pose estimation
            showcase.switch_task("pose")
        elif key == ord('5'):
            # Switch to OBB
            showcase.switch_task("obb")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
