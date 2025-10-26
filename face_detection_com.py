import cv2
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# --- Configuration Constants ---
NUM_BENCHMARK_FRAMES = 100 # Number of frames to process during benchmark mode

# --- Setup for external libraries (YOLO) ---
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None 


# --- Initialization Functions ---

def initialize_webcam():
    """Initializes and returns the webcam video capture object."""
    # Ensure this function only attempts to open the camera, not display
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Ensure no other application is using the camera.")
        sys.exit(1)
    return cap

def setup_detectors():
    """Initializes and returns all necessary model objects."""
    detectors = {}

    # 1. Haar Cascade
    # FIX: Corrected typo from 'haascades' to 'haarcascades'
    detectors['haar'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 2. DNN-SSD (requires model files)
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    try:
        detectors['dnn'] = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    except Exception:
        print("Warning: DNN model files not found. DNN mode may fail.")
        detectors['dnn'] = None

    # 3. YOLOv8
    if YOLO is not None:
        try:
            detectors['yolo'] = YOLO('yolov8n-face.pt')
        except Exception as e:
            print(f"Warning: Failed to load YOLO model: {e}")
            detectors['yolo'] = None
    else:
        detectors['yolo'] = None
        
    return detectors


# --- Utility Functions ---

def draw_results(frame, boxes, color, label, fps):
    """Draws bounding boxes, confidence, label, and FPS on the frame."""
    if boxes:
        for (x1, y1, x2, y2, confidence_placeholder) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            text = f'{label}'
            
            if confidence_placeholder != 1.0:
                 conf = confidence_placeholder
                 text = f'{label}: {conf*100:.1f}%'

            cv2.putText(frame, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame

def process_haar_frame(frame, face_cascade):
    """Processes a single frame using Haar Cascade and returns results (x1, y1, x2, y2, 1.0) and fps."""
    t = cv2.getTickCount()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    time_taken = (cv2.getTickCount() - t) / cv2.getTickFrequency()
    fps = 1 / time_taken if time_taken > 0 else 0
    results = [(x, y, x + w, y + h, 1.0) for (x, y, w, h) in faces]
    return results, fps

def process_dnn_frame(frame, net):
    """Processes a single frame using DNN-SSD and returns results (x1, y1, x2, y2, confidence) and fps."""
    if net is None: return [], 0.0

    t = cv2.getTickCount()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    time_taken = (cv2.getTickCount() - t) / cv2.getTickFrequency()
    fps = 1 / time_taken if time_taken > 0 else 0

    results = []
    confidence_threshold = 0.5 
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            results.append((x1, y1, x2, y2, confidence))
    return results, fps

def process_yolo_frame(frame, model):
    """Processes a single frame using YOLOv8 Face and returns results (x1, y1, x2, y2, confidence) and fps."""
    if model is None: return [], 0.0

    t = cv2.getTickCount()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_list = model.predict(rgb_frame, verbose=False)
    time_taken = (cv2.getTickCount() - t) / cv2.getTickFrequency()
    fps = 1 / time_taken if time_taken > 0 else 0

    results = []
    for result in results_list:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            results.append((x1, y1, x2, y2, conf))
    return results, fps


# --- Benchmarking and Plotting Functions ---

def benchmark_detector(cap, processor, model, label, num_frames=NUM_BENCHMARK_FRAMES):
    """Runs a detector for a specified number of frames and collects FPS data."""
    print(f"\n--- Running Benchmark for: {label} ({num_frames} frames) ---")
    
    # Skip if model is missing (DNN/YOLO only)
    if model is None and label != 'Haar Cascade':
        print(f"Skipping {label}: Model not available.")
        return []

    fps_data = []
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame during benchmark. Stopping early.")
            break
        
        _, fps = processor(frame, model)
        fps_data.append(fps)
        
        if (i + 1) % 100 == 0 or i == num_frames - 1:
            print(f"  Processed {i+1}/{num_frames} frames...")
            
    return fps_data


def generate_graphs(all_fps_data):
    """Generates and displays comparison graphs based on collected FPS data."""
    
    labels = list(all_fps_data.keys())
    filtered_labels = [label for label in labels if all_fps_data[label]]
    filtered_data = [all_fps_data[label] for label in filtered_labels]
    
    if not filtered_labels:
        print("\nNo valid FPS data collected for graphing.")
        return

    avg_fps = [np.mean(data) for data in filtered_data]
    colors = ['green', 'blue', 'red']
    
    # --- 1. Average FPS Bar Chart ---
    plt.figure(figsize=(10, 6))
    plt.bar(filtered_labels, avg_fps, color=colors[:len(filtered_labels)])
    for i, avg in enumerate(avg_fps):
        plt.text(i, avg + 1, f'{avg:.2f}', ha='center', va='bottom', fontsize=12)
        
    plt.title('Average Performance Comparison (FPS)')
    plt.xlabel('Detection Method')
    plt.ylabel('Average Frames Per Second (FPS)')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

    # --- 2. FPS Stability Over Time (Line Plot) ---
    plt.figure(figsize=(12, 7))
    for i, label in enumerate(filtered_labels):
        data = filtered_data[i]
        plt.plot(data, label=f'{label} (Avg: {avg_fps[i]:.2f} FPS)', color=colors[i], alpha=0.7)

    plt.title(f'FPS Stability Over {NUM_BENCHMARK_FRAMES} Frames')
    plt.xlabel('Frame Index')
    plt.ylabel('Instantaneous Frames Per Second (FPS)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()


def run_benchmark_mode(detectors):
    """Coordinates the benchmark run and graph generation."""
    detector_map = {
        'Haar Cascade': (process_haar_frame, detectors['haar']),
        'DNN-SSD': (process_dnn_frame, detectors['dnn']),
        'YOLOv8': (process_yolo_frame, detectors['yolo'])
    }
    
    all_fps_data = {}
    
    # Run benchmark for each detector sequentially and cleanly
    for label, (processor, model) in detector_map.items():
        # Only benchmark if the model is initialized or it's Haar
        if model is not None or label == 'Haar Cascade':
            
            # 1. Initialize CAP for this specific benchmark run
            cap_local = None
            try:
                cap_local = initialize_webcam()
                
                # 2. Run benchmark using the local cap object
                fps_data = benchmark_detector(cap_local, processor, model, label)
                
                if fps_data:
                    all_fps_data[label] = fps_data

            finally:
                # 3. CRITICAL CLEANUP: Release the camera and destroy all windows immediately
                if cap_local:
                    cap_local.release()
                cv2.destroyAllWindows()
                
                # The CRITICAL FIX: Increased pause to 2.0s to allow the OS/driver to fully reset the camera resource.
                print(f"--- Pausing 2.0s for {label} cleanup... ---")
                time.sleep(2.0) 

    # Generate and show graphs
    if len(all_fps_data) > 0:
        generate_graphs(all_fps_data)
    else:
        print("\nNo performance data collected. Check model file availability and webcam access.")


# --- Main Detection Loop Functions ---

def run_single_detector(cap, detectors, mode):
    """Runs a single detector loop (haar, dnn, or yolo)."""
    detector_map = {
        'haar': (process_haar_frame, detectors['haar'], (0, 255, 0), 'Haar Cascade'),
        'dnn': (process_dnn_frame, detectors['dnn'], (255, 0, 0), 'DNN-SSD'),
        'yolo': (process_yolo_frame, detectors['yolo'], (0, 0, 255), 'YOLOv8')
    }
    
    processor, model, color, label = detector_map[mode]
    # FIX: Define a constant window name outside the loop to prevent multiple windows from opening.
    window_name = f'{label} Detection' 

    if model is None and mode != 'haar':
        print(f"Cannot run {mode} mode due to missing model or library.")
        return

    print(f"\n--- Running Single Detector: {label} ---")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        annotated_frame = frame.copy()
        boxes, fps = processor(annotated_frame, model)
        
        draw_results(annotated_frame, boxes, color, label, fps)
        
        # FIX: Use the constant window_name for stable display
        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def run_comparison_detector(cap, detectors):
    """Runs all three detectors simultaneously for comparison."""
    if detectors['dnn'] is None or detectors['yolo'] is None:
        print("\n--- WARNING: Cannot run full comparison mode. DNN or YOLO models are missing. ---")
        return

    print("\n--- Running Comparison Mode (Three Windows) ---")
    print("Observe speed (FPS) and accuracy simultaneously. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # 1. Haar Cascade
        haar_frame = frame.copy()
        haar_boxes, haar_fps = process_haar_frame(haar_frame, detectors['haar'])
        draw_results(haar_frame, haar_boxes, (0, 255, 0), 'Haar', haar_fps)
        cv2.imshow('1. Haar Cascade (Green)', haar_frame)
        
        # 2. DNN-SSD
        dnn_frame = frame.copy()
        dnn_boxes, dnn_fps = process_dnn_frame(dnn_frame, detectors['dnn'])
        draw_results(dnn_frame, dnn_boxes, (255, 0, 0), 'DNN-SSD', dnn_fps)
        cv2.imshow('2. DNN-SSD (Blue)', dnn_frame)

        # 3. YOLOv8
        yolo_frame = frame.copy()
        yolo_boxes, yolo_fps = process_yolo_frame(yolo_frame, detectors['yolo'])
        draw_results(yolo_frame, yolo_boxes, (0, 0, 255), 'YOLOv8 (Red)', yolo_fps)
        cv2.imshow('3. YOLOv8 (Red)', yolo_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Main Execution ---

def main():
    """Main function to parse arguments and select the detection method."""
    if len(sys.argv) < 2:
        print("Usage: python face_detection_comparison.py [haar | dnn | yolo | compare | benchmark]")
        print("Example: python face_detection_comparison.py benchmark")
        return

    mode = sys.argv[1].lower()
    
    cap = None
    try:
        detectors = setup_detectors()

        if mode == 'benchmark':
            # Benchmark mode handles its own camera initialization/cleanup internally
            run_benchmark_mode(detectors)
            
        elif mode in ['haar', 'dnn', 'yolo', 'compare']:
            # Real-time modes only initialize the camera once
            cap = initialize_webcam()
            if mode in ['haar', 'dnn', 'yolo']:
                run_single_detector(cap, detectors, mode)
            elif mode == 'compare':
                run_comparison_detector(cap, detectors)
            
        else:
            print(f"Invalid mode: {mode}. Choose one of: haar, dnn, yolo, compare, benchmark.")
            
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        
    finally:
        # Cleanup the main CAP object if it was opened
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
