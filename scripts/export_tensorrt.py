from ultralytics import YOLO

# Load the original PyTorch model
model = YOLO("yolo11n.pt") 

# Export with dynamic=True to allow variable batch sizes
model.export(format="engine", dynamic=True, batch=2)
