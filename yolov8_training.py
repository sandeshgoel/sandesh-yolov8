from ultralytics import YOLO
model = YOLO("yolov8s.pt")

results = model.train(data='config-lemon.yaml', epochs=50)

