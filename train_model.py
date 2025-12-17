from ultralytics import YOLO

def train():
    # Load the smallest YOLO model (fastest on CPU)
    model = YOLO('yolov8n.pt') 

    # Start training
    results = model.train(
        data='data.yaml', 
        epochs=10,        # 10 loops over the data (good for a quick test)
        imgsz=640,
        device='cpu'      # Change 'cpu' to 0 if you have a dedicated NVIDIA GPU
    )

if __name__ == '__main__':
    train()