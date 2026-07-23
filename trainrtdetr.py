from ultralytics import RTDETR

# 1. Load the pre-trained Vision Transformer model
model = RTDETR('rtdetr-l.pt')  # 'l' stands for large (standard for RT-DETR)

# 2. Train the model
print(" Starting RT-DETR Training...")
results = model.train(
    data='Master_Dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    plots=True
)
print("RT-DETR Training Complete!")