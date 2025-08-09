# POKEMON CARD DETECTOR
Developed a real-time Pokémon card stat reader using a custom-trained YOLOv8n object detection model and OCR. The goal is to pull live data from the web using https://pokemontcg.io/ to pull card pricing, Name, set_ID, and Card_Number to uniquely identify each card scanned.<br>

# WORKFLOW<br>
![CV_PFD](https://github.com/user-attachments/assets/fbe7533c-9856-40df-8ae3-682c2784a045) <br>

1. Gathered Pokémon card datasets from Kaggle [Example]( https://www.kaggle.com/datasets/bzhbzh35/pokemon-cards-per-set)<br>
2. Annotated data on Roboflow, labeling for Set_ID, Name, Card_Number, and the Card.
   - Augmentation steps include:
      - 90° Rotate: Clockwise, Counter-Clockwise
      - Shear: ±9° Horizontal, ±10° Vertical
      - Noise: Up to 0.1% of pixels
   - See annotated dataset [Here]( https://universe.roboflow.com/invention-time/pokemon-card-detection-haiqn)
   - You can test out the RF-DETR real-time object detection model [Here](https://app.roboflow.com/invention-time/pokemon-card-detection-haiqn/models/pokemon-card-detection-haiqn/3) for a fun example. (Must have Roboflow account, its free dont worry)
3. Extracted the dataset for YOLOV8 model that way we get the train, val, and test data split of both images and labels and a yaml file , as the only model that will fit on the SONYIMX500 AI Camera is a YOLOV8N model given its 8MB



