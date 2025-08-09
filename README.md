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
   - You can test out the RF-DETR real-time object detection model [Here](https://app.roboflow.com/invention-time/pokemon-card-detection-haiqn/models/pokemon-card-detection-haiqn/3) for a fun example. (Must have Roboflow account, it's fre,e dont worry)
3. Extracted the dataset for the YOLOV8 model from Roboflow
   - This gives us a training, validation, and testing image and labeling data split for all images and labels, as well as a YAML file that contains the number of classes, class names, and local location of the data splits.
     - You can see the data_example folder to see an image and coordinate pairing for demonstration. For a normal bounding box, the first two coordinates are the x and y center of the box relative to the grid cell, and the last two are the height and width of the bounding box.
   - Additionally, given the small memory size of the IMX500 [8MB](https://developer.sony.com/imx500), we will need to quantize and compress this model into a YOLOV8N (smallest YOLOV8) model using Ultralytic.
4. Extracted a YOLOV8N model from Ultralytics
   - Thankfully, Ultralytics has a demo on how to export a YOLO model to the IMX500 Camera



