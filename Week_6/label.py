import cv2
import os
import json
from copy import deepcopy

template = {
    "version": "0.4.16",
    "flags": {},
    "shapes": [
        {
            "label": "face",
            "text": "",
            "points": [
                [0, 0],
                [0, 0]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
    ],
    "imagePath": "",
    "imageData": None,
    "imageHeight": 0,
    "imageWidth": 0
}

image_dir = "image"
annotation_dir = "annotation"

os.makedirs(annotation_dir, exist_ok=True)

net = cv2.dnn.readNet("face-yolov3-tiny_41000.weights", "face-yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)

    if image is None:
        continue

    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    smallest_box = None
    smallest_area = float('inf')

    for out in outs:
        for detection in out:
            confidence = detection[4]
            if confidence > 0.9:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x1 = int(center_x - w / 2)
                y1 = int(center_y - h / 2)
                x2 = int(center_x + w / 2)
                y2 = int(center_y + h / 2)

                area = w * h
                if area < smallest_area:
                    smallest_area = area
                    smallest_box = [x1, y1, x2, y2]

    if smallest_box:
        ann = deepcopy(template)
        ann["imagePath"] = image_name
        ann["imageHeight"] = height
        ann["imageWidth"] = width
        ann["shapes"][0]["points"] = [
            [float(smallest_box[0]), float(smallest_box[1])],
            [float(smallest_box[2]), float(smallest_box[3])]
        ]

        json_name = os.path.splitext(image_name)[0] + ".json"
        json_path = os.path.join(annotation_dir, json_name)

        with open(json_path, "w") as f:
            json.dump(ann, f, indent=2)

print(f"Annotations saved in '{annotation_dir}'")
