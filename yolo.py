import cv2.dnn
import numpy as np
import os


class YOLO:

    def __init__(self):
        print("Loading YOLOv8n model...")
        self.model = cv2.dnn.readNetFromONNX("yolov8n.onnx")
        print("Model loaded successfully")

    def predict(self, input_image, output=None):
        original_image = input_image.copy()
        [height, width, _] = input_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = input_image

        # Calculate scale factor
        scale = length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1 / 255, size=(640, 640), swapRB=True
        )

        self.model.setInput(blob)

        # Perform inference
        outputs = self.model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(
                classes_scores
            )
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0],
                    outputs[0][i][1],
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

                # Apply NMS (Non-maximum suppression)
                result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5, 0.5)

                # List of confidence, xcenter, ycenter, width, height
                detections = []

                # Iterate through NMS results to draw bounding boxes and labels
                for i in range(len(result_boxes)):
                    index = result_boxes[i]
                    box = boxes[index]

                    detections.append(
                        [
                            scores[index],
                            round(box[0] * scale),
                            round(box[1] * scale),
                            round(box[2] * scale),
                            round(box[3] * scale),
                        ]
                    )

                    if output:
                        self._draw_bounding_box(
                            original_image,
                            round((box[0] - box[2] / 2) * scale),
                            round((box[1] - box[3] / 2) * scale),
                            round((box[0] + box[2] / 2) * scale),
                            round((box[1] + box[3] / 2) * scale),
                        )

        if output:
            cv2.imwrite(output, original_image)

        return detections

    def _square_padding(self, image, size):
        # zero padding
        h, w = image.shape[:2]
        if h > w:
            pad = (h - w) // 2
            image = cv2.copyMakeBorder(
                image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        else:
            pad = (w - h) // 2
            image = cv2.copyMakeBorder(
                image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        # resize
        image = cv2.resize(image, (size, size))

        return image

    def _draw_bounding_box(self, img, x, y, x_plus_w, y_plus_h):
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 0, 255), 2)


if __name__ == "__main__":
    yolo = YOLO()

    dataset = "./examples30/images"

    for img in os.listdir(dataset):
        image = cv2.imread(os.path.join(dataset, img))
        detections = yolo.predict(image, output=f"./results/{img}")
        print(f"Predicted {len(detections)} objects in {img}")
