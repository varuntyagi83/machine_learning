from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObjectDetectionModel:
    def __init__(self, model_name="facebook/detr-resnet-50", revision="no_timm", threshold=0.9):
        self.processor = DetrImageProcessor.from_pretrained(model_name, revision=revision)
        self.model = DetrForObjectDetection.from_pretrained(model_name, revision=revision)
        self.threshold = threshold

    def detect_objects(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=self.threshold)[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            detection = {
                "label": self.model.config.id2label[label.item()],
                "confidence": round(score.item() * 100, 2),  # Convert confidence to percentage
                "box": box
            }
            detections.append(detection)

        return detections, image

def display_results(detections, image):
    # Display the image
    plt.imshow(image)

    ax = plt.gca()
    for detection in detections:
        # Draw boxes on the image
        box = detection["box"]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Display label and confidence
        plt.text(box[0], box[1], f"{detection['label']} {detection['confidence']}%",
                 bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

if __name__ == "__main__":
    # Example usage
    # url = "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
    url = "http://images.cocodataset.org/val2017/000000102805.jpg"
    model = ObjectDetectionModel()
    image_path = requests.get(url, stream=True).raw
    detections, image = model.detect_objects(image_path)

    for detection in detections:
        print(f"Detected {detection['label']} with confidence {detection['confidence']}% at location {detection['box']}")

    # Display the results on the image
    display_results(detections, image)
