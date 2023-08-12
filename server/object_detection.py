# Load model with pretrained weights
from super_gradients.training import models
from super_gradients.common.object_names import Models
model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")

       
#runs infernce and sends the return the boundinx boxes 
def run_inference(image):
    prediction = model.predict(image)
    output = prediction._images_prediction_lst[0]
    pred_image = output.image
    bboxes = output.prediction.bboxes_xyxy
    confidences = output.prediction.confidence
    classes = output.class_names
    class_name_indexes = output.prediction.labels.astype(int)
    inference = { 'image' : pred_image,'bbox' : bboxes , 'conf' : confidences , 'classes' : classes , 'indexes' : class_name_indexes}
    return inference

  