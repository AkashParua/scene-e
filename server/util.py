
from ultralytics import YOLO
from rembg import remove
from PIL import Image
import cv2

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

def inference(img_path):
    bbox = None
    classid = None
    names = None
    results = model(img_path)  # predict on an image
    for result in results :
        #xmid, ymid ,h ,w 
        bbox = result.boxes.xywh.tolist()
        classid = result.boxes.cls.tolist()
        names = result.names 
    return {'bbs': bbox , 'name': [nm for nm in map(lambda x : names[int(x)], classid)] }   
        
def extract_roi(img_path):
    bbox_paths = []
    result = inference(img_path)
    image = cv2.imread(img_path)
    for i,(bbox , name) in enumerate(zip(result['bbs'] , result['name'])) :
        save_path = f'{i}_{name}_{img_path}'
        xmid , ymid , w , h = bbox
        xmin = xmid - w/2
        ymin = ymid - h/2
        xmax = xmid + w/2
        ymax = ymid + h/2
        
        roi = image[int(ymin) : int(ymax) , int(xmin) : int(xmax)]
        
        cv2.imwrite(save_path , roi)
        
        inpath = Image.open(save_path)
        output = remove(inpath)
        output = output.convert('RGB')
        output.save(save_path)
        bbox_paths.append(save_path)
        
    return bbox_paths
                
        


