from flask import Flask, jsonify, request
import torch
import json
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

model = resnet18(pretrained=True)
model.eval()

app = Flask(__name__)

with open('imagenet_classes.txt') as f:
  labels = [line.strip() for line in f.readlines()]

testing_types = [
  { 'name': 'unit testing', 'description': 'testing individual units of source code' }
]

transform = transforms.Compose([            #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(224),                #[3]
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
    mean=[0.485, 0.456, 0.406],                #[6]
    std=[0.229, 0.224, 0.225]                  #[7]
    )])

@app.route("/hello", methods=['POST'])
    def hello():
        return "Hello World!"

@app.route('/classify', methods=['POST'])
def predict():
    postedData = {"path": "/home/hasif/personal/python3-nginx-gunicorn/zeb.jpg"}
    path = postedData["path"]
    # print(postedData)
    img = Image.open(path)

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    out = model(batch_t)

    retJson = {}
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    # print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])
    # out = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
    for idx in indices[0][:1]:
        retJson[f'pred_{i}'] = str("label : {} and confidence : {}".format(labels[idx],percentage[idx].item()))

    return jsonify(retJson)
  
  
if __name__ == '__main__':
    app.run(debug=True)  