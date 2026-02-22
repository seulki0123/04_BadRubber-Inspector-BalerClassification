import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models

from baler_classification.utils import load_config
from .utils import sort_filename, combine_images

class Classifier:
    def __init__(
        self,
    ):
        config = load_config()["classifier"]
        checkpoint_path = config["checkpoint"]
        num_classes = config["num_classes"]
        self.class_names = config["class_names"]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.img_size = config["img_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.num_classes = self._load_checkpoint(checkpoint_path, num_classes)

    def _load_checkpoint(self, checkpoint_path: str, num_classes: Optional[int]):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        num_classes = self._detect_num_classes(ckpt)

        if not self._valid_model_structure(ckpt['model_state_dict'].keys()):
            raise ValueError("Invalid model structure")

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        return model, num_classes

    def _valid_model_structure(self, state_dict_keys: dict):
        has_simple_fc = 'fc.weight' in state_dict_keys and 'fc.bias' in state_dict_keys
        has_sequential_fc = 'fc.1.weight' in state_dict_keys or 'fc.4.weight' in state_dict_keys
        return has_simple_fc and not has_sequential_fc

    def _detect_num_classes(self, checkpoint: dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            fc_weight_keys = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]
            if fc_weight_keys:
                def get_fc_index(key):
                    import re
                    match = re.search(r'fc\.(\d+)\.weight', key)
                    return int(match.group(1)) if match else -1
                
                fc_weight_keys_sorted = sorted(fc_weight_keys, key=get_fc_index)
                last_fc_key = fc_weight_keys_sorted[-1]
                return state_dict[last_fc_key].shape[0]
        
        print("No FC layer found, returning default 3 classes")
        return 3

    def classify(self, top_path: str, bottom_path: str):
        top_path, bottom_path, _ = sort_filename(top_path, bottom_path)

        top_img = Image.open(top_path).convert("RGB")
        bottom_img = Image.open(bottom_path).convert("RGB")

        img = combine_images(top_img, bottom_img, self.img_size)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        class_number = int(np.argmax(probs))
        class_name = self.class_names[class_number]
        confidence = float(probs[class_number])

        return class_number, class_name, confidence