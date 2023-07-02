import os
import csv
import json
import torch
import random
import requests
import logging
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def collate_fn(batch):
    return tuple(zip(*batch))

class MMQuery(Dataset):
    def __init__(
        self,
        data_path,
        split,
    ):
        self.hyponyms = []
        self.hypernyms = []
        self.image_paths = []
        self.tags = []
        self.split = split
        
        with open(data_path, 'r') as f:
            data = json.load(f)
            for d in data:
                self.hyponyms.append(d['hyponym'])
                self.hypernyms.append(d['hypernym'])
                self.image_paths.append(d['image_path'])
                self.tags.append(d['label'])
        assert len(self.tags)==len(self.hypernyms)==len(self.hyponyms)==len(self.image_paths)
                
    def __len__(self):
        return len(self.tags)
    
    def __getitem__(self, index):
        hyponym = self.hyponyms[index]
        hypernym = self.hypernyms[index]
        image_path = self.image_paths[index]
        tag = self.tags[index]
        
        image = Image.open(image_path).convert('RGB')
        image_resized = T.Resize(size=(224, 224))(image)
        image_tensor = T.ToTensor()(image_resized)
        
        if self.split:
            image_tensors = []
            boxes = [
                (0, 0, 112, 112),
                (112, 0, 224, 112),
                (0, 112, 112, 224),
                (112, 112, 224, 224),
            ]
            image = Image.open(image_path).convert('RGB')
            image_resized = T.Resize(size=(224, 224))(image)
            image_tensor = T.ToTensor()(image_resized)
            image_tensors.append(image_tensor)
            for box in boxes:
                cropped_image = image.crop(box)
                image_resized = T.Resize(size=(224, 224))(cropped_image)
                image_tensor = T.ToTensor()(image_resized)
                image_tensors.append(image_tensor)
            return hypernym, hyponym, image_tensors, tag
        
        else:
            image = Image.open(image_path).convert('RGB')
            image_resized = T.Resize(size=(224, 224))(image)
            image_tensor = T.ToTensor()(image_resized)
            return hypernym, hyponym, image_tensor, tag
      
class MultitaskDataset(Dataset):
    def __init__(
        self,
        data_path,
        split,
    ):
        self.hyponyms = []
        self.hypernyms = []
        self.image_paths = []
        self.negatives = []
        self.split = split
        
        with open(data_path, 'r') as f:
            data = json.load(f)
            for d in data:
                self.hyponyms.append(d['hyponym'])
                self.hypernyms.append(d['hypernym'])
                self.image_paths.append(d['image_path'])
                self.negatives.append(d['negative_hypernym'])
        assert len(self.negatives)==len(self.hypernyms)==len(self.hyponyms)==len(self.image_paths)
                
    def __len__(self):
        return len(self.hypernyms)
    
    def __getitem__(self, index):
        hyponym = self.hyponyms[index]
        hypernym = self.hypernyms[index]
        image_path = self.image_paths[index]
        negative = self.negatives[index]
        
        # image = Image.open(image_path).convert('RGB')
        # image_resized = T.Resize(size=(224, 224))(image)
        # image_tensor = T.ToTensor()(image_resized)
        
        if self.split:
            image_tensors = []
            boxes = [
                (0, 0, 112, 112),
                (112, 0, 224, 112),
                (0, 112, 112, 224),
                (112, 112, 224, 224),
            ]
            image = Image.open(image_path).convert('RGB')
            image_resized = T.Resize(size=(224, 224))(image)
            image_tensor = T.ToTensor()(image_resized)
            image_tensors.append(image_tensor)
            for box in boxes:
                cropped_image = image.crop(box)
                image_resized = T.Resize(size=(224, 224))(cropped_image)
                image_tensor = T.ToTensor()(image_resized)
                image_tensors.append(image_tensor)
            return hypernym, hyponym, image_tensors, negative
        
        else:
            image = Image.open(image_path).convert('RGB')
            image_resized = T.Resize(size=(224, 224))(image)
            image_tensor = T.ToTensor()(image_resized)
            return hypernym, hyponym, image_tensor, negative