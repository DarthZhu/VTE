import torch
import numpy as np

class Preprocessor():
    def __init__(self, tokenizer, extractor, config, device) -> None:
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.config = config
        self.device = device
       
    def preprocess_text(self, texts):
        encoded_texts = self.tokenizer(
            texts,
            add_special_tokens=False,
            max_length=self.config.max_len,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids = torch.tensor(encoded_texts["input_ids"]).to(self.device)
        attention_masks = torch.tensor(encoded_texts["attention_mask"]).to(self.device)
        return input_ids, attention_masks
    
    def preprocess_image(self, images):
        pixel_values = []
        for image in images:
            encoded_image = self.extractor(image)['pixel_values']
            pixel_values.append(encoded_image)
        pixel_values = torch.tensor(np.array(pixel_values)).squeeze().to(self.device)
        return pixel_values