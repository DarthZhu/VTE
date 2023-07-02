import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, ResNetModel, AutoFeatureExtractor, BertTokenizer, ResNetConfig

from models.losses import InfoNCE

class CodebookForImage(nn.Module):
    def __init__(self, config):
        super().__init__()
        codes = torch.randn(size=(config.num_codes, config.codebook_embedding_size))
        self.codes = nn.Parameter(codes, requires_grad=False)  # num * embedding size
        self.convert_distance = nn.Softmax(dim=1)
        
        self.cluster_contrast_loss = InfoNCE(config.temperature)
        
    def forward(self, image_tensor):
        indexes = self.index(image_tensor)
        codes = self.get_codes(indexes)
        codes = image_tensor + (codes - image_tensor).detach() # stop gradient

        return indexes, codes
    
    def train_forward(self, image_tensor):        
        indexes = self.index(image_tensor)
        codes = self.get_codes(indexes)
        codes = image_tensor + (codes - image_tensor).detach() # stop gradient
        
        _, cluster_loss = self.cluster_contrast_loss(image_tensor, codes)
        
        return indexes, codes, cluster_loss
    
    def index(self, inputs_embed):
        """get index based on euclidean distance

        Args:
            inputs_embed (torch.Tensor): batch_size * embedding_size
            self.codes (torch.Tensor): num_codes * embedding_size
        """
        vec_distance = torch.cdist(inputs_embed, self.codes) # should be batch * num_codes
        indexes = torch.argmin(vec_distance, dim=1)
        
        return indexes
    
    def get_codes(self, indexes):
        """return codes based on the input indexes

        Args:
            indexes (torch.Tensor): batch_size * 1

        Returns:
            torch.Tensor: returned codes
        """
        return self.codes.index_select(dim=0, index=indexes)
    
    @torch.no_grad()
    def _momentum_update(self, mu, cluster_result):
        """perform exponential moving average

        Args:
            mu (float): temperature of moving average
            cluster_result (torch.Tensor): c_i in cluster_result should be average of encoder output clustered to centroid i

        Returns:
            None
        """
        self.codes = self.codes.mul_(mu).add_(cluster_result, alpha=1 - mu)
    
class CodebookModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.codebook = CodebookForImage(config)
        
        self.image_feature_extractor = AutoFeatureExtractor.from_pretrained("resnet-101")
        self.image_encoder = ResNetModel.from_pretrained("resnet-101")
        self.image_config = self.image_encoder.config
        # self.image_config = ResNetConfig.from_pretrained("resnet-101")
        # self.image_encoder = ResNetModel(self.image_config)
        self.image_projection = nn.Linear(self.image_config.hidden_sizes[-1], config.codebook_embedding_size)
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_config = self.text_encoder.config
        self.text_projection = nn.Linear(self.text_config.hidden_size, config.codebook_embedding_size)

        self.integration = config.integration
        self.modal_integration = config.modal_integration
        self.auto_constraint = config.auto_constraint
        self.auto_add = config.auto_add
        
        if config.integration == "dot":
            self.fc = nn.Linear(config.codebook_embedding_size * 3, 1)
        else:
            self.fc = nn.Linear(config.codebook_embedding_size * 2, 1)
            if config.integration == "bilinear":
                matrix = torch.randn(size=(config.codebook_embedding_size, config.codebook_embedding_size))
                self.transformation_matrix = nn.Parameter(matrix, requires_grad=True)

        if config.auto_constraint:
            self.hyper_transfer = nn.Linear(config.codebook_embedding_size, config.codebook_embedding_size + 1)
            self.proto_transfer = nn.Linear(config.codebook_embedding_size, config.codebook_embedding_size + 1)
        else:
            self.hyper_transfer = nn.Linear(config.codebook_embedding_size, config.codebook_embedding_size)
            self.proto_transfer = nn.Linear(config.codebook_embedding_size, config.codebook_embedding_size)

        if self.modal_integration == "concat":
            self.hyper_integrate = nn.Linear(config.codebook_embedding_size * 2, config.codebook_embedding_size)
            self.hypo_integrate = nn.Linear(config.codebook_embedding_size * 2, config.codebook_embedding_size)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.text_contrast_loss = InfoNCE()
        self.constraint_contrast_loss = InfoNCE(temperature=0.3)
        self.classifier_loss = nn.BCELoss()

    def transform_add(self, x, y):
        a = torch.matmul(torch.matmul(x, self.transformation_matrix), y.T)
        a = F.softmax(a / x.size(1), dim=1)
        new_x = x + torch.matmul(a.T, y)
        new_y = y + torch.matmul(a, x)
        return new_x, new_y

    def constraint_control(self, hyper, proto):
        text_hypernym_tran = self.hyper_transfer(hyper)
        proto_tran = self.proto_transfer(proto)
        if self.auto_constraint:
            text_tau = self.sigmoid(text_hypernym_tran[:, -1])
            text_hypernym_tran = text_tau.view(-1, 1).expand_as(text_hypernym_tran[:, 0: -1]) * text_hypernym_tran[:, 0: -1]
            proto_tau = self.sigmoid(proto_tran[:, -1])
            proto_tran = proto_tau.view(-1, 1).expand_as(proto_tran[:, 0: -1]) * proto_tran[:, 0: -1]
        _, constraint_loss = self.constraint_contrast_loss(text_hypernym_tran, proto_tran)
        return text_hypernym_tran, proto_tran, constraint_loss

    def multitask(self, hypernym, hyponym, pixel_values, negatives, targets):
        # cluster loss
        image_tensor = self.image_encoder(pixel_values).pooler_output.squeeze()
        image_tensor_reduced = F.normalize(self.image_projection(image_tensor))
        indexes, codes, cluster_loss = self.codebook.train_forward(image_tensor_reduced)
        
        # text loss
        text_hypernym = self.text_encoder(hypernym[0], hypernym[1]).last_hidden_state
        text_hypernym = torch.mean(text_hypernym, dim=1)
        text_hypernym = self.text_projection(text_hypernym)
        text_hyponym = self.text_encoder(hyponym[0], hyponym[1]).last_hidden_state
        text_hyponym = torch.mean(text_hyponym, dim=1)
        text_hyponym = self.text_projection(text_hyponym)
        _, text_loss = self.text_contrast_loss(text_hypernym, text_hyponym)

        # constraint loss
        text_hypernym_tran, proto_tran, constraint_loss = self.constraint_control(text_hypernym, codes)

        # hyper and hypo representation
        text_negative = self.text_encoder(negatives[0], negatives[1]).last_hidden_state
        text_negative = torch.mean(text_negative, dim=1)
        text_negative = self.text_projection(text_negative)
        if self.auto_add:
            text_negative_tran = self.hyper_transfer(text_negative)
            if self.auto_constraint:
                text_negative_tau = self.sigmoid(text_negative_tran[:, -1])
                text_negative_tran = text_negative_tau.view(-1, 1).expand_as(text_negative_tran[:, 0: -1]) * text_negative_tran[:, 0: -1]

        if self.modal_integration == "add":
            if self.auto_add:
                alpha_hyper = self.sigmoid(nn.CosineSimilarity()(text_hypernym_tran, proto_tran))
                alpha_neg = self.sigmoid(nn.CosineSimilarity()(text_negative_tran, proto_tran))
                hyper = alpha_hyper.view(-1, 1).expand_as(text_hypernym) * text_hypernym + (1 - alpha_hyper).view(-1, 1).expand_as(codes) * codes
                hypo = alpha_hyper.view(-1, 1).expand_as(text_hyponym) * text_hyponym + (1 - alpha_hyper).view(-1, 1).expand_as(image_tensor_reduced) * image_tensor_reduced
                neg = alpha_neg.view(-1, 1).expand_as(text_negative) * text_negative + (1 - alpha_neg).view(-1, 1).expand_as(codes) * codes
                hypo_neg = alpha_neg.view(-1, 1).expand_as(text_hyponym) * text_hyponym + (1 - alpha_neg).view(-1, 1).expand_as(image_tensor_reduced) * image_tensor_reduced

                hyper_classification = torch.cat([hyper, neg], dim=0)
                hypo_classification = torch.cat([hypo, hypo_neg], dim=0)
            else:
                hyper = text_hypernym + codes
                hypo = text_hyponym + image_tensor_reduced
                neg = text_negative + codes

                hyper_classification = torch.cat([hyper, neg], dim=0)
                hypo_classification = torch.cat([hypo, hypo], dim=0)

        else:
            hyper = self.hyper_integrate(torch.cat([text_hypernym, codes], dim=1))            
            hypo = self.hypo_integrate(torch.cat([text_hyponym, image_tensor_reduced], dim=1))  
            neg = self.hyper_integrate(torch.cat([text_negative, codes], dim=1))

            hyper_classification = torch.cat([hyper, neg], dim=0)
            hypo_classification = torch.cat([hypo, hypo], dim=0)

        if self.integration == "dot":
            dot_classification = hyper_classification * hypo_classification
            concat_feature = torch.cat([hyper_classification, hypo_classification, dot_classification], dim=1)
        elif self.integration == "bilinear":
            hyper_classification_new, hypo_classification_new = self.transform_add(hyper_classification, hypo_classification)
            concat_feature = torch.cat([hyper_classification_new, hypo_classification_new], dim=1)
        else:
            concat_feature = torch.cat([hyper_classification, hypo_classification], dim=1)
            
        logits = self.fc(concat_feature)
        logits = self.sigmoid(logits)
        classifier_loss = self.classifier_loss(logits.squeeze(), targets.squeeze())

        loss = text_loss + cluster_loss + constraint_loss + classifier_loss

        return indexes, image_tensor_reduced, logits, (text_loss, cluster_loss, constraint_loss, classifier_loss, loss)

    def inference_termwise(self, hypernym, hyponym, pixel_values, targets):
        # cluster loss
        image_tensor = self.image_encoder(pixel_values).pooler_output.squeeze()
        image_tensor_reduced = F.normalize(self.image_projection(image_tensor))
        indexes, codes, cluster_loss = self.codebook.train_forward(image_tensor_reduced)
        
        # text loss
        text_hypernym = self.text_encoder(hypernym[0], hypernym[1]).last_hidden_state
        text_hypernym = torch.mean(text_hypernym, dim=1)
        text_hypernym = self.text_projection(text_hypernym)
        text_hyponym = self.text_encoder(hyponym[0], hyponym[1]).last_hidden_state
        text_hyponym = torch.mean(text_hyponym, dim=1)
        text_hyponym = self.text_projection(text_hyponym)
        _, text_loss = self.text_contrast_loss(text_hypernym, text_hyponym)

        # constraint loss
        text_hypernym_tran, proto_tran, constraint_loss = self.constraint_control(text_hypernym, codes)

        # hyper and hypo representation
        if self.modal_integration == "add":
            if self.auto_add:
                alpha = self.sigmoid(nn.CosineSimilarity()(text_hypernym_tran, proto_tran))
                hyper = alpha.view(-1, 1).expand_as(text_hypernym) * text_hypernym + (1 - alpha).view(-1, 1).expand_as(codes) * codes
                hypo = alpha.view(-1, 1).expand_as(text_hyponym) * text_hyponym + (1 - alpha).view(-1, 1).expand_as(image_tensor_reduced) * image_tensor_reduced
            else:
                # hyper and hypo representation by adding
                hyper = text_hypernym + codes
                hypo = text_hyponym + image_tensor_reduced
        else:
            hyper = self.hyper_integrate(torch.cat([text_hypernym, codes], dim=1))            
            hypo = self.hypo_integrate(torch.cat([text_hyponym, image_tensor_reduced], dim=1))

        # enhance representation
        if self.integration == "dot":
            dot_classification = hyper * hypo
            concat_feature = torch.cat([hyper, hypo, dot_classification], dim=1)
        elif self.integration == "bilinear":
            hyper_new, hypo_new = self.transform_add(hyper, hypo)
            concat_feature = torch.cat([hyper_new, hypo_new], dim=1)
        else:
            concat_feature = torch.cat([hyper, hypo], dim=1)
        
        logits = self.fc(concat_feature)
        logits = self.sigmoid(logits)

        classifier_loss = self.classifier_loss(logits.squeeze(), targets.squeeze())

        loss = text_loss + cluster_loss + constraint_loss + classifier_loss

        return logits, loss