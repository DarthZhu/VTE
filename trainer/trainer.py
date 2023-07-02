import os
import math
import json
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from torch.optim.lr_scheduler import StepLR

from utils.get_gpu_id import get_freer_gpu
from utils.preprocessor import Preprocessor

class BaseTrainer():
    def __init__(self,
                 config = None,
                 model: nn.Module = None,
                 train_dataset: Dataset = None,
                 dev_dataset: Dataset = None,
                 test_dataset: Dataset = None,
                 collate_fn = None,
                 preprocessor = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: Optional[StepLR] = None,
                 optimizer_grouped_parameters: Optional[List[Dict]] = None,
                 logger: logging.Logger = None):
        # save config
        self.config = config
        if self.config.use_gpu:
            self.device = torch.device(f"cuda:{get_freer_gpu()}")
        else:
            self.device = torch.device("cpu")
        self.logger = logger
        if self.config.save_dir:
            if not os.path.exists(self.config.save_dir):
                os.mkdir(self.config.save_dir)
        
        # prepare data loader and preprocessor
        if train_dataset and dev_dataset is not None:
            self.train_dataset = train_dataset
            self.dev_dataset = dev_dataset
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            self.dev_dataloader = DataLoader(
                dataset=self.dev_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
        if test_dataset is not None:
            self.test_dataset = test_dataset
            self.test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
        
        # prepare model, preprocessor, optimizer and scheduler
        self.model = model(self.config)
        self.preprocessor = preprocessor(self.model.tokenizer, self.model.image_feature_extractor, self.config, self.device)
        
        if optimizer_grouped_parameters is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        self.optimizer = optimizer(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )
        if config.resume_path is None:
            self.logger.warning("No checkpoint given!")
            self.best_dev_loss = 10000
            self.best_f1 = 0
            self.last_epoch = -1
            self.scheduler = StepLR(
                optimizer=self.optimizer,
                step_size=config.lr_step_size,
                gamma=self.config.lr_gamma,
                last_epoch=self.last_epoch,
            )
        else:
            self.logger.info(f"Loading model from checkpoint: {self.config.resume_path}.")
            checkpoint = torch.load(self.config.resume_path, map_location=self.device)
            self.model.to(self.device)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.best_dev_loss = checkpoint["best_val_loss"]
            self.best_f1 = 0
            self.last_epoch = -1
            # self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler = StepLR(
                optimizer=self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
                last_epoch=self.last_epoch
            )
            # self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.model.to(self.device)
        
        self.logger.info("Trainer init done.") 
        
    def train(self):
        raise NotImplementedError
    
    def eval(self, test_dataloader=None):
        raise NotImplementedError

class CodebookTrainer(BaseTrainer):
    def __init__(self,
                 config = None,
                 model: nn.Module = None,
                 train_dataset: Dataset = None,
                 dev_dataset: Dataset = None,
                 test_dataset: Dataset = None,
                 collate_fn = None,
                 preprocessor = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: Optional[StepLR] = None,
                 optimizer_grouped_parameters: Optional[List[Dict]] = None,
                 logger: logging.Logger = None):
        super().__init__(config, model, train_dataset, dev_dataset, test_dataset, collate_fn, preprocessor, optimizer, scheduler, optimizer_grouped_parameters, logger)
        
    def _cluster_result(self, indexes, encoder_tensors):
        cluster_result_add = torch.zeros_like(self.model.codebook.codes).scatter_add_(0, indexes.unsqueeze(-1).expand_as(encoder_tensors), encoder_tensors)
        cnt_indexes = torch.zeros((cluster_result_add.size(0))).to(self.device).scatter_add_(0, indexes, torch.ones_like(indexes).float().to(self.device)) + 0.001
        
        cluster_result = torch.div(cluster_result_add.t(), cnt_indexes).t()
        
        return cluster_result
    
    def train(self):
        for epoch in range(self.config.train_epochs):
            self.logger.info("--------------------------")
            self.logger.info(f"epoch {epoch} starts...")
            self.model.train()
            num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
            progress_bar = tqdm(range(num_update_steps_per_epoch))
            total_loss = 0
            if self.config.mode is not None:
                mode = self.config.mode
            else:
                if epoch < self.config.warmup_epochs:
                    mode = "warmup"
                else:
                    mode = "normal"
            self.logger.info(f"Current training mode: {mode}")    
            
            for step, batch in enumerate(self.train_dataloader):
                hypers, hypos, images, negatives = batch
                targets = torch.tensor([1] * len(batch[0]) + [0] * len(batch[0])).float().to(self.device)
                hyper_input_ids, hyper_attention_mask = self.preprocessor.preprocess_text(hypers)
                hypo_input_ids, hypo_attention_mask = self.preprocessor.preprocess_text(hypos)
                neg_input_ids, neg_attention_mask = self.preprocessor.preprocess_text(negatives)
                pixel_values = self.preprocessor.preprocess_image(images)
                indexes, image_tensor_reduced, logits, (text_loss, cluster_loss, constraint_loss, classifier_loss, loss) = self.model.multitask((hyper_input_ids, hyper_attention_mask), (hypo_input_ids, hypo_attention_mask), pixel_values, (neg_input_ids, neg_attention_mask), targets)
                loss = loss / self.config.gradient_accumulation_steps
                total_loss += loss.detach().item()

                loss.backward()
                # momentum update 
                cluster_result = self._cluster_result(indexes.detach(), image_tensor_reduced.detach())
                self.model.codebook._momentum_update(self.config.mu, cluster_result)

                if step % self.config.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    progress_bar.set_postfix(
                        {
                            'loss': loss.item(),
                            'text_loss': text_loss.detach().item(),
                            'cluster_loss': cluster_loss.detach().item(),
                            'constraint_loss': constraint_loss.detach().item(),
                            'index_variance': torch.var((indexes.detach().float()-2048)/2048).item(),
                            'classifier_loss': classifier_loss.detach().item(),
                        }
                    )
                    progress_bar.update(1)
            
            # after one epoch step scheduler
            self.scheduler.step()

            # save checkpoint every epoch
            self.logger.info(f"Epoch {epoch} training loss: {total_loss / (step + 1)}")
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "last_epoch": epoch + self.last_epoch + 1,
                "best_val_loss": total_loss / (step + 1),
                "config": self.config
            }
            save_path = os.path.join(self.config.save_dir, f"epoch_{self.last_epoch + epoch + 1}.pt")
            self.logger.info(f"Saving model to {save_path}")
            torch.save(checkpoint, save_path)
                    
            # eval
            self.model.eval()
            if mode == "warmup":
                dev_loss = self.eval_warmup(self.dev_dataloader)
                self.logger.info(f"Epoch {epoch} dev loss: {dev_loss}")
                if dev_loss < self.best_dev_loss:
                    self.best_dev_loss = dev_loss
                    checkpoint = {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "last_epoch": epoch + self.last_epoch + 1,
                        "best_val_loss": total_loss / (step + 1),
                        "config": self.config
                    }
                    save_path = os.path.join(self.config.save_dir, "best.pt")
                    self.logger.info(f"Saving best model to: {save_path}")
                    torch.save(checkpoint, save_path)
                    
            elif mode == "normal" or mode == "multitask":
                dev_loss, preds, truth = self.eval(self.dev_dataloader)
                accu, precision, recall, f1 = self.metric(preds, truth)

                self.logger.info(f"Epoch {epoch} dev loss: {dev_loss}")
                self.logger.info(f"Epoch {epoch} dev accuracy: {accu}")
                self.logger.info(f"Epoch {epoch} dev precision: {precision}")
                self.logger.info(f"Epoch {epoch} dev recall: {recall}")
                self.logger.info(f"Epoch {epoch} dev f1: {f1}")
                
                if dev_loss < self.best_dev_loss:
                    self.best_dev_loss = dev_loss
                    checkpoint = {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "last_epoch": epoch + self.last_epoch + 1,
                        "best_val_loss": total_loss / (step + 1),
                        "config": self.config
                    }
                    save_path = os.path.join(self.config.save_dir, "best.pt")
                    self.logger.info(f"Saving best model to: {save_path}")
                    torch.save(checkpoint, save_path)
                
    @torch.no_grad()
    def eval_warmup(self, test_dataloader):
        total_dev_loss = 0
        for step, batch in enumerate(test_dataloader):
            hypers, hypos, images, _ = batch
            hyper_input_ids, hyper_attention_mask = self.preprocessor.preprocess_text(hypers)
            hypo_input_ids, hypo_attention_mask = self.preprocessor.preprocess_text(hypos)
            pixel_values = self.preprocessor.preprocess_image(images)
            
            _, _, loss = self.model.warmup(pixel_values, (hyper_input_ids, hyper_attention_mask), (hypo_input_ids, hypo_attention_mask))
            total_dev_loss += loss.detach().cpu()
        return total_dev_loss / (step + 1)
    
    @torch.no_grad()
    def eval(self, test_dataloader):
        total_dev_loss = 0
        preds = []
        truth = []
        for step, batch in enumerate(test_dataloader):
            hypers, hypos, images, targets = batch
            hyper_input_ids, hyper_attention_mask = self.preprocessor.preprocess_text(hypers)
            hypo_input_ids, hypo_attention_mask = self.preprocessor.preprocess_text(hypos)
            pixel_values = self.preprocessor.preprocess_image(images)
            targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
            
            logits, loss = self.model.inference_termwise((hyper_input_ids, hyper_attention_mask), (hypo_input_ids, hypo_attention_mask), pixel_values, targets)
            total_dev_loss += loss.detach().cpu()
            for id in range(logits.size()[0]):
                output = logits[id]
                if output > 0.5:
                    pred = 1
                else:
                    pred = 0
                preds.append(pred)
                truth.append(targets[id])
        return total_dev_loss / (step + 1), preds, truth

    @torch.no_grad()
    def metric(self, preds, targets):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for id, label in enumerate(targets):
            output = preds[id]
            if label == 1:
                if output == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if output == 0:
                    tn += 1
                else:
                    fp += 1
        accu = (tp + tn) / (tp + tn + fp + fn)
        try:
            precision = tp / (tp + fp)
        except:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except:
            recall = 0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except:
            f1 = 0
        return accu, precision, recall, f1

    @torch.no_grad()
    def test(self):
        write_data = []
        # _, preds, truth = self.eval(self.test_dataloader)
        preds = []
        truth = []
        for step, batch in tqdm(enumerate(self.test_dataloader)):
            hypers, hypos, images, targets = batch
            hyper_input_ids, hyper_attention_mask = self.preprocessor.preprocess_text(hypers)
            hypo_input_ids, hypo_attention_mask = self.preprocessor.preprocess_text(hypos)
            pixel_values = self.preprocessor.preprocess_image(images)
            targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
            
            logits, loss = self.model.inference_termwise((hyper_input_ids, hyper_attention_mask), (hypo_input_ids, hypo_attention_mask), pixel_values, targets)
            for id in range(logits.size()[0]):
                output = logits[id]
                if output > self.config.threshold:
                    pred = 1
                else:
                    pred = 0
                if self.config.output_path:
                    write_data.append({
                        "hyponym": hypers[id],
                        "hypernym": hypos[id],
                        "label": int(targets[id].item()),
                        "pred": pred,
                    })
                preds.append(pred)
                truth.append(targets[id])
        if self.config.output_path:
            with open(self.config.output_path, "w") as fout:
                json.dump(write_data, fout, indent=2, ensure_ascii=False)

        return self.metric(preds, truth)


    @torch.no_grad()
    def test_cosine(self):
        write_data = []
        # _, preds, truth = self.eval(self.test_dataloader)
        preds = []
        truth = []
        for step, batch in enumerate(self.test_dataloader):
            hypers, hypos, images, targets = batch
            hyper_input_ids, hyper_attention_mask = self.preprocessor.preprocess_text(hypers)
            hypo_input_ids, hypo_attention_mask = self.preprocessor.preprocess_text(hypos)
            pixel_values = self.preprocessor.preprocess_image(images)
            targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
            
            logits = self.model.inference_cosine((hyper_input_ids, hyper_attention_mask), (hypo_input_ids, hypo_attention_mask), pixel_values)
            for id in range(logits.size()[0]):
                output = logits[id]
                if output > 0:
                    pred = 1
                else:
                    pred = 0
                if self.config.output_path:
                    write_data.append({
                        "hyponym": hypers[id],
                        "hypernym": hypos[id],
                        "label": int(targets[id].item()),
                        "pred": pred,
                    })
                preds.append(pred)
                truth.append(targets[id])
        if self.config.output_path:
            with open(self.config.output_path, "w") as fout:
                json.dump(write_data, fout, indent=2, ensure_ascii=False)

        return self.metric(preds, truth)

    
    @torch.no_grad()
    def infer(self):
        write_data = []
        # _, preds, truth = self.eval(self.test_dataloader)
        preds = []
        truth = []
        for step, batch in tqdm(enumerate(self.test_dataloader)):
            hypers, hypos, images, targets = batch
            hyper_input_ids, hyper_attention_mask = self.preprocessor.preprocess_text(hypers)
            hypo_input_ids, hypo_attention_mask = self.preprocessor.preprocess_text(hypos)
            pixel_values = self.preprocessor.preprocess_image(images)
            targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
            
            logits, loss = self.model.inference_termwise((hyper_input_ids, hyper_attention_mask), (hypo_input_ids, hypo_attention_mask), pixel_values, targets)
            for id in range(logits.size()[0]):
                output = logits[id]
                if output > self.config.threshold:
                    pred = 1
                else:
                    pred = 0
                if self.config.output_path:
                    write_data.append({
                        "hypernym": hypers[id],
                        "hyponym": hypos[id],
                        "pred": pred,
                    })
                preds.append(pred)
                truth.append(targets[id])
        if self.config.output_path:
            with open(self.config.output_path, "w") as fout:
                json.dump(write_data, fout, indent=2, ensure_ascii=False)