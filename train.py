import os
import sys
import time
import torch
import logging
from transformers import logging as t_logging

from utils.parser import get_parser
from utils.get_gpu_id import get_freer_gpu
from utils.preprocessor import Preprocessor
from utils.data_utils import MMQuery, MultitaskDataset, collate_fn
from trainer.trainer import CodebookTrainer
from models.modeling_codebook import CodebookModel

t_logging.set_verbosity_error()

if __name__ == "__main__":
    # config
    parser = get_parser()
    config = parser.parse_args()
    
    # logger
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_path = 'logs/'
    logging.basicConfig(
        filename=os.path.join(log_path, f'{timestr}.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger("train")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    # dataset
    train_datapath = os.path.join(config.train_datapath)
    train_dataset = MultitaskDataset(train_datapath, False)
    
    dev_datapath = os.path.join(config.dev_datapath)
    dev_dataset = MMQuery(dev_datapath, False)
    
    # preprocessor
    preprocessor = Preprocessor
    
    # model
    model = CodebookModel
    
    # optimizers
    optimizer = torch.optim.AdamW
    scheduler = torch.optim.lr_scheduler.StepLR
    
    # trainer
    trainer = CodebookTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        collate_fn=collate_fn,
        preprocessor=preprocessor,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
    )
    
    trainer.train()