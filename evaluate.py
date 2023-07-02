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

if __name__ == "__main__":
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

    # test dataset
    dataset = MMQuery(config.test_datapath, False)

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
        test_dataset=dataset,
        collate_fn=collate_fn,
        preprocessor=preprocessor,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
    )

    accu, precision, recall, f1 = trainer.test()
    print(f"accuracy: {accu}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")