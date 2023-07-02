from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser()
    
    # data path
    parser.add_argument("--train_datapath", default="data/train.json")
    parser.add_argument("--dev_datapath", default="data/dev.json")
    parser.add_argument("--test_datapath", default="data/test.json")
    parser.add_argument("--save_dir")
    parser.add_argument("--resume_path")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--model_type", choices=["codebook", "base"])

    # hyper parameters
    # training
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--lr_step_size", type=int, default=5)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=int, default=0.0005)
    parser.add_argument("--mu", type=float, default=0.99)
    # inference
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", default=0.5)
    # model
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num_codes", type=int, default=4096)
    parser.add_argument("--codebook_embedding_size", type=int, default=256)
    parser.add_argument("--use_codes", action="store_true")
    parser.add_argument("--auto_constraint", action="store_true")
    parser.add_argument("--auto_add", action="store_true")
    parser.add_argument("--integration", choices=["concat", "dot", "bilinear"])
    parser.add_argument("--modal_integration", choices=["add", "concat"], default="add")

    # train settings
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--no_train", action="store_true", default=False)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", choices=["warmup", "normal", "multitask"])
    parser.add_argument("--save_all_checkpoints", action="store_true", default=False)
    
    return parser