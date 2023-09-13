# VTE

This is the official repository for *Towards Visual Taxonomy Expansion*.

## Dataset

### Chinese Taxonomy Dataset

The constructed Chinese taxonomy dataset is in `./Chinese taxonomy dataset`.
Due to the large size of image files, we currently are unable to upload them.
But, we will upload them as soon as possible.

In the training file, every hypernymy pair is stored as a list, where the first item denotes the hyponym, while the second item denotes the hypernym.

In the test file, the first item denotes the hyponym, while the second item denotes a potential parent derived from user click logs.

### Semeval-2016 Dataset

The original dataset can be view [here](https://alt.qcri.org/semeval2016/task13/index.php?id=data-and-tools).
Same reason as the Chinese Taxonomy Dataset, image files are not available now.

## How to run

To reproduce our resutls reported in our paper, run:

```bash
python train.py \
    --train_datapath $TRAIN_DATAPATH \
    --dev_datapath $DEV_DATAPATH \
    --test_datapath $TEST_DATAPATH \
    --save_dir $YOUR_SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --modal_integration add \
    --auto_add \
    --integration dot \
    --train_epochs $EPOCHS    
```
