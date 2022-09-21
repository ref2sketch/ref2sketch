#### Title
[Reference based sketch extraction via attention mechanism]



        
## Train
    $ python main.py --mode train_deep \
                     --scope ref2sketch \
                     --name_data examples \
                     --dir_data ./datasets \
                     --dir_log ./log \
                     --dir_checkpoint ./checkpoint \
                     --direction A2B
                     --gpu_ids 0

* Set **[scope name]** uniquely.
* For settings from original paper, use 'train' for --mode. However we recommend use 'train_deep' for producing better quality of model with deeper convolution layers.
* To understand hierarchy of directories based on their arguments, see **directories structure** below. 


## Test
    $ python test.py --name_weight test \
                     --name_data examples \
                     --direction A2B \
                     --cuda



