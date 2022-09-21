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

* For settings from original paper, use 'train' for --mode. However we recommend to use 'train_deep' for producing better quality of model with deeper convolution layers.
* To understand hierarchy of dataset, see **Dataset directories structure** below. 


## Test
    $ python test.py --name_weight test \
                     --name_data examples \
                     --direction A2B \
                     --cuda



## Dataset directories structure
    ref2sketch
    +---[dir_data]
    |   \---[name_data]
    |       +---test
    |       |   +---a
    |       |       |   +---test_input1.png
    |       |       |   +---test_input2.png
    |       |   +---b
    |       |       |   +---train_groundtruth1.png #not necessary for testing
    |       |       |   +---train_groundtruth2.png #not necessary for testing
    |       |   +---c
    |       |       |   +---style1.png
    |       |       |   +---style1.png
    |       +---train
    |       |   +---a
    |       |       |   +---train_input1.png
    |       |       |   +---train_input2.png
    |       |   +---b
    |       |       |   +---train_groundtruth1.png
    |       |       |   +---train_groundtruth2.png

---
