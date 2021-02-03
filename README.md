# Single-image-dehazing-using-improved-cycleGAN
This is the code for the journal paper `Single image dehazing using improved cycleGAN` published in <a href="https://www.journals.elsevier.com/journal-of-visual-communication-and-image-representation">JVCI</a>.



## Acknowledgements
The base code of cycleGAN's implementation in tensorflow has been taken from this <a href="https://github.com/xhujoy/CycleGAN-tensorflow">Repo</a> by <a href="https://github.com/xhujoy">xhujoy</a>. 😊

## Dataset
- Place the <a href="https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html">NYU</a> or <a href="https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2">reside-β</a> dataset into `dataset` directory.
- Split the hazy and clear images into train, test sets.
- Rename the directory name containing the training images of hazy, clear images as `trainA`, `trainB` respectively.
- Do the same for the corresponding directories containing the test set images.

The final file structure should be as follows
```
project
│   README.md
│   get_ssim.py
│   main.py
│   model.py
│   module.py
│   ops.py
│   requirements.txt
│   utils.py
│
└───dataset
    └───<dataset_name>
        └───trainA
        |   | img1.jpg
        |   | img2.jpg
        |   | ...
        |
        └───trainB
        |   | img1.jpg
        |   | img2.jpg
        |   | ...
        |
        └───testA
        |   | img1.jpg
        |   | img2.jpg
        |   | ...
        |
        └───testB
            | img1.jpg
            | img2.jpg
            | ...
```

The rest of the directories required to store the saved model, sample results and log files will be created automatically when `main.py` is executed.
