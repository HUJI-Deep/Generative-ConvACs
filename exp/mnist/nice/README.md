# Recreating NICE Inpainted Missing Data Experiment
This experiment checks the missing data classification scheme of inpainting the missing data with the NICE model, and classifying the resulting images using a CNN classifier. Both the NICE model and CNN are trained on clean datasets.

To recreate this experiment, follow these steps:

1. Train NICE Model on MNIST as detailed [here](https://github.com/HUJI-Deep/nice).

2. Run `exp/mnist/nice/inpaint/run.py` to inpaint the missing data datasets (rects and iid noise). This is a computation heavy procedure, you can run the script from multiple machines to speed up the process.

3. Train a CNN classifier for MNIST by running `exp/mnist/lenet/train/run.py` (trained on clean dataset).

4. When all inpainted datasets have been generated, run `exp/mnist/nice/test_rects/run.py`, `exp/mnist/nice/test_iid/run.py` to perform the classification task using the pre-trained CNN model, and `exp/mnist/nice/test_XXX/summarize.py` to obtain a summary of the results.