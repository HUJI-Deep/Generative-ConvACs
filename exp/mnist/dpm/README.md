# Recreating DPM Inpainted Missing Data Experiment
This experiment checks the missing data classification scheme of inpainting the missing data with a Diffusion Probabilistic Model, and classifying the resulting images using a CNN classifier. Both the DPM and CNN are trained on clean datasets.

To recreate this experiment, follow these steps:

1. Train Diffusion Probabilistic Model on MNIST as detailed [here](https://github.com/HUJI-Deep/Diffusion-Probabilistic-Models).

2. Run `exp/mnist/dpm/inpaint/run.py` to inpaint the missing data datasets (rects and iid noise). This is a computation heavy procedure, you can run the script from multiple machines to speed up the process.

3. Train a CNN classifier for MNIST by running `exp/mnist/lenet/train/run.py` (trained on clean dataset).

4. When all inpainted datasets have been generated, run `test_rects/run.py`, `test_iid/run.py` to perform the classification task using the pre-trained CNN model, and `test_XXX/summarize.py` to obtain a summary of the results.
