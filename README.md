# Tensorial Mixture Models
This repository contains the scripts used to reproduce the experiments from the article [Tensorial Mixture Models](http://drop.sharir.org/articles/tmm/tmm_arxiv_v1.pdf). At the moment some of the experiments are missing, but we will add them in the coming weeks. If you are using this code in your article, or want to reference our results, please cite our article:

```
@article{sharir2016TMM,
  Author = {Sharir, Or and Tamari, Ronen and Cohen, Nadav and Shashua, Amnon},
  Journal = {arXiv preprint arXiv:1610.xxxx}, % Waiting for final arXiv identifier.
  Title = {Tensorial Mixture Models},
  Year = {2016}
}
```

## Disclaimer

Even though this code had been thoroughly tested on our systems, it might not perform exactly the same on other systems. Additionally, though we have made every effort to provide a robust code (e.g. using fixed seeds), which should reproduce the exact same numbers as presented in the article, even slight variations in the computing enviornment could lead to meaningful differences in the computed results. If the generated results are not sufficient, we recommend another pass of cross-validation to fine-tune the parameters to your exact setting. You will find the use of our hyper_train.py script very usefull for this task.

## Installation
First, note that this repo uses submodules, and some of them contain their own submodules. To simplify installation, please clone the repository with the `--recursive` flag:
```
git clone --recursive https://github.com/HUJI-Deep/TMM.git
```

After you have clones the repository, be sure to install the python modules listed in requirements.txt:
```
pip install -r requirements.txt
```

Finally, please review the installation instructions of each of the submodule's repositories under the `deps` directory. If you just wish to run the TMM models, then it's enough to follow the installation instructions of `deps/simnets`, our [Caffe fork for the SimNets Architecture](https://github.com/HUJI-Deep/caffe-simnets).

## Experiments
Our experiments on MNIST are under `exp/mnist`, which contain separate directories for each model. Each model directory contains three sub-directories: train, test_iid and test_rects. The `train` directory contains all the scripts to train the given model on the dataset. The `test_iid` and `test_rects` use the previously trained model to test it on either the iid or missing rectangles missingness distributions. Use the `run.py` script present in each subdirectory to either train the model or test it. After all the tests are done, you can use the `summary.py` script to pretty print all the results. Please do not move the scripts, as they all assume they are stored in their original location relative to the root of the repository. Additionally, notice that some experiments support multiple GPUs, and might require user-input to use all of them appropriately (e.g. see the train_plan.json files under the train directories of cp_model and ht_model).

Please be advised that both training and testing might take a while to complete. If you wish to skip training the models yourself and just want to run the test, you can use [one of the pretrained models](https://github.com/HUJI-Deep/TMM/wiki/Pre-trained-Models) we have prepared for you. Simply place the extracted files in their respective `train` directory.
