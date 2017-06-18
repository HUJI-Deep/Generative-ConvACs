## **SPN Missing Data Experiments on MNIST** ##
Since no existing implementation was available for structure learning and classification of missing data using SPNs, we used the Java code from [1] to learn the structure of the SPN and [2] (spn-opt) for learning the parameters and for classification with marginalization over missing data.

**Usage**
1. Make sure the `spn-poon-2011` and `spn-opt-discrim` projects are in the `deps` directory, and are compiled correctly according to the instructions in their respective repos.
2. Run the structure learning algorithm to learn the SPN structure following the instructions [here](https://github.com/HUJI-Deep/spn-poon-2011). 
3. Run `python exp/mnist/spn/train/run.py` to train a model on clean data for classification of missing data. This will probably take a long time (more than a day).
4. Run `python exp/mnist/spn/test/run.py` to test the trained model over missing data. At this stage, you can run this command on multiple machines for quicker results. 
5. Run `python exp/mnist/spn/test/summarize.py` to gather the performance over all missing data datasets.


 [[1]](http://www.cs.cmu.edu/~hzhao1/papers/ICML2016/spn_release.zip) Implementation of the methods described in:
- **A Unified Approach for Learning the Parameters of Sum-Product Networks**
by H. Zhao, P. Poupart and G. Gordon, NIPS 2016.
- **Collapsed Variational Inference for Sum-Product Networks**
by H. Zhao, T. Adel, G. Gordon and B. Amos, ICML 2016.
[[2]](http://spn.cs.washington.edu/spn/) - **Sum-Product Networks: A New Deep Architecture** by Poon, Domingos ,  UAI 2011

