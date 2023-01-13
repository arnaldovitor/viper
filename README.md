# 🐍 ViPeR: Video Pattern Recognition

ViPeR was designed to facilitate the training and evaluation of TensorFlow models for classifying videos/images of violence from security cameras, however, it can be used for a general purpose. The development was done during my bachelor's degree in Computer Science and paused during it. I intend to refactor all the code and include new features but the legacy version already does a lot if you want to use it.

# Legacy/Stable version

All the source code is in the `/legacy` folder and it works fine (as far as I've tested it), you can find some usage examples in `/legacy/demo.ipynb`. For installation it is highly recommended to use [Anaconda](https://www.anaconda.com/) and run the following commands:

1) `conda create --name viper python=3.7`
2) `conda activate viper`
3) `conda install tensorflow-gpu==2.2.0`
4) `conda install opencv pandas matplotlib`
5) `pip install gdown`

# Next steps

- [ ] Refactoring to be easier to use, similar to other frameworks like [Detectron2](https://github.com/facebookresearch/detectron2)
- [ ] Integration with linting and testing tools
- [ ] Integration with MLFlow
- [ ] Integration with new models
- [ ] Improve annotation tool
- [ ] Add feature extraction module

