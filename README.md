# tensorflow Memo

### install

https://www.tensorflow.org/install/install_mac


e.g.

install directory -> tensorflow:
```
sudo easy_install pip
sudo pip install --upgrade virtualenv
virtualenv --system-site-packages tensorflow
source tensorflow/bin/activate
pip install --upgrade tensorflow
```


### retrain

https://www.tensorflow.org/tutorials/image_retraining

```
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos --architecture mobilenet_0.25_128_quantize
```
