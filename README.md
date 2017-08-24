# tensorflow Memo

### install

https://www.tensorflow.org/install/install_mac


e.g.

install directory -> tensorflow:
```
sudo easy_install pip3
sudo pip3 install --upgrade virtualenv
virtualenv --system-site-packages tensorflow
source tensorflow/bin/activate
pip3 install --upgrade tensorflow
```

### docker

```
brew cask install docker
open /Applications/Docker.app
```

https://hub.docker.com/r/tensorflow/tensorflow/
```
docker run -it -p 8888:8888 tensorflow/tensorflow

docker run -v ~/Products/github/tensorflow:/tensorflow_dev -it tensorflow/tensorflow
docker ps
docker exec -it [container id] python /tensorflow_dev/script/helloworld.py
```
