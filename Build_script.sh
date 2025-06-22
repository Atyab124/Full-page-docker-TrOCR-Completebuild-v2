sudo docker build -t trocr .
sudo docker run -p 8080:8080 -it --gpus all trocr
