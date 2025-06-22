# TrOCR multiline API on Docker image

This folder contains a Dockerfile and a Bash script to build and run a Docker image with GPU support for the Flask API of TrOCR multiline.
It includes a gradio demo as well which can connect to the API to give a Graphical User Interface

``Note:`` The compatibility of the entire project has been ensured with ``Python 3.9`` and might give wheel building errors on other versions

## Usage

- If you dont have a docker installation on your system, simply execute the ``Docker_install_script.sh`` bash script in terminal with command:
```
bash Docker_install_script.sh
```
- If your docker installation doesn't have nvidia-container-toolkit, execute the ``nvidia-container-toolikit.sh`` bash script in terminal with command:
```
bash nvidia-container-toolikit.sh
```
- If you already have docker with nvidia-container-toolkit, or have executed the above step, Execute the ``Build_script.sh`` with command:
```
bash Build_script.sh
```
to build and execute the trocr API docker image/container

- The API will go live on "localhost:8080/hand_OCR" once the container executes completely

- To run the gradio demo, simply install ``gradio_requirements.txt`` into a python environment with command:
```
pip install -r gradio_requirements.txt
```
and execute the gradio_demo.py file once the API goes live with the command:
```
python gradio_demo.py
```

``NOTE`` :- The Docker file runs the ``'master_api_demo.py'`` file by default, which works with the gradio demo. To execute the api that returns coordinates as well, open the ``Dockerfile`` and edit the line:
```
CMD ["python3", "master_api_demo.py"]
```
to
```
CMD ["python3", "master_api.py"]
```
