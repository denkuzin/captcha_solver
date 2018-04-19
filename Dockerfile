FROM ubuntu:16.04

# install python3.6
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install wget -y
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get install -y python3.6-dev
RUN apt-get install -y python3.6-venv
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py
RUN ln -s -f /usr/bin/python3.6 /usr/local/bin/python3
RUN ln -s -f /usr/local/bin/pip /usr/local/bin/pip3

# install requirements
RUN pip3 install pyvirtualdisplay

# install chrominium
RUN apt-get install -y chromium-browser

# Set the working directory to /app
WORKDIR /WebCrawler

# Copy the current directory contents into the container at /app
ADD . /WebCrawler

RUN apt-get install -y python3-tk

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install lxml

# Install PyTorch
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl 
RUN pip3 install torchvision

ENV PYTHONIOENCODING=utf8

# Run app.py when the container launches
RUN chmod a+x /WebCrawler/run.py
CMD ["python3", "run.py"]

