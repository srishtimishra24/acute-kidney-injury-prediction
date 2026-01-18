FROM ubuntu:noble

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -yq install python3-pip python3-venv

WORKDIR /model

# Copy model first so any change invalidates cache
COPY model.py /model/
COPY requirements.txt /model/

# Create virtual environment and install dependencies
RUN python3 -m venv /model
RUN /model/bin/pip3 install -r /model/requirements.txt

ENTRYPOINT ["/model/bin/python3", "/model/model.py"]
