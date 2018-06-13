FROM python:3.6.5

ADD learner/requirements.txt /app/requirements.txt


RUN pip install -r /app/requirements.txt

ADD . /app
WORKDIR /app/learner

RUN python setup.py install

ENTRYPOINT ["/bin/bash"]