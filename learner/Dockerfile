FROM python:3.6.5

ADD requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

ADD . /app
RUN python setup.py install

ENTRYPOINT ["/bin/bash"]