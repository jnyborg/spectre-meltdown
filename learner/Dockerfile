FROM python:3.6.5

ADD . /app
ADD ./pyspectre.c /app/pyspectre.c

WORKDIR /app

RUN apt-get install -y gcc && pip install -r requirements.txt &&  python setup.py install

ENTRYPOINT ["/bin/bash"]