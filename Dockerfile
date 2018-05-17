FROM ubuntu:trusty-20170620

ADD . /spectre

RUN apt-get update && apt-get install -y gcc vim curl make

ENTRYPOINT ["/bin/bash"]
