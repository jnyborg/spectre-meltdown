# spectre-meltdown
Experiments with the Meltdown and Spectre exploits.

## Test the Spectre exploit
```
docker build -t spectre .
docker run -it --name spectre spectre
cd /spectre
./spectre.o
```
