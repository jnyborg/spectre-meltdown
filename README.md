# spectre-meltdown
Experiments with the Meltdown and Spectre exploits.

## Test the Spectre exploit
```
docker build -t spectre .
docker run -it --name spectre spectre
cd /spectre
./spectre.o
```
# TODO
- Implement Spectre in another language, compare high level vs low level?
- Investigate mitigation techniques. Compare pre/post mitigation performance of some programs?
- Increase accuracy of reads using ML? Some reads will always be faster than others, i.e. 0 and 1 
will be faster than 242. 
