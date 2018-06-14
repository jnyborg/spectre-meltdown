all:
	gcc -o spectre.o -std=c99 spectre.c
	gcc -o spectre35.o -std=c99 spectre35.c
	gcc -o spectre150.o -std=c99 spectre150.c
