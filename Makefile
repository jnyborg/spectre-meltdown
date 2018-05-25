all:
	gcc -o spectre.o -std=c99 spectre.c
	gcc -o victim.o -std=c99 victim.c
