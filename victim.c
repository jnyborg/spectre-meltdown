/* Simple long running process with a secret variable that we will try to extract with spectre */
#include <stdio.h>

char *secret = "correcthorsebatterystaple";

int main(int argc, const char **argv)
{
    printf("Secret address: %p", &secret);
    printf("Press ENTER to stop program");
    getchar();
}