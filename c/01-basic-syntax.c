#include <stdio.h>

int main() {
    int age;
    printf("Enter your age: ");
    scanf("%d", &age); // %d is the format specifier for an integer, &age is the address of variable 'age' not the variable itself

    if (age >= 18) {
        printf("You're an adult. \n");
    } else {
        printf("You're an adult in %d years \n", 18-age); // found this quite confusing initially, it makes a lot of sense now though
    }

    int doubleAge = age * 2;
    printf("Twice your age is %d.\n", doubleAge);

    return 0;
}
