#include <stdio.h>

int main() {
    int age;
    float height;
    char name[50];  // a string of characters (up to 49 + null terminator)

    printf("Enter your name: ");
    scanf("%49s", name);  // read string safely (no spaces)

    printf("Enter your age: ");
    scanf("%d", &age);

    printf("Enter your height in meters: ");
    scanf("%f", &height);

    printf("\nSummary:\n");
    printf("Name: %s\n", name);
    printf("Age: %d\n", age);
    printf("Height: %.2f meters\n", height);
    printf("In 5 years, you'll be %d.\n", age + 5);

    return 0;
}
