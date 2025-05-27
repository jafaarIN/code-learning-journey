#include <iostream>
using namespace std;

int doubleIt(int x) {
    return x * 2;
}

int main() {
    int age;
    cout << "Enter your age: ";
    cin >> age;
    if (age >= 18) {
        cout << "You're an adult. \n";
    } else {
        cout << "You will be an adult in " << 18-age << " years. \n" ;
    }
    cout << "Twice your age is " << doubleIt(age) << ".";
    
    return 0;
}
