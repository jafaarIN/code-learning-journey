// I'm following the USACO Guide, this marks the official beginning of my preparation for the IOI
// I've studied binary searches, learnt how they work and why they're effective
// This details how to find x within an abstract monotonic function such that f(x) is true

#include <bits/stdc++.h>
using namespace std;

int last_true(int lo, int hi, function<bool(int)> f){
  // 'lo' and 'hi' are values such that [lo, hi] is the space where x resides
  lo--; // if none of the values in the range contains no x such that f(x) returns true, then we return lo-1
  while (lo < hi) {
    // find the middle of the range (rounding up)
    int mid = lo + (hi - lo + 1) / 2;
    // You may feel inclined to use the formula of (lo+hi)/2 floored, however, this isn't good practice as lo + hi could result in an overflow error
    if (f(mid)) {
      // if mid works, then all numbers smaller than mid also work
      lo = mid;
    } else {
      // if mid does not work, greater values would not work either (as the function is monotonic, if f(x) is false, it remains false)
      hi = mid - 1
    }
  }
  return lo;
}

int main() {
  // all numbers that satisfy the condition (outputs 10)
  cout << last_true(2, 10, [](int x) { return true; }) << endl

	// outputs 5
	cout << last_true(2, 10, [](int x) { return x * x <= 30; }) << endl;

	// no numbers satisfy the condition (outputs 1)
	cout << last_true(2, 10, [](int x) { return false; }) << endl;
}
// Binary searches have a maximum of 30 iterations Olog(N)
