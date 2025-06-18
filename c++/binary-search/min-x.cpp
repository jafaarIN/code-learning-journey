// The first_true function works very similair to the last_true function in the 'max-x' bit of code
// However, here, we're constructing the function firstTrue such that firstTrue(lo, hi, f) returns the first x in the range [lo, hi] such that f(x) = true
// if no such x exists, it should return hi + 1

#include <bits/stdc++.h>
using namespace std;

int first_true(int lo, int hi, function<bool(int)> f) {
	hi++;
	while (lo < hi) {
		int mid = lo + (hi - lo) / 2;
		if (f(mid)) { 
			hi = mid; // if mid satisfies the condition of f, then we want nothing beyond that point 
		} else {
			lo = mid + 1;
		}
	}
	return lo;
}

int main() {
	// outputs 2
	cout << first_true(2, 10, [](int x) { return true; }) << endl;
	// outputs 6
	cout << first_true(2, 10, [](int x) { return x * x >= 30; }) << endl;
	// outputs 11
	cout << first_true(2, 10, [](int x) { return false; }) << endl;
}
