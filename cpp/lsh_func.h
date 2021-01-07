#ifndef LSH_H
#define LSH_H


#include <iostream>
#include <cstdlib>
#include <string>
#include <list>
#include <vector>
#include <cstring>
#include <fstream>
#include <byteswap.h>
#include <climits>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <bits/stdc++.h> 
#include <bitset>
#include <chrono> 

#include "util.h"
#include "hash.h"

vector <neighbor> knn(hashtable** ht,image query, int k, int L);
vector <unsigned int> range_search(hashtable** ht,image query, double r, int L, int c);


#endif
