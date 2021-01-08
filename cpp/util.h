#ifndef UTIL_H
#define UTIL_H

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

using namespace std;

typedef struct image{
	int id;
	int cluster;
	int old_cluster;
	int g;
	vector<unsigned int> pixels;
}image;

typedef struct{
	image p;
	int distance;
	
}neighbor;

bool my_cmp(const neighbor & a, const neighbor & b);
bool cmp(const neighbor & a, const neighbor & b);

unsigned int l1_distance(image x, image y);

int modular_pow(int base, int exp, int modulo);
int average_NN(image *arr, int sample);
neighbor brute_force_NN(image *arr, image query,int size);


vector <double>* compute_s(int w,int k,int size,uniform_real_distribution<double> distribution, default_random_engine* generator);
void hamming(string str, int i, int changesLeft, vector<unsigned int>* vertices, const unsigned int limit);

//
//static std::vector<char> ReadAllBytes(string filename);

#endif
