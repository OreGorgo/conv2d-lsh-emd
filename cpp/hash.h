#ifndef HASH_H
#define HASH_H

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

class hashtable{

protected:
    vector <list <image> > *table;

    int buckets;
    int w;
    int k;
    int dimension;
    int M;

    vector <double>* S_arr;

public:
    hashtable(int buckets,int dimension, int k, int w,int M, vector <double>* S_arr);
    ~hashtable();
    
    unsigned int compute_h(image input, int i);
    
    virtual unsigned int hash(image input);
    void insert_image(image input);
    
    list <image>* get_bucket(unsigned int g);
    unsigned int get_bucket_no();
    
};


class hypercube_HT: public hashtable
{
	int d;
	unordered_map <unsigned int, char> memory;
	
	public:
		hypercube_HT(int d,int dimension, int k, int w,int M, vector <double>* S_arr);
		~hypercube_HT();
		
		unsigned int hash(image input) override;
		int get_d();
	
};

#endif
