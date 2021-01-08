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

using namespace std;

bool my_cmp(const neighbor & a, const neighbor & b)
{
    // smallest comes first
    return a.distance < b.distance;
}

bool cmp(const neighbor & a, const neighbor & b)
{
    // smallest comes first
    return a.distance == b.distance;
}


unsigned int l1_distance(image x, image y)
{
    int size = x.pixels.size();
    unsigned int sum=0;
    
    for(int i=0;i<size;i++)
    {
        sum += abs((int) (x.pixels[i] - y.pixels[i]) );
    }

	return sum;
}


int modular_pow(int base, int exp, int modulo)
{
	int result = 1;
    base = base%modulo;
    
	while(exp > 0)
    {
	    if (exp%2 == 1)
    		result = (result * base) % modulo;
        exp = exp >> 1;
        base = (base * base) % modulo;
	}
	
	return result;
}


int average_NN(image *arr, int sample)
{
    int min,sum=0,dist;

    for(int i=0;i<sample;i++)
    {
        min = INT_MAX;
        for(int j=0;j<sample;j++)
        {
            if(i!=j)
            {
                dist = l1_distance(arr[i], arr[j]);

                if(dist < min)
                    min = dist;
            }
        }

        sum+=min;
    }

    sum /= sample;

    return sum;
}

neighbor brute_force_NN(image *arr, image query,int size)
{
    int dist;    
	neighbor result;
	
	result.p = arr[0];
    result.distance = l1_distance(arr[0], query);
    

    for(int i=1;i<size;i++)
    {
        dist = l1_distance(arr[i], query);
        
        if(dist < result.distance)
        {
        	result.p = arr[i];
        	result.distance=dist;
		}
        
    }

    return result;
}




vector<double>* compute_s(int w,int k,int size,uniform_real_distribution<double> distribution, default_random_engine* generator)
{
	
    double si;
    vector <double> *S_arr = new vector <double>[k];

    for(int i=0;i<k;i++)
    {
        for(int j=0;j<size;j++)
        {
            si=distribution(*generator);
            S_arr[i].push_back(si);
        }
    }

    return S_arr;
}


void hamming(string str, int i, int changesLeft, vector<unsigned int>* vertices, const unsigned int limit) 
{
    if (changesLeft == 0) 
	{
        unsigned int temp;

        temp= (unsigned int) stoul(str,nullptr,2);
        
        vertices->push_back(temp);
        return;
    }
    
    if (i < (int)limit) return;
    // flip current bit
    str[i] = str[i] == '0' ? '1' : '0';
    hamming(str, i-1, changesLeft-1, vertices,limit);
    // or don't flip it (flip it again to undo)
    str[i] = str[i] == '0' ? '1' : '0';
    hamming(str, i-1, changesLeft, vertices,limit);
}




