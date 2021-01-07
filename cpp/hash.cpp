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

using namespace std;


hashtable::hashtable(int buckets,int dimension, int k, int w,int M, vector <double>* S_arr)
{
    this->w = w;
    this->buckets = buckets;
    this->k = k;
    this->dimension = dimension;
    this->M = M;

    this->S_arr = S_arr;
    
    this->table = new vector <list <image> >(buckets);

    list <image> l;

    fill(this->table->begin(),this->table->end(), list <image>() );
    //cout<<"table size is "<<table->size()<<endl;

}

hashtable::~hashtable()
{
	delete[] this->S_arr;
	delete this->table;
}



unsigned int hashtable::compute_h(image input, int i)
{
    vector <int> A;
    int ai;

//    for(int j=0;j<this->dimension;j++)
//    {
//        ai = (int)(input.at(j) - this->S_arr[i].at(j)) / w;
//        A.push_back(ai);
//    }


    //int m = *max_element(A.begin(), A.end()) + 1; //OR TAKE 
	int m = 2^(32-5);

    //pipes gia ypologismo h

    unsigned int h=0,temp;
    
    for(int d=0;d<this->dimension-1;d++)
    {
    	ai = (int)(input.pixels.at(d) - this->S_arr[i].at(d)) / w;
		A.push_back(ai);
        
        
        if(ai == 0)
        	continue;

        temp = (modular_pow(m,this->dimension-1-d,this->M) * A.at(d));
        h += temp;

    }
    
    ai = (int)(input.pixels.at(this->dimension-1) - this->S_arr[i].at(this->dimension-1)) / w;
    A.push_back(ai);

	h+=A.at(this->dimension-1);

    return h;
}


unsigned int hashtable::hash(image input)
{
    vector <unsigned int> H;
    unsigned int g, h;

    for(int i=0;i<this->k;i++) //build the k different h
    {
    	
        h = this->compute_h(input, i); //compute hi
        
        H.push_back(h); //save it on H vector
		
		
		
        if(i>0)
            g = g | (H.at(i) << i*(32/k) ); //bitwise concatenation of h in g
        else
            g=H.at(i);
            
    }

    return g%this->buckets;
}


void hashtable::insert_image(image input)
{
	unsigned int g = this->hash(input);
	
	
	//insert g into the table
    this->table->at(g).push_back(input);

    return;
}


list <image>* hashtable::get_bucket(unsigned int g)
{
	return &(this->table->at(g));	
}

unsigned int hashtable::get_bucket_no()
{
	return this->buckets;
}

//////////////////////////////////////////////////



hypercube_HT::hypercube_HT(int d,int dimension, int k, int w,int M, vector <double>* S_arr) : hashtable(pow(2,d), dimension, k, w, M, S_arr)
{
	this->d = d; 
}


hypercube_HT::~hypercube_HT(){
}

unsigned int hypercube_HT::hash(image input)
{
	
    unsigned int g, h;
    unsigned int fh;
    
    

    for(int i=0;i<this->d;i++) //build the d' different h
    {
    	
        h = this->compute_h(input, i); //compute hi
        
        
        if(this->memory.find(h) == memory.end()) //if h doesn't exist in memory
        {	
        	fh = rand()%2;
        	this->memory[h] = fh;
		}
        
        else
        	fh=this->memory[h];
        
        				
        if(i>0)
            g = g | (fh << i); //bitwise concatenation of h in g
        else
            g=fh;
            
    }
    	

    return g%this->buckets;
}

int hypercube_HT::get_d()
{
	return this->d;
}

////////////////////////////////////////////////////

