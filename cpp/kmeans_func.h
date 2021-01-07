#ifndef KMEANS_H
#define KMEANS_H

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

typedef struct cluster{
	image centroid;
	vector<image> points;
}cluster;


unsigned int min_dist(image x, vector <image> centroids);

unsigned int min_cluster(image x, vector <cluster>* clusters);
double centroid_radius(vector <cluster>* clusters);

int assign_cluster(image* x, vector <cluster>* clusters);

int assign_lsh(hashtable** ht, vector <cluster>* clusters, image* arr, int arr_size, int L);

int assign_hypercube(hypercube_HT* ht, vector <cluster>* clusters, image* arr, int arr_size, int limit, int probes);
vector <image> kmeans_pp(image* arr, unsigned int d, unsigned int arr_size, int k, default_random_engine *generator);


void update_centroids(vector<cluster>* clusters);
vector<cluster> kmeans_lloyds(image* arr, unsigned int d, unsigned int arr_size, int K, int threshold);
vector<cluster> kmeans_lsh(image* arr, unsigned int d, unsigned int arr_size, int K, int L, int k_lsh, int w, int threshold);
vector<cluster> kmeans_hypercube(image* arr, unsigned int d, unsigned int arr_size, int K, int cube_k, int limit, int probes, int w, int lsh_k, int threshold);

vector <double> silhouette(image* arr, unsigned int arr_size,vector <cluster> clusters);
unsigned int objective(vector <cluster> clusters);



#endif
