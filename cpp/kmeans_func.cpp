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
#include "kmeans_func.h"

using namespace std;

//----------------------------------------------------------------

unsigned int min_dist(image x, vector <image> centroids)
{
	unsigned int min = l1_distance(x, centroids[0]);
	unsigned int dist;
	
	for(int i=1;i<centroids.size();i++)
	{
		dist = l1_distance(x, centroids[i]);
		
		if(dist<min)
			min=dist;
	}
	
	return min;
}

unsigned int min_cluster(image x, vector <cluster>* clusters)
{
	unsigned int min_dist = UINT_MAX;
	unsigned int min = 0;
	unsigned int dist;
	
	for(int i=0;i<clusters->size();i++)
	{
		if(x.cluster != i+1) //if this isn't the same cluster
		{
			dist = l1_distance(x, clusters->at(i).centroid);
			
			if(dist<min_dist)
			{
				min_dist = dist;
				min=i;
			}
		}
			
	}
	
	return min;
}

double centroid_radius(vector <cluster>* clusters)
{
	unsigned int min = INT_MAX;
	unsigned int dist;
	
	for(int i=0;i<clusters->size()-1;i++)
	{
		for(int j=i+1;j<clusters->size();j++)
		{
			dist = l1_distance(clusters->at(i).centroid, clusters->at(j).centroid);
		
			if(dist<min)
				min=dist;
		}
		
	}
	
	return (double)min/2.0;
}


int assign_cluster(image* x, vector <cluster>* clusters)
{
	unsigned int min = l1_distance(*x, clusters->at(0).centroid);
	unsigned int dist;
	
	unsigned int cluster=0;
	
	for(int i=1;i<clusters->size();i++)
	{
		dist = l1_distance(*x, clusters->at(i).centroid);
		
		if(dist<min)
		{
			min=dist;
			cluster=i;
		}	
	}
	
	int result;


    result=0;
	if(x->old_cluster != cluster+1)
	{
        result = 1;
    }
    x->cluster = cluster+1;

	clusters->at(cluster).points.push_back(*x);
	
	return result;
}

int assign_lsh(hashtable** ht, vector <cluster>* clusters, image* arr, int arr_size, int L)
{
	double R = centroid_radius(clusters); //starting range
	
	unsigned int g,size,dist,temp;
	image query;
	int counter=0;
	bool flag=1;
	list <image>* bucket;
	
	std::list<image>::iterator image_it;

	
	unsigned int buckets = ht[0]->get_bucket_no();
	vector <unsigned int> keys;

	while(flag) //loop for assignment
	{
		flag=0;
		
		for(int c=0;c<clusters->size();c++) //for each centroid
		{

			counter=0;
			query = clusters->at(c).centroid;
			
			for(int i=0;i<L;i++) //for each hashtable
		    {
		    	g = ht[i]->hash(query); //find the query bucket
		    	
				query.g = g;
		    	bucket = ht[i]->get_bucket(query.g%buckets); //get the hashtable bucket
		    	
				size = bucket->size();
		    	
		        image_it = bucket->begin();
		        
		        for(int j=0;j<size;j++) //check every element in the bucket
		        {		        	
					if(query.g != image_it->g) //check only same g
		        		continue;
		        		
		            dist = l1_distance(query, *image_it);
		
					
		            if((double)dist <= R) //if point is in centroid range
		            {	
						if(arr[image_it->id].cluster == -1) //if this point has not been assigned to a cluster
						{
							arr[image_it->id].cluster=c+1; //add the cluster to the image
							counter++;
						}
						
						else if(arr[image_it->id].cluster != c+1) //if there is a clash between centroids for this point
						{
							if(l1_distance(arr[image_it->id], clusters->at(arr[image_it->id].cluster).centroid) > l1_distance(arr[image_it->id], query)) //compare distances
							{
								arr[image_it->id].cluster = c+1;
								counter++;
							}	
						
						}
						
					}	
					
					std::advance(image_it,1);
					
		        }
		        
		
		    }
		 	
			if(counter > 0)
		 		flag=1;
		    
		}//end for centroids
		
	}//end while
	
		
	int changes=0; 
	for(int i=0;i<arr_size;i++) //count the number of cluster changes
	{
		
		if(arr[i].cluster == -1)
		{
			changes += assign_cluster(&arr[i],clusters);
		}
		
		else if(arr[i].cluster != arr[i].old_cluster)
		{
			changes++;
			clusters->at(arr[i].cluster-1).points.push_back(arr[i]);
		}
		
		else
			clusters->at(arr[i].cluster-1).points.push_back(arr[i]);
		
	}
	
		
	return changes;
}


int assign_hypercube(hypercube_HT* ht, vector <cluster>* clusters, image* arr, int arr_size, int limit, int probes)
{
	if(!limit)
	{
		cout<< "Limit is 0"<<endl;
		return 0;
	}
	
	if(!probes)
	{
		cout<< "Probes are 0"<<endl;
		return 0;
	}
			
	int dist,counter=0,size,d,point_lim=0,bucket_lim=0;
	unsigned int g,vertex;
	list <image>* bucket;
	image query;
	
	std::list<image>::iterator image_it;

	
	double R = centroid_radius(clusters); //starting range
	
	unsigned int bucket_no = ht->get_bucket_no();
	
	string gstr = bitset<32>(g).to_string(); //to binary
	vector <unsigned int> ham_vec;
	
	d = ht->get_d(); //number of bits in g
	
	bool flag=1;
	while(flag)
	{
		flag=0;
		
		for(int c=0;c<clusters->size();c++)
		{
			point_lim=0;
			bucket_lim=0;
			counter=0;
			
			query = clusters->at(c).centroid;
			g = ht->hash(query); //find the query bucket
		
			for(int i=0;i<=d;i++) //for every hamming distance
		    {
		    	hamming(gstr, gstr.length()-1, i, &ham_vec,gstr.length()-d); //add all the vertices of hamming distance i to ham_vec
		
		
		    	for(int j=0;j<ham_vec.size();j++) //for each bucket in distance i
				{
					vertex = ham_vec[j];
					bucket = ht->get_bucket(vertex%bucket_no);
					
					size = bucket->size();
					image_it = bucket->begin();
					
					for(int k=0;k<size;k++) //check every element in the bucket
			        {     	
			            dist = l1_distance(query, *image_it);
						
			            if(dist <= R)
			            {					
							if(arr[image_it->id].cluster == -1) //if this point has not been assigned to a cluster
							{
								arr[image_it->id].cluster=c+1; //add the cluster to the image
								counter++;
							}
							
							else if(arr[image_it->id].cluster != c+1) //if there is a clash between centroids for this point
							{
								if(l1_distance(arr[image_it->id], clusters->at(arr[image_it->id].cluster).centroid) > l1_distance(arr[image_it->id], query)) //compare distances
								{
									arr[image_it->id].cluster = c+1;
									counter++;
								}	
							
							}
						}	
						
						std::advance(image_it,1);
						
						if(++bucket_lim >= probes) //force exit the first 3 loops (goto next c)
						{
							i=d+1;
							j=ham_vec.size();
							k=size;
						}
							
						
			        } //end for k
			        
			        //probes
					if(++point_lim >= limit) //force exit the first 2 loops (goto next c)
					{
						i=d+1;
						j=ham_vec.size();
					}	
					
				} // end for j
		
		    	ham_vec.clear(); //empty the hamming vector
		        
		
		    }//end for i
		    
		    if(counter > 0) //exit condition for while
		 		flag=1;
		    
		} //end for c
		
	}//end while
	
	
		
	int changes=0; 
	for(int i=0;i<arr_size;i++) //count the number of cluster changes
	{
		
		if(arr[i].cluster == -1)
		{
			changes += assign_cluster(&arr[i],clusters);
		}
		
		else if(arr[i].cluster != arr[i].old_cluster)
		{
			changes++;
			clusters->at(arr[i].cluster-1).points.push_back(arr[i]);
		}
		
		else
			clusters->at(arr[i].cluster-1).points.push_back(arr[i]);
		
	}
	
		
	return changes;
	
}

vector <image> kmeans_pp(image* arr, unsigned int d, unsigned int arr_size, int k, default_random_engine *generator) {
    int t = 1;

    unsigned int temp = rand() % arr_size;

    unsigned int dist, max_dist;
    unsigned long sum;
    bool flag;

    vector<unsigned int> dists;
    vector<double> sums;

    vector<image> result;
    result.push_back(arr[temp]);
    result[0].id = 1;


    vector<int> temp_indexes;      // id(index) of every image selected as a centroid
    temp_indexes.push_back(temp);

    double x;



    while (t < k)
    {

        max_dist = 0;
        for (int i = 0; i < arr_size; i++)
        {
            //find the closest centroid
            dist = min_dist(arr[i], result);

            if (dist != 0)
                dists.push_back(dist);

            if (dist > max_dist)
                max_dist = dist;

        }


        sum = 0;
        sums.push_back(0.0);
        for (int i = 0; i < arr_size - t; i++)
        {
            sum += pow(dists[i], 2);
            sums.push_back((double) sum / (double) max_dist);
        }



        std::random_device rand_dev;
        std::mt19937 gen(rand_dev());
        uniform_real_distribution<double> distribution(0.0, (double) sum / (double) max_dist);
        
//		x=distribution(*generator); //take the first value and throw it away;


        flag = 1;



        while (flag)
        {
            //    uniform_real_distribution<double> distribution(0.0, (double) sum/(double) max_dist);
            x = distribution(gen);
            temp = lower_bound(sums.begin(), sums.end(), x) - sums.begin();

            flag = 0;

            for (int j = 0; j < temp_indexes.size(); j++)
            {
                if (temp_indexes[j] == temp) {
                    flag = 1;
                    break;
                }
            }
            /*
            for(int i=0;i<result.size();i++)
            {

                if(result[i].id == arr[temp].id)
                {
                    flag=1;
                    break;
                }
            }
             */
        }



        result.push_back(arr[temp]);
        result[t].id = t + 1;

        temp_indexes.push_back(temp);

        sums.clear();
        dists.clear();
        t++;

    }


    return result;
}



void update_centroids(vector<cluster>* clusters)
{
	unsigned int dimension = clusters->at(0).points[0].pixels.size(); //image vector dimension
	unsigned int sum=0,images;
	
	image centroid;
	
	for(int i=0;i<clusters->size();i++)
	{
		images = clusters->at(i).points.size();
		centroid.id = clusters->at(i).centroid.id;
		centroid.pixels.clear();
		
		for(int j=0;j<dimension;j++)
		{
			if(images==0)
				continue;
				
			sum=0;
			for(int k=0;k<images;k++)
			{
				sum += clusters->at(i).points[k].pixels[j]; //i-th cluster. k-th image, j-th dimension
			}
			
			sum /= images;
			
			
			centroid.pixels.push_back(sum);
		}
		
		clusters->at(i).centroid = centroid;
	}
	
	return;
}

vector<cluster> kmeans_lloyds(image* arr, unsigned int d, unsigned int arr_size, int K, int threshold)
{
	default_random_engine generator;
	
    
	//k means++
    vector<image> centroids=kmeans_pp(arr,d,arr_size,K,&generator);

    vector<cluster> clusters;
    cluster temp;
    int changes;
    
    for(int i=0;i<K;i++)
    {
    	temp.centroid = centroids[i];
    	clusters.push_back(temp);
	}
    
    while(1)
    {
    	//clear the clusters for new assignment
    	for(int i=0;i<K;i++)
    	{
	    	clusters[i].points.clear();
		}
		
		for(int i=0;i<arr_size;i++)
    	{
    		arr[i].old_cluster = arr[i].cluster;
    		arr[i].cluster = -1;	
    	}
	
		//assign each point to a cluster
		changes=0;
    	for(int i=0;i<arr_size;i++)
    	{
    		changes += assign_cluster(&(arr[i]), &clusters);
		}
		
		//cout<< "CHANGES "<<changes<<endl;
		
		
		update_centroids(&clusters);
		
		if(changes < threshold)
			break;
		
	}
    
    return clusters;
}

vector<cluster> kmeans_lsh(image* arr, unsigned int d, unsigned int arr_size, int K, int L, int k_lsh, int w, int threshold)
{
	default_random_engine generator;
	
    
	//k means++
    vector<image> centroids=kmeans_pp(arr,d,arr_size,K,&generator);

    vector<cluster> clusters;
    cluster temp;
    int changes;
    
    for(int i=0;i<K;i++)
    {
    	temp.centroid = centroids[i];
    	clusters.push_back(temp);
	}
	
	
	//insert the images into lsh array
	hashtable** lsh = new hashtable *[L];
    
    //random engine for uniform distribution
    uniform_real_distribution<double> distribution(0.0, w);
    int buckets = arr_size/64;
    unsigned int M = pow(2,32/k_lsh);
    vector <double>* S_arr;
    for(int i=0;i<L;i++)
    {
    	S_arr = compute_s(w,k_lsh,d,distribution,&generator); //compute the d*k different s for the hashtable
    	lsh[i] = new hashtable(buckets,d,k_lsh,w,M,S_arr);
	}
    
    for(int i=0;i<arr_size;i++)
    {
    	for(int j=0;j<L;j++)
        {
        	lsh[j]->insert_image(arr[i]);
		}
    }
    
	
    while(1)
    {
    	//clear the clusters for new assignment
    	for(int i=0;i<K;i++)
    	{
	    	clusters[i].points.clear();
		}
		
		for(int i=0;i<arr_size;i++)
    	{
    		arr[i].old_cluster = arr[i].cluster;
    		arr[i].cluster = -1;
    		
    	}
	
		//assign each point to a cluster
		changes = assign_lsh(lsh, &clusters, arr, arr_size, L);
		
		
		cout<< "CHANGES "<<changes<<endl;
		
		update_centroids(&clusters);
		
		if(changes < threshold)
			break;
		
	}
	
	for(int i=0;i<L;i++)
    {
    	delete lsh[i];
	}
	delete[] lsh;
    
    return clusters;
}

vector<cluster> kmeans_hypercube(image* arr, unsigned int d, unsigned int arr_size, int K, int cube_k, int limit, int probes, int w, int lsh_k, int threshold)
{
	default_random_engine generator;
	uniform_real_distribution<double> distribution(0.0, w);
	
    unsigned int M = pow(2,32/cube_k);
    
	//k means++
    vector<image> centroids=kmeans_pp(arr,d,arr_size,K,&generator);

    vector<cluster> clusters;
    cluster temp;
    int changes;
    
    for(int i=0;i<K;i++)
    {
    	temp.centroid = centroids[i];
    	clusters.push_back(temp);
	}
	
	
	//insert points into hypercube
	hypercube_HT* cube;
	
	vector <double>* S_arr = compute_s(w,cube_k,d,distribution,&generator); //compute the d*k different s for the hashtable
	cube = new hypercube_HT(cube_k,d,lsh_k,w,M,S_arr);
    
    srand(time(NULL));
    for(int i=0;i<arr_size;i++)
    {
    	cube->insert_image(arr[i]);	
    }
    
	
    while(1)
    {
    	//clear the clusters for new assignment
    	for(int i=0;i<K;i++)
    	{
	    	clusters[i].points.clear();
		}
		
		for(int i=0;i<arr_size;i++)
    	{
    		arr[i].old_cluster = arr[i].cluster;
    		arr[i].cluster = -1;
    		
    	}
	
		//assign each point to a cluster
		changes = assign_hypercube(cube, &clusters, arr, arr_size, limit, probes);
		
		
		cout<< "CHANGES "<<changes<<endl;
		
		update_centroids(&clusters);
		
		if(changes < threshold)
			break;
		
	}
	
    delete cube;
    return clusters;
}


vector <double> silhouette(image* arr, unsigned int arr_size,vector <cluster> clusters)
{
	unsigned int a;
	unsigned int b;
	double s,s_cluster;
	
	unsigned int dist;
	cluster* clust_p;
	
	vector <double> si (arr_size,0.0);
	vector <double> result;
	
	
	for(int i=0;i<clusters.size();i++) //for every cluster
	{
		vector<vector<unsigned int>> same_dist(clusters[i].points.size(), vector<unsigned int> (clusters[i].points.size(), 0));
		s_cluster = 0.0;
		
		//distances of same cluster
		for(int j=0;j<clusters[i].points.size()-1;j++) //for every image in the cluster
		{
			for(int k=j+1;k<clusters[i].points.size();k++) //compare it with all the others
			{
				dist = l1_distance(clusters[i].points[j],clusters[i].points[k]);
				
				same_dist[j][k] = dist;
				same_dist[k][j] = dist;	
			}
		}
		
		
		for(int j=0;j<clusters[i].points.size();j++) //for each point in the cluster
		{
			a=0;
			for(int k=0;k<clusters[i].points.size();k++)
			{
				a += same_dist[j][k];
			}
			
			a /= clusters[i].points.size();
			
			
			//find the next best cluster
			clust_p = &clusters[min_cluster(clusters[i].points[j],&clusters)];
			
			b=0;
			for(int k=0;k<clust_p->points.size();k++)
			{
				b += l1_distance(clusters[i].points[j],clust_p->points[k]);
			}
			b /= clust_p->points.size();
			
			s=0.0;
			if(a<b)
			{
				s = 1.0 - ( (double) a/ (double) b ); 
			}
			
			else if(b<a)
			{
				s = ( (double) b / (double) a ) - 1.0;
			}
			
			s_cluster += s;
			
			si[clusters[i].points[j].id] = s;
			
		}//end for point
		
		s_cluster /= (double) clusters[i].points.size();
		result.push_back(s_cluster);
		
	} //END FOR CLUSTER
	
	//compute total s for data set
	s=0.0;
	for(int i=0;i<arr_size;i++)
	{
		s += si[i];
	}
	s /= (double)arr_size;
	
	result.push_back(s);
	
	return result;
}

unsigned int objective(vector <cluster> clusters)
{
	unsigned int dist,result=0;
	
	for(int i=0;i<clusters.size();i++)
	{
		for(int j=0;j<clusters[i].points.size();j++)
		{
			dist = l1_distance(clusters[i].points[j],clusters[i].centroid);
			result += dist;
		}
	}
	
	return result;
}


