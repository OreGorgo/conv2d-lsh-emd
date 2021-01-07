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
#include "lsh_func.h"

using namespace std;

vector <neighbor> knn(hashtable** ht,image query, int k, int L)
{
	
	vector <neighbor> result;
	neighbor temp;
	bool flag=0;
	
	std::vector<neighbor>::iterator neighbor_it;
	
	int max_dist,dist,counter=0,size,limit=20*L;
	unsigned int g;
	list <image>* bucket;
	
	// Create iterator pointing to first element
    	std::list<image>::iterator image_it;
    
	max_dist = INT_MAX; //?
	unsigned int buckets = ht[0]->get_bucket_no();
	
    for(int i=0;i<L;i++)
    {
    	g = ht[i]->hash(query); //find the query bucket
    	
    	query.g = g;
    	bucket = ht[i]->get_bucket(g%buckets);
    	
		size = bucket->size();
    	
        
        
        image_it = bucket->begin();
        
        for(int j=0;j<size;j++) //check every element in the bucket
        {
        	if(query.g != image_it->g)
        		continue;
        		
            	dist = l1_distance(query, *image_it); //*it is bucket->at(j)
           
		temp.distance = dist;
		temp.p = *image_it; 
		
		
		result.push_back(temp); //save the neighbor in result as well as the distance
		
				
		if(result.size() > limit) //stop if you have too many collisions OPTIONAL!!
    	{
    		std::sort(result.begin(), result.end(), my_cmp);
			result.erase( unique( result.begin(), result.end(),cmp ), result.end() );
			return result;
		}
			
			
		std::advance(image_it, 1); //get next list item
			
        }

    }
    
    std::sort(result.begin(), result.end(), my_cmp);
    result.erase( unique( result.begin(), result.end(),cmp ), result.end() );

    return result;
}

vector <unsigned int> range_search(hashtable** ht,image query, double r, int L, int c)
{
	int dist,counter=0,size,limit=20*L;
	unsigned int g;
	list <image>* bucket;
	bool flag;
	
	std::list<image>::iterator image_it;

	
	double range = r*c;
	unsigned int buckets = ht[0]->get_bucket_no();
	
//	cout<< "R-Near neighbors:"<<endl;
	
	vector <unsigned int> visited;
	
    for(int i=0;i<L;i++)
    {
    	g = ht[i]->hash(query); //find the query bucket
    	
		query.g = g;
    	bucket = ht[i]->get_bucket(g%buckets);
    	
		size = bucket->size();
    	
        image_it = bucket->begin();
        
        for(int j=0;j<size;j++) //check every element in the bucket
        {
        	if(query.g != image_it->g)
        		continue;
        		
            dist = l1_distance(query, *image_it);

			
            if(dist <= range)
            {	
            	flag=0;
            	for(int v=0;v<visited.size();v++) //dodge copies of the same point
            	{
            		if(visited[v] == image_it->id)
            		{
            			flag=1;
            			break;
					}		
				}
				if(flag)
					continue;
            	
            	counter++;
			
	//			cout << "image_number_"<< image_it->id<<endl;

				visited.push_back(image_it->id);
				
				if(counter > limit) //stop if you have too many collisions OPTIONAL!!
        		    return visited;
			}	
			
			std::advance(image_it,1);
			
        }
        
        //cout <<endl;

    }


    return visited;
}


