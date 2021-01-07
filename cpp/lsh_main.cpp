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
using namespace std::chrono; 


int main(int argc,char **argv)
{
    string data_path,query_path,output_file;
    string new_data_path, new_query_path;
    int k=4,L=5,N=1;
	double R=10000.0;

    for(int i=1;i<argc;i++)
    {
        if(strcmp(argv[i],"-d") == 0)
        {
            i++;
            data_path=argv[i];
        }


        if(strcmp(argv[i],"-q") == 0)
        {
            i++;
            query_path=argv[i];
        }
        
        if(strcmp(argv[i],"-i") == 0)
        {
            i++;
            new_data_path=argv[i];
        }


        if(strcmp(argv[i],"-s") == 0)
        {
            i++;
            new_query_path=argv[i];
        }

        if(strcmp(argv[i],"-o") == 0)
        {
            i++;
            output_file=argv[i];
        }


        if(strcmp(argv[i],"-k") == 0)
        {
            i++;
            k=atoi(argv[i]);
        }

        if(strcmp(argv[i],"-L") == 0)
        {
            i++;
            L=atoi(argv[i]);
        }


        if(strcmp(argv[i],"-R") == 0)
        {
            i++;
            R=atof(argv[i]);
        }


    }



    if(data_path.empty())
    {
        cout<<"Give dataset path!"<<endl;
        cin >> data_path;
    }

    if(query_path.empty())
    {
        cout<<"Give query path!"<<endl;
        cin >> query_path;
    }
    
    if(new_data_path.empty())
    {
        cout<<"Give new dimension dataset path!"<<endl;
        cin >> new_data_path;
    }

    if(new_query_path.empty())
    {
        cout<<"Give new dimension query path!"<<endl;
        cin >> new_query_path;
    }

    if(output_file.empty())
    {
        cout<<"Give output path!"<<endl;
        cin >> output_file;
    }
    
 
    int images,rows,cols;

    ifstream fp(data_path.c_str(),  ios::in | ios::binary );
    if (!fp.is_open())
    {
        cout << "Unable to open file";
        return 1;
    }


    //get metadata
    fp.seekg(4,fp.beg);
    fp.read ((char*)&images,4);
    images = bswap_32(images); //swap from big endian to little endian
    fp.read((char*)&rows,4);
    rows = bswap_32(rows);
    fp.read((char*)&cols,4);
    cols = bswap_32(cols);
	
    image* arr = new image[images];
    unsigned char temp;
    int size = rows*cols;

    for(int j=0;j<images;j++) //read the images and put them in an array
    {
    	arr[j].id = j;
        arr[j].cluster = -1;
        arr[j].old_cluster = -1;
        arr[j].g = 0;
			
        for(int i=0; i<size; i++)
        {
            fp >> temp;
            arr[j].pixels.push_back(temp);
            
        }

    }


    //int w = average_NN(arr, 500);
    int w = 40000;





    unsigned int M = pow(2,32/k); //modulo number
    unsigned int buckets = images/8;
    vector <double>* S_arr;
    
    
    hashtable** lsh = new hashtable *[L];



    //random engine for uniform distribution
    default_random_engine generator;
    //uniform_real_distribution<double> distribution(0.0, w);
    
    std::random_device rand_dev;
    std::mt19937 gen(rand_dev());
    uniform_real_distribution<double> distribution(0.0, w);
    
    srand(time(NULL));
    
    
    for(int i=0;i<L;i++)
    {
    	S_arr = compute_s(w,k,size,distribution,&generator); //compute the d*k different s for the hashtable
    	lsh[i] = new hashtable(buckets,size,k,w,M,S_arr);
	}
    
    

    for(int i=0;i<images;i++)
    {
    	for(int j=0;j<L;j++)
        {
        	lsh[j]->insert_image(arr[i]);
	}
		
    }



    ifstream query_fp(query_path.c_str(),  ios::in | ios::binary );
    if (!query_fp.is_open())
    {
        cout << "Unable to open file";
        return 1;
    }

//    image* query_arr = new image[images];

	vector <image> query_arr;
	bool eof=false;
	image temp_image;
	int query_images=0;
	int aa;
	
	
	//bypass metadata and get file size
    query_fp.seekg(4,query_fp.beg);
    query_fp.read ((char*)&aa,4);
    query_images = bswap_32(aa);
    query_fp.read((char*)&aa,4);
    query_fp.read((char*)&aa,4);
    for(int j=0;j<query_images;j++)
    {
		query_arr.push_back(temp_image);
        query_arr[j].id = j;
        query_arr[j].cluster = -1;
        query_arr[j].old_cluster = -1;
        query_arr[j].g = 0;


        for(int i=0; i<size; i++)
        {
            query_fp >> temp;
            query_arr[j].pixels.push_back(temp);
        }
                
    }

    query_fp.close();
    fp.close();
    
    
    //KAINOURIA--------------------------------------------------------
    
    //read the dataset file
    
    ifstream latent_fp(new_data_path.c_str(),  ios::in | ios::binary );
    if (!latent_fp.is_open())
    {
        cout << "Unable to open file";
        return 1;
    }
    
    int latent_images=0;

    //get metadata
    latent_fp.seekg(4,latent_fp.beg);
    latent_fp.read ((char*)&latent_images,4);
    latent_images = bswap_32(latent_images); //swap from big endian to little endian
    latent_fp.read((char*)&rows,4);
    rows = bswap_32(rows);
    latent_fp.read((char*)&cols,4);
    cols = bswap_32(cols);
	
    image* latent_arr = new image[latent_images];    
	unsigned short temp_s;
	
    size = rows*cols;

    for(int j=0;j<images;j++) //read the images and put them in an array
    {
    	latent_arr[j].id = j;
        latent_arr[j].cluster = -1;
        latent_arr[j].old_cluster = -1;
        latent_arr[j].g = 0;
			
        for(int i=0; i<size; i++)
        {
            latent_fp.read((char*)&temp_s,2);
            temp_s = bswap_16(temp_s);
            
            latent_arr[j].pixels.push_back(temp_s);
            
        }

    }
    
    latent_fp.close();
    
    
    //read the query file
    
    ifstream latent_query_fp(new_query_path.c_str(),  ios::in | ios::binary );
    if (!latent_query_fp.is_open())
    {
        cout << "Unable to open file";
        return 1;
    }
    
    vector <image> latent_query_arr;
	eof=false;
	int latent_query_images=0;
	
	
	//bypass metadata and get file size
    latent_query_fp.seekg(4,latent_query_fp.beg);
    latent_query_fp.read ((char*)&aa,4);
    latent_query_images = bswap_32(aa);
    latent_query_fp.read((char*)&aa,4);
    latent_query_fp.read((char*)&aa,4);
    for(int j=0;j<latent_query_images;j++)
    {
		latent_query_arr.push_back(temp_image);
        latent_query_arr[j].id = j;
        latent_query_arr[j].cluster = -1;
        latent_query_arr[j].old_cluster = -1;
        latent_query_arr[j].g = 0;

        for(int i=0; i<size; i++)
        {
            latent_query_fp.read((char*)&temp_s,2);
            temp_s = bswap_16(temp_s);
                        
            latent_query_arr[j].pixels.push_back(temp_s);
            
        }
                
    }
    
    latent_query_fp.close();

	//read the query file and use the 1-nn function in every query image the func
    //use the old brute force nn function
	
	vector <neighbor> lsh_result;
	neighbor latent_result;
	neighbor brute_result;
	
	double tlsh=0.0,ttrue=0.0,tlatent=0.0;
	double latent_factor=0.0, lsh_factor=0.0;
	
    ofstream output_fp;
    output_fp.open (output_file);
    
    int counter1=0,counter2=0;

    for(int i=0;i<query_images;i++)
	{
        auto start = high_resolution_clock::now();

        lsh_result = knn(lsh, query_arr[i], 1, L); //knn algorithm

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        tlsh += (double)duration.count();

        start = high_resolution_clock::now();

        brute_result = brute_force_NN(arr,query_arr[i],images); //brute force knn

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        ttrue += (double)duration.count();
        
        start = high_resolution_clock::now();

        latent_result = brute_force_NN(latent_arr,latent_query_arr[i],images); //brute force knn in reduced dimension

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        tlatent += (double)duration.count();
        
        if(brute_result.distance != 0)
		{
			latent_factor += ( (double) l1_distance(arr[latent_result.p.id], query_arr[i]) / (double) brute_result.distance );
			counter1++;
		}        
        if(!lsh_result.empty() && brute_result.distance != 0) //if lsh produces no result, don't increment the lsh factor
        {
        	lsh_factor += (double) lsh_result[0].distance / (double) brute_result.distance;
        	counter2++;
		}
        

        output_fp<<"Query: "<<i+1<<endl<<endl;

        output_fp<<"Nearest neighbor Reduced: "<< latent_result.p.id+1 <<endl;
        if(!lsh_result.empty())
        	output_fp<<"Nearest neighbor LSH: "<< lsh_result[0].p.id+1 <<endl;
        output_fp<<"Nearest neighbor True: "<< brute_result.p.id+1 <<endl<<endl;

		
        
		output_fp<<"distanceReduced: "<<l1_distance(arr[latent_result.p.id], query_arr[i]) <<endl;
		if(!lsh_result.empty())
			output_fp<<"distanceLSH: "<<lsh_result[0].distance <<endl;
        output_fp<<"distanceTrue: "<<brute_result.distance <<endl<<endl;


    }
    
    output_fp <<"tReduced: "<<tlatent  << endl;
    output_fp <<"tLSH: "<<tlsh  << endl;
    output_fp <<"tTrue: "<<ttrue  << endl<<endl;


	latent_factor /= (double) counter1;
	lsh_factor /= (double) counter2;
	
	
	output_fp <<"Approximation Factor LSH: "<<lsh_factor  << endl;
    output_fp <<"Approximation Factor Reduced: "<<latent_factor  << endl;
    
    
    
    output_fp.close();
    //  ../t10k-images.idx3-ubyte

    
    
	for(int i=0;i<L;i++)
    {
    	delete lsh[i];
	}
	delete[] lsh;
    delete[] arr;
    delete[] latent_arr;

    return 0;
}

//train-images.idx3-ubyte

