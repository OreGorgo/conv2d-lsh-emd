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

//$./cluster –i <input file> –c <configuration file> -o <output file> -complete
//<optional> -m <method: Classic OR LSH or Hypercube>

using namespace std;
using namespace std::chrono; 





int main(int argc,char **argv)
{
    string input_file,new_input_file,config_file,output_file,classes_file;


    for(int i=1;i<argc;i++)
    {
        if(strcmp(argv[i],"-d") == 0)
        {
            i++;
            input_file=argv[i];
        }
        
        if(strcmp(argv[i],"-i") == 0)
        {
            i++;
            new_input_file=argv[i];
        }
        
        if(strcmp(argv[i],"-n") == 0)
        {
            i++;
            classes_file=argv[i];
        }


        if(strcmp(argv[i],"-c") == 0)
        {
            i++;
            config_file=argv[i];
        }

        if(strcmp(argv[i],"-o") == 0)
        {
            i++;
            output_file=argv[i];
        }


    }



    if(input_file.empty())
    {
        cout<<"Give input file!"<<endl;
        cin >> input_file;
    }
    
    if(new_input_file.empty())
    {
        cout<<"Give new input file!"<<endl;
        cin >> input_file;
    }

    if(config_file.empty())
    {
        cout<<"Give config file!"<<endl;
        cin >> config_file;
    }

    if(output_file.empty())
    {
        cout<<"Give output file!"<<endl;
        cin >> output_file;
    }
	
	
    int images,rows,cols;

    ifstream fp(input_file.c_str(),  ios::in | ios::binary );
    if (!fp.is_open())
    {
        cout << "Unable to open file";
        return 1;
    }

    srand(time(nullptr));


    //get metadata
    fp.seekg(4,fp.beg);
    fp.read ((char*)&images,4);

    images = bswap_32(images);

    //fp.seekg(8,fp.beg);
    fp.read((char*)&rows,4);

    rows = bswap_32(rows);

    //fp.seekg(12,fp.beg);
    fp.read((char*)&cols,4);

    cols = bswap_32(cols);

    //Data starts here
    //fp.seekg(16,fp.beg);
    

    image* arr = new image[images];
    unsigned char temp;
    

    int size = rows*cols;

    for(int j=0;j<images;j++)
    {
    	arr[j].id = j;
        arr[j].cluster = -1;
        arr[j].old_cluster = -1;
        arr[j].g = 0;
        
		for(int i=0; i<size; i++)
        {
            fp >> temp;
            
            arr[j].pixels.push_back(temp);

            //cout << (int) arr[j].pixels.at(i) <<" ";
        }

    }
    
    
    
    
    ifstream latent_fp(new_input_file.c_str(),  ios::in | ios::binary );
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
	
    int latent_size = rows*cols;

    for(int j=0;j<images;j++) //read the images and put them in an array
    {
    	latent_arr[j].id = j;
        latent_arr[j].cluster = -1;
        latent_arr[j].old_cluster = -1;
        latent_arr[j].g = 0;
			
        for(int i=0; i<latent_size; i++)
        {
            latent_fp.read((char*)&temp_s,2);
            temp_s = bswap_16(temp_s);
            
            latent_arr[j].pixels.push_back(temp_s);
            
        }

    }
    
    latent_fp.close();
    
    
    
    
    int K = 4; //KMEANS
    
	int L = 3; //number of hashtables
    int lsh_k = 4;//number of h for LSH
    
    int cube_M = 10;//hypercube M
    int cube_k = 3;//hypercube k
    int probes = 2;//number of probes for hypercube
    
    
    char* input;
    
    //reading of config file
   	ifstream conf(config_file.c_str());
	if (conf.is_open()) {
	  	string line;
	    while (std::getline(conf, line)) {
	        
	        if(strncmp("number_of_clusters:",line.c_str(),strlen("number_of_clusters:")) == 0)
	        {
	        	input = (char*)line.c_str();
	        	input += strlen("number_of_clusters:");
	        	while(*input == ' ')
	        		input++;
	        	
	        	K = atoi(input);
			}
			else if(strncmp("number_of_vector_hash_tables:",line.c_str(),strlen("number_of_vector_hash_tables:")) == 0)
	        {
	        	input = (char*) line.c_str() + strlen("number_of_vector_hash_tables:");
	        	while(*input == ' ')
	        		input++;
	        	
	        	L = atoi(input);
			}
			else if(strncmp("number_of_vector_hash_functions:",line.c_str(),strlen("number_of_vector_hash_functions:")) == 0)
	        {
	        	input = (char*) line.c_str() + strlen("number_of_vector_hash_functions:");
	        	while(*input == ' ')
	        		input++;
	        	
	        	lsh_k = atoi(input);
			}
			
	    }
	    conf.close();
	} 
    
	//int w = average_NN(arr, 500);
    int w = 40000;
    
    int threshold = images/500;
    


    ofstream output_fp;
    output_fp.open (output_file);

    vector<cluster> s1_clusters;
    vector <double> sil;


    auto start = high_resolution_clock::now();


    s1_clusters = kmeans_lloyds(arr, size, images, K, threshold);


    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start); 
    
    //prints

	for(int i=0;i<s1_clusters.size();i++)
    {
	    output_fp<<"CLUSTER-"<<i+1<<" {size: "<<s1_clusters[i].points.size()<<" , "<<"centroid [ ";
	    for(int j=0;j<size;j++)
	    {
            output_fp << s1_clusters[i].centroid.pixels[j];
            if (j < size - 1)
                output_fp << " , ";
        }

	    output_fp<<" ] }"<<endl;
    }


    output_fp <<"clustering_time: "<< (double)duration.count()/1000000.0 << endl;


    sil = silhouette(arr,images,s1_clusters);

    output_fp<<"Silhouette: [ ";
    for(int i=0;i<sil.size();i++)
    {
        output_fp<<sil[i];
        if(i<sil.size()-1)
            output_fp<<" , ";
    }
    output_fp<<" ]"<<endl;
    
    unsigned int obj = objective(s1_clusters);
	output_fp<<"Value of Objective Function: "<<obj<<".0"<<endl<<endl<<endl;
    
    
    //---------------------------------------------
    
    vector<cluster> latent_clusters;
    vector <double> s2_sil;


    start = high_resolution_clock::now();


    latent_clusters = kmeans_lloyds(latent_arr, size, images, K, threshold);


    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start); 
    
    //create s2 cluster result
    //It's the same as latent clusters but in original dimension
    
    vector<cluster> s2_clusters;
    cluster temp_cluster; 
    
    for(int i=0;i<latent_clusters.size();i++)
    {	
		temp_cluster.centroid = latent_clusters[i].centroid;
		//assign the points in original dimension to a cluster
		for(int j=0;j<latent_clusters[i].points.size();j++)
		{
			temp_cluster.points.push_back(arr[latent_clusters[i].points[j].id]);
		}
		
		s2_clusters.push_back(temp_cluster);
		temp_cluster.points.clear();	
		
	}	
	update_centroids(&s2_clusters); //create new centroids in original space for s2
	
	//prints
	
	
    //prints

	for(int i=0;i<s2_clusters.size();i++)
    {
	    output_fp<<"CLUSTER-"<<i+1<<" {size: "<<s2_clusters[i].points.size()<<" , "<<"centroid [ ";
	    for(int j=0;j<size;j++)
	    {
            output_fp << s2_clusters[i].centroid.pixels[j];
            if (j < size - 1)
                output_fp << " , ";
        }

	    output_fp<<" ] }"<<endl;
    }


    output_fp <<"clustering_time: "<< (double)duration.count()/1000000.0 << endl;


    s2_sil = silhouette(arr,images,s2_clusters);

    output_fp<<"Silhouette: [ ";
    for(int i=0;i<s2_sil.size();i++)
    {
        output_fp<<s2_sil[i];
        if(i<s2_sil.size()-1)
            output_fp<<" , ";
    }
    output_fp<<" ]"<<endl;
    
    obj = objective(s2_clusters);
	output_fp<<"Value of Objective Function: "<<obj<<".0"<<endl<<endl<<endl;
    
    
    // -----S3-----------------
    //parse the classes file
    
    string line;
    ifstream class_fp(classes_file.c_str(),  ios::in);
    if (!class_fp.is_open())
    {
        cout << "Unable to open file";
        return 1;
    }
    
    vector<cluster> s3_clusters;
    image temp_centroid;
    int cluster_size,index;
    string::iterator it;
    string line_part;
    for(int i=0;i<s2_clusters.size();i++)
    {
    	if(i<s2_clusters.size()-1)
    		class_fp.seekg(strlen("CLUSTER-1 { size: "),class_fp.cur);
    	else
    		class_fp.seekg(strlen("CLUSTER-10 { size: "),class_fp.cur);
    		
    	//read a line (cluster from the file
    	getline(class_fp,line);
    		
    	//number of images in this cluster
    	cluster_size = atoi(line.c_str());    	
		
		it = line.begin();
		while(isdigit(*it)) //bypass numbers
		{
			it++;
		}
		it+=2; //bypass ', '
		
		for(int j=0;j<cluster_size;j++)
		{
			line_part = string(it,line.end());
			index = atoi(line_part.c_str());
			
			//output_fp<<index<<endl;
			
			temp_cluster.points.push_back(arr[index]);
			
			
			while(isdigit(*it)) //bypass numbers
			{
				it++;
			}
			if(j != cluster_size-1)
				it+=2; //bypass ', '
			
		}
		
		temp_centroid.id = i+1;
		temp_cluster.centroid = temp_centroid;
		
		s3_clusters.push_back(temp_cluster);
		temp_cluster.points.clear();
		
	}
    update_centroids(&s3_clusters); //create new centroids in original space for s3
    
    //prints

	for(int i=0;i<s3_clusters.size();i++)
    {
	    output_fp<<"CLUSTER-"<<i+1<<" {size: "<<s3_clusters[i].points.size()<<" , "<<"centroid [ ";
	    for(int j=0;j<size;j++)
	    {
            output_fp << s3_clusters[i].centroid.pixels[j];
            if (j < size - 1)
                output_fp << " , ";
        }

	    output_fp<<" ] }"<<endl;
    }


	vector <double> s3_sil;
    s3_sil = silhouette(arr,images,s3_clusters);

    output_fp<<"Silhouette: [ ";
    for(int i=0;i<s3_sil.size();i++)
    {
        output_fp<<s3_sil[i];
        if(i<s3_sil.size()-1)
            output_fp<<" , ";
    }
    output_fp<<" ]"<<endl;
    
    obj = objective(s3_clusters);
	output_fp<<"Value of Objective Function: "<<obj<<".0"<<endl<<endl<<endl;
    

	delete[] arr;
	delete[] latent_arr;
    fp.close();
    output_fp.close();
    
    return 0;
}



//train-images.idx3-ubyte

