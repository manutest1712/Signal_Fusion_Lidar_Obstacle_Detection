/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>

#include "../../render/render.h"
#include "../../render/box.h"
#include <chrono>
#include <string>
#include "kdtree.h"

typedef unsigned int uint;

// Arguments:
// window is the region to draw box around
// increase zoom to see more of the area
pcl::visualization::PCLVisualizer::Ptr initScene(Box window, int zoom)
{
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("2D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
  	viewer->initCameraParameters();
  	viewer->setCameraPosition(0, 0, zoom, 0, 1, 0);
  	viewer->addCoordinateSystem (1.0);

  	viewer->addCube(window.x_min, window.x_max, window.y_min, window.y_max, 0, 0, 1, 1, 1, "window");
  	return viewer;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData(std::vector<std::vector<float>> points)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  	
  	for(int i = 0; i < points.size(); i++)
  	{
  		pcl::PointXYZ point;
  		point.x = points[i][0];
  		point.y = points[i][1];
  		point.z = 0;

  		cloud->points.push_back(point);

  	}
  	cloud->width = cloud->points.size();
  	cloud->height = 1;

  	return cloud;

}


void render2DTree(Node* node, pcl::visualization::PCLVisualizer::Ptr& viewer, Box window, int& iteration, uint depth=0)
{

	if(node!=NULL)
	{
		Box upperWindow = window;
		Box lowerWindow = window;
		// split on x axis
		if(depth%2==0)
		{
			viewer->addLine(pcl::PointXYZ(node->point[0], window.y_min, 0),pcl::PointXYZ(node->point[0], window.y_max, 0),0,0,1,"line"+std::to_string(iteration));
			lowerWindow.x_max = node->point[0];
			upperWindow.x_min = node->point[0];
		}
		// split on y axis
		else
		{
			viewer->addLine(pcl::PointXYZ(window.x_min, node->point[1], 0),pcl::PointXYZ(window.x_max, node->point[1], 0),1,0,0,"line"+std::to_string(iteration));
			lowerWindow.y_max = node->point[1];
			upperWindow.y_min = node->point[1];
		}
		iteration++;

		render2DTree(node->left,viewer, lowerWindow, iteration, depth+1);
		render2DTree(node->right,viewer, upperWindow, iteration, depth+1);


	}

}

void proximity(int idx,
    const std::vector<std::vector<float>>& points,
    std::vector<bool>& processed,
    std::vector<int>& cluster,
    KdTree* tree,
    float distanceTol)
{
    // Mark this point as processed
    processed[idx] = true;

    // Add the point’s index to the cluster
    cluster.push_back(idx);

    // Search for nearby points
    std::vector<int> nearby = tree->search(points[idx], distanceTol);


    // Explore all neighbors
    for (int id : nearby)
    {
        if (!processed[id])
        {
            // Recursive expansion
            proximity(id, points, processed, cluster, tree, distanceTol);
        }
    }
}

std::vector<std::vector<int>> euclideanCluster(const std::vector<std::vector<float>>& points, KdTree* tree, float distanceTol)
{

	std::vector<std::vector<int>> clusters;

    // Track processed points
    std::vector<bool> processed(points.size(), false);

    // Iterate through every point
    for (int i = 0; i < points.size(); i++)
    {
        if (!processed[i])
        {
            std::vector<int> cluster;
            proximity(i, points, processed, cluster, tree, distanceTol);
            clusters.push_back(cluster);
        }
    }
 
	return clusters;

}

// ----------------------------------------
//   STRUCT FOR STIXEL
// ----------------------------------------
struct Stixel
{
    float x;
    float y;
    float min_z;
    float max_z;
};

// ----------------------------------------
std::vector<Stixel> createStixels(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster,
    float columnWidth = 0.2f)
{
    std::vector<Stixel> stixels;

    if (cluster->empty())
        return stixels;

    // Get min and max bounds
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cluster, minPt, maxPt);

    // Divide into vertical bins along X axis
    for (float x = minPt.x; x <= maxPt.x; x += columnWidth)
    {
        float minZ = +1e9;
        float maxZ = -1e9;
        float y_avg = 0;
        int count = 0;

        for (const auto& p : cluster->points)
        {
            if (p.x >= x && p.x < x + columnWidth)
            {
                minZ = std::min(minZ, p.z);
                maxZ = std::max(maxZ, p.z);
                y_avg += p.y;
                count++;
            }
        }

        if (count > 0)
        {
            Stixel s;
            s.x = x + columnWidth / 2.0f;
            s.y = y_avg / count;
            s.min_z = minZ;
            s.max_z = maxZ;
            stixels.push_back(s);
        }
    }

    return stixels;
}

int stixelGenerator()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Load your PCD
    if (pcl::io::loadPCDFile("C:/Projects/SignalFusion/Signal_Fusion_Lidar_Obstacle_Detection/src/sensors/data/pcd/data_1/0000000000.pcd", *cloud) == -1)
    {
        std::cerr << "ERROR: Cannot read input.pcd\n";
        return -1;
    }

    std::cout << "Loaded point cloud: " << cloud->size() << " points\n";

    // ----------------------------------------
    // 1. Ground removal using RANSAC Plane Fit
    // ----------------------------------------
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);   // adjust depending on LiDAR
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty())
    {
        std::cerr << "No ground plane found.\n";
        return -1;
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;

    // Extract ground
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*ground);

    // Extract obstacles
    pcl::PointCloud<pcl::PointXYZ>::Ptr obstacles(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setNegative(true);
    extract.filter(*obstacles);

    std::cout << "Ground points: " << ground->size() << "\n";
    std::cout << "Obstacle points: " << obstacles->size() << "\n";

    // ----------------------------------------
    // 2. Euclidean Clustering
    // ----------------------------------------
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(obstacles);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    ec.setClusterTolerance(0.5);  // meters
    ec.setMinClusterSize(20);
    ec.setMaxClusterSize(10000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(obstacles);
    ec.extract(clusterIndices);

    std::cout << "Clusters found: " << clusterIndices.size() << "\n";

    // ----------------------------------------
    // 3. For each cluster, generate stixels
    // ----------------------------------------
    int id = 0;
    for (const auto& indices : clusterIndices)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (int idx : indices.indices)
            cluster->points.push_back(obstacles->points[idx]);

        auto stixels = createStixels(cluster);

        std::cout << "\nCluster " << id++ << ": " << cluster->size() << " points\n";
        std::cout << "Generated " << stixels.size() << " stixels\n";

        for (const Stixel& s : stixels)
        {
            std::cout << "  x=" << s.x
                << " y=" << s.y
                << " z_min=" << s.min_z
                << " z_max=" << s.max_z
                << "\n";
        }
    }

    return 0;
}

int main ()
{
    stixelGenerator();

	// Create viewer
	Box window;
  	window.x_min = -10;
  	window.x_max =  10;
  	window.y_min = -10;
  	window.y_max =  10;
  	window.z_min =   0;
  	window.z_max =   0;
	pcl::visualization::PCLVisualizer::Ptr viewer = initScene(window, 25);

	// Create data
	std::vector<std::vector<float>> points = { {-6.2,7}, {-6.3,8.4}, {-5.2,7.1}, {-5.7,6.3}, {7.2,6.1}, {8.0,5.3}, {7.2,7.1}, {0.2,-7.1}, {1.7,-6.9}, {-1.2,-7.2}, {2.2,-8.9} };
	//std::vector<std::vector<float>> points = { {-6.2,7}, {-6.3,8.4}, {-5.2,7.1}, {-5.7,6.3} };
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData(points);

	KdTree* tree = new KdTree;
  
    //for (int i=0; i<points.size(); i++) 
    //	tree->insert(points[i],i); 

  	int it = 0;
  	//render2DTree(tree->root,viewer,window, it);

    KdTree* treeNew = new KdTree ;
    treeNew->buildTree(points);

    render2DTree(treeNew->root, viewer, window, it);
  
  	std::cout << "Test Search" << std::endl;
  	std::vector<int> nearby = treeNew->search({-6,7},3.0);
  	for(int index : nearby)
      std::cout << index << ",";
  	std::cout << std::endl;

  	// Time segmentation process
  	auto startTime = std::chrono::steady_clock::now();
  	//
  	std::vector<std::vector<int>> clusters = euclideanCluster(points, treeNew, 3.0);
  	//
  	auto endTime = std::chrono::steady_clock::now();
  	auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  	std::cout << "clustering found " << clusters.size() << " and took " << elapsedTime.count() << " milliseconds" << std::endl;

  	// Render clusters
  	int clusterId = 0;
	std::vector<Color> colors = {Color(1,0,0), Color(0,1,0), Color(0,0,1)};
  	for(std::vector<int> cluster : clusters)
  	{
  		pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZ>());
  		for(int indice: cluster)
  			clusterCloud->points.push_back(pcl::PointXYZ(points[indice][0],points[indice][1],0));
  		renderPointCloud(viewer, clusterCloud,"cluster"+std::to_string(clusterId),colors[clusterId%3]);
  		++clusterId;
  	}
  	if(clusters.size()==0)
  		renderPointCloud(viewer,cloud,"data");
	
  	while (!viewer->wasStopped ())
  	{
  	  viewer->spinOnce ();
  	}

    
  	
}
