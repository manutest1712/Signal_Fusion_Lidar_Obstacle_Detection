// PCL lib Functions for processing point clouds 

#include "processPointClouds.h"

//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, 
    Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint,
    Eigen::Vector4f roofMinPoint, Eigen::Vector4f roofMaxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // ---------------------------------------------------------
    // 1. Apply Voxel Grid Downsampling
    // ---------------------------------------------------------

    pcl::VoxelGrid<PointT> vg;
    typename pcl::PointCloud<PointT>::Ptr voxelFiltered(new pcl::PointCloud<PointT>);

    vg.setInputCloud(cloud);
    vg.setLeafSize(filterRes, filterRes, filterRes);  // cube resolution
    vg.filter(*voxelFiltered);

    std::cout << "Voxel grid applied cloud size: " << voxelFiltered->size() << std::endl;

    // ---------------------------------------------------------
    // 2. Region of Interest Filtering (CropBox)
    // ---------------------------------------------------------
    pcl::CropBox<PointT> region(true);  // 'true' = keep organized structure if used
    region.setMin(minPoint);            // minPoint = Eigen::Vector4f(x_min, y_min, z_min, 1.0)
    region.setMax(maxPoint);            // maxPoint = Eigen::Vector4f(x_max, y_max, z_max, 1.0)

    typename pcl::PointCloud<PointT>::Ptr cloudRegion(new pcl::PointCloud<PointT>);
    region.setInputCloud(voxelFiltered);
    region.filter(*cloudRegion);

    std::cout << "After ROI filter cloud size: " << cloudRegion->size() << std::endl;

    // ---------------------------------------------------------
    // 3. Remove points inside a "roof" region
    // ---------------------------------------------------------

    // a. Find indices inside the CropBox
    std::vector<int> indices;

    pcl::CropBox<PointT> roof(true);
    roof.setMin(roofMinPoint);
    roof.setMax(roofMaxPoint);
    roof.setInputCloud(cloudRegion);
    roof.filter(indices);    // indices of points to REMOVE

    // b. Put indices into PointIndices format
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    for (int point_index : indices)
    {
        inliers->indices.push_back(point_index);
    }

    // c. Extract and REMOVE those indices
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloudRegion);
    extract.setIndices(inliers);
    extract.setNegative(true);        // <-- true = REMOVE these points
    extract.filter(*cloudRegion);     // result written back to same cloud

    
    std::cout << "After Removing Roof top filter cloud size: " << cloudRegion->size() << std::endl;

    return cloudRegion;



    // TODO:: Fill in the function to do voxel grid point reduction and region based filtering

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloud;

}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud)
{
    // Extract plane (inliers)
    typename pcl::PointCloud<PointT>::Ptr planeCloud(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr obstacleCloud(new pcl::PointCloud<PointT>());

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);

    // Extract plane points
    extract.setNegative(false);
    extract.filter(*planeCloud);

    // Extract everything else (obstacles)
    extract.setNegative(true);
    extract.filter(*obstacleCloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstacleCloud, planeCloud);
    return segResult;
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr,
    typename pcl::PointCloud<PointT>::Ptr>
    ProcessPointClouds<PointT>::Segment(typename pcl::PointCloud<PointT>::Ptr cloud,
        int maxIterations,
        float distanceThreshold)
{
    auto startTime = std::chrono::steady_clock::now();

    // --- Call your custom RANSAC PLANE algorithm ---
    std::unordered_set<int> inliersSet =
        RansacPlane(cloud, maxIterations, distanceThreshold);

    if (inliersSet.size() == 0)
    {
        std::cerr << "No plane found." << std::endl;
        return {};
    }

    // --- Convert unordered_set → pcl::PointIndices ---
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    for (int idx : inliersSet)
        inliers->indices.push_back(idx);

    // --- Separate the cloud into plane & obstacles ---
    auto result = SeparateClouds(inliers, cloud);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Custom RANSAC plane segmentation took "
        << elapsedTime.count() << " ms" << std::endl;

    return result;
}



template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // Containers for plane coefficients and inliers
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    // Setup RANSAC plane segmentation
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);


    if (inliers->indices.size() == 0) {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        return std::pair<typename pcl::PointCloud<PointT>::Ptr,
            typename pcl::PointCloud<PointT>::Ptr>();
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers, cloud);
    return segResult;
}


template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // Create KD-Tree object for the search method
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud);

    // Euclidean clustering object
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance);  // distance tolerance in meters
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusterIndices;
    ec.extract(clusterIndices);

    // Convert cluster indices into point cloud clusters
    for (const pcl::PointIndices& indices : clusterIndices)
    {
        typename pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>());
        for (int idx : indices.indices)
        {
            cluster->points.push_back(cloud->points[idx]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;

        clusters.push_back(cluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "clustering took " << elapsedTime.count()
        << " milliseconds and found " << clusters.size()
        << " clusters" << std::endl;

    return clusters;
}

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::ClusteringCustom(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{
    auto startTime = std::chrono::steady_clock::now();
    // -------------------------------
    // 1. Convert PCL -> vector<float>
    // -------------------------------
    std::vector<std::vector<float>> points;
    points.reserve(cloud->points.size());

    for (auto& p : cloud->points)
        points.push_back({ p.x, p.y, p.z });

    // -------------------------------
    // 2. Build KD-Tree
    // -------------------------------
    KdTreeCustom* tree = new KdTreeCustom();
    tree->buildTree(points);


    // -------------------------------
    // 3. Perform custom clustering
    // -------------------------------
    std::vector<std::vector<int>> clusterIndices =
        euclideanCluster(points, tree, clusterTolerance);

    // -------------------------------
    // 4. Convert indices -> PCL clusters
    // -------------------------------
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    for (const auto& indices : clusterIndices)
    {
        if (indices.size() < minSize || indices.size() > maxSize)
            continue;

        typename pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>());

        for (int idx : indices)
            cluster->points.push_back(cloud->points[idx]);

        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;

        clusters.push_back(cluster);
    }


    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "clustering took " << elapsedTime.count()
        << " milliseconds and found " << clusters.size()
        << " clusters" << std::endl;

    return clusters;
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII(file, *cloud);
    std::cerr << "Saved " << cloud->points.size() << " data points to " + file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size() << " data points from " + file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{ dataPath }, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr,
    typename pcl::PointCloud<PointT>::Ptr>
    SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud,
        int maxIterations,
        float distanceThreshold)
{
    // Time segmentation
    auto startTime = std::chrono::steady_clock::now();

    // Containers for plane coefficients and inliers
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    // Setup RANSAC plane segmentation
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0) {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        return std::pair<typename pcl::PointCloud<PointT>::Ptr,
            typename pcl::PointCloud<PointT>::Ptr>();
    }

    // Extract plane (inliers)
    typename pcl::PointCloud<PointT>::Ptr planeCloud(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr obstacleCloud(new pcl::PointCloud<PointT>());

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);

    // Extract plane points
    extract.setNegative(false);
    extract.filter(*planeCloud);

    // Extract everything else (obstacles)
    extract.setNegative(true);
    extract.filter(*obstacleCloud);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Plane point count: " << planeCloud->size() << std::endl;
    std::cout << "Obstacle point count: " << obstacleCloud->size() << std::endl;
    std::cout << "Manu lane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    return std::make_pair(obstacleCloud, planeCloud);
}


template<typename PointT>
std::unordered_set<int> ProcessPointClouds<PointT>::RansacPlane(typename pcl::PointCloud<PointT>::Ptr cloud,
    int maxIterations,
    float distanceTol)
{
    std::unordered_set<int> bestInliersResult;
    srand(time(NULL));

    int cloudSize = cloud->points.size();

    for (int i = 0; i < maxIterations; i++)
    {
        std::unordered_set<int> inliers;

        while (inliers.size() < 3)
            inliers.insert(rand() % cloudSize);

        auto it = inliers.begin();
        int idx1 = *it; it++;
        int idx2 = *it; it++;
        int idx3 = *it;

        PointT p1 = cloud->points[idx1];
        PointT p2 = cloud->points[idx2];
        PointT p3 = cloud->points[idx3];

        float v1x = p2.x - p1.x;
        float v1y = p2.y - p1.y;
        float v1z = p2.z - p1.z;

        float v2x = p3.x - p1.x;
        float v2y = p3.y - p1.y;
        float v2z = p3.z - p1.z;

        float A = v1y * v2z - v1z * v2y;
        float B = v1z * v2x - v1x * v2z;
        float C = v1x * v2y - v1y * v2x;
        float D = -(A * p1.x + B * p1.y + C * p1.z);

        float norm = sqrt(A * A + B * B + C * C);
        if (norm == 0)
            continue;

        for (int j = 0; j < cloudSize; j++)
        {
            float distance =
                fabs(A * cloud->points[j].x +
                    B * cloud->points[j].y +
                    C * cloud->points[j].z + D) / norm;

            if (distance < distanceTol)
                inliers.insert(j);
        }

        if (inliers.size() > bestInliersResult.size())
            bestInliersResult = inliers;
    }

    return bestInliersResult;
}

template<typename PointT>
void ProcessPointClouds<PointT>::proximity(int idx,
    const std::vector<std::vector<float>>& points,
    std::vector<bool>& processed,
    std::vector<int>& cluster,
    KdTreeCustom* tree,
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

template<typename PointT>
std::vector<std::vector<int>> ProcessPointClouds<PointT>::euclideanCluster(const std::vector<std::vector<float>>& points, KdTreeCustom* tree, float distanceTol)
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

