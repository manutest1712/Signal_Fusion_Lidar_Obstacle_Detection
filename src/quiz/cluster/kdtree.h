/* \author Aaron Brown */
// Quiz on implementing kd tree

#include "../../render/render.h"

typedef unsigned int uint;


// Structure to represent node of kd tree
struct Node
{
	std::vector<float> point;
	int id;
	Node* left;
	Node* right;

	Node(std::vector<float> arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}

	~Node()
	{
		delete left;
		delete right;
	}
};

struct KdTree
{
	Node* root;

	KdTree()
	: root(NULL)
	{}

	~KdTree()
	{
		delete root;
	}

	// Helper function to insert recursively
	void insertHelper(Node*& node, uint depth, const std::vector<float>& point, int id)
	{
		if (node == NULL)
		{
			node = new Node(point, id);
		}
		else
		{
			uint cd = depth % point.size(); // current dimension index

			if (point[cd] < node->point[cd])
				insertHelper(node->left, depth + 1, point, id);
			else
				insertHelper(node->right, depth + 1, point, id);
		}
	}


	void insert(std::vector<float> point, int id)
	{
		insertHelper(root, 0, point, id);
	}

	// Build balanced tree
	Node* buildBalanced(std::vector<std::vector<float>>& points,
		std::vector<int>& ids,
		int start, int end,
		int depth)
	{
		if (start >= end)
			return nullptr;

		int k = points[0].size();   // dimension
		int axis = depth % k;       // current split axis

		// Sort indices [start, end) based on axis
		std::sort(ids.begin() + start, ids.begin() + end,
			[&points, axis](int a, int b)
			{
				return points[a][axis] < points[b][axis];
			});

		int mid = start + (end - start) / 2;

		Node* node = new Node(points[ids[mid]], ids[mid]);

		// Recursive build
		node->left = buildBalanced(points, ids, start, mid, depth + 1);
		node->right = buildBalanced(points, ids, mid + 1, end, depth + 1);

		return node;
	}

	// Public interface to build tree
	void buildTree(std::vector<std::vector<float>>& points)
	{
		std::vector<int> ids(points.size());
		for (int i = 0; i < points.size(); i++)
			ids[i] = i;

		root = buildBalanced(points, ids, 0, points.size(), 0);
	}


	// Helper for search
	void searchHelper(Node* node,
		const std::vector<float>& target,
		float distanceTol,
		uint depth,
		std::vector<int>& ids)
	{
		if (node == NULL)
			return;

		bool insideBox = true;
		for (size_t i = 0; i < target.size(); i++)
		{
			if (node->point[i] < (target[i] - distanceTol) ||
				node->point[i] > (target[i] + distanceTol))
			{
				insideBox = false;
				break;
			}
		}

		if (insideBox)
		{
			// Compute Euclidean distance
			float dist = 0.0f;
			for (size_t i = 0; i < target.size(); i++)
				dist += (node->point[i] - target[i]) * (node->point[i] - target[i]);

			if (sqrt(dist) <= distanceTol)
				ids.push_back(node->id);
		}

		uint cd = depth % target.size();

		// Explore left or right branch if needed
		if ((target[cd] - distanceTol) < node->point[cd])
			searchHelper(node->left, target, distanceTol, depth + 1, ids);

		if ((target[cd] + distanceTol) > node->point[cd])
			searchHelper(node->right, target, distanceTol, depth + 1, ids);
	}


	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(std::vector<float> target, float distanceTol)
	{
		std::vector<int> ids;
		searchHelper(root, target, distanceTol, 0, ids);
		return ids;
	}
	

};




