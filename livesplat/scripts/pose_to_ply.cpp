#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cfloat>
#include <cmath>
#include <list>

template<int D>
struct SHs
{
	float shs[(D + 1) * (D + 1) * 3];
};

struct Scale
{
	float scale[3];
};

struct Rot
{
	float rot[4];
};

typedef Scale Pos;

template<int D>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
};

// Load the Gaussians from the given file.
int loadPly(const char* filename, std::vector<Pos>&pos)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good()) {
		std::cout << "Error: Unable to find model's PLY file: " << filename << std::endl;
        return 0;
    }

	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	std::vector<RichPoint<1>> points(count);
	infile.read((char*)points.data(), count * sizeof(RichPoint<1>));

	pos.resize(count);

	for (int k = 0; k < count; k++)
	{
		pos[k] = points[k].pos;
	}
	return count;
}

int loadObj(const char* filename, std::vector<Pos>& pos, float scalefactor)
{
	std::ifstream infile(filename);
    if (!infile.good()) {
        std::cout << "Error: Unable to find OBJ file: " << filename << std::endl;
        return 0;
    }

	std::string line;
	while (std::getline(infile, line))
	{
		if (line.length() > 2 && line[0] == 'v' && line[1] == ' ')
		{
			std::stringstream ss(line.substr(1));
			float x, y, z;
			ss >> x >> y >> z;
			pos.push_back({ scalefactor * x, scalefactor * y, scalefactor * z });
		}
	}
	return pos.size();
}

int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cout << "Usage: ./pose_to_ply <input_obj_path> <input_ply_path> <output_assoc_path>" << std::endl;
        return 1;
    }

    std::string objfile = argv[1];
    std::string plyfile = argv[2];
    std::string assocfile = argv[3];

    std::cout << "Processing Association..." << std::endl;
    std::cout << "  OBJ: " << objfile << std::endl;
    std::cout << "  PLY: " << plyfile << std::endl;

    std::vector<Pos> plypos;
    std::vector<Pos> objpos;
    
    int ply_count = loadPly(plyfile.c_str(), plypos);
    int obj_count = loadObj(objfile.c_str(), objpos, 1.0f);

    if (ply_count == 0 || obj_count == 0) {
        std::cout << "Error: Empty point cloud or mesh." << std::endl;
        return 1;
    }

    std::ofstream assoc(assocfile);
    if (!assoc.is_open()) {
        std::cout << "Error: Could not open output file " << assocfile << std::endl;
        return 1;
    }

    for (size_t i = 0; i < plypos.size(); i++)
    {
        Pos ply = plypos[i];
        float mindist = FLT_MAX;
        int bestind = 0;
        
        // Simple nearest neighbor search (O(N*M))
        for (size_t j = 0; j < objpos.size(); j++)
        {
            Pos obj = objpos[j];

            Pos diff = {
                ply.scale[0] - obj.scale[0],
                ply.scale[1] - obj.scale[1],
                ply.scale[2] - obj.scale[2]
            };

            float dist2 = diff.scale[0] * diff.scale[0] + 
                          diff.scale[1] * diff.scale[1] + 
                          diff.scale[2] * diff.scale[2];

            if (dist2 < mindist)
            {
                mindist = dist2;
                bestind = j;
            }
        }
        assoc << bestind << ((i == (plypos.size() - 1)) ? "" : "\n");
    }

    std::cout << "Association saved to: " << assocfile << std::endl;
    return 0;
}