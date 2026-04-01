#include <fstream>
#include <iostream>
#include <rasterizer.h>
#include <sstream>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "json.hpp"
#include "stb_image_write.h"
#include <cuda_runtime_api.h>

struct Vector3f {
  float vals[3];
};
typedef Vector3f Pos;
template <int D> struct SHs {
  float shs[(D + 1) * (D + 1) * 3];
};
struct Scale {
  float scale[3];
};
struct Rot {
  float rot[4];
};
template <int D> struct RichPoint {
  Pos pos;
  float n[3];
  SHs<D> shs;
  float opacity;
  Scale scale;
  Rot rot;
};

float sigmoid(const float m1) { return 1.0f / (1.0f + exp(-m1)); }

// Load the Gaussians from the given file.
template <int D>
int loadPly(std::string filename, std::vector<Pos> &pos,
            std::vector<SHs<D>> &shs, std::vector<float> &opacities,
            std::vector<Scale> &scales, std::vector<Rot> &rot) {
  std::ifstream infile(filename, std::ios_base::binary);

  if (!infile.good())
    throw std::runtime_error("Unable to find model's PLY file");

  // "Parse" header (it has to be a specific format anyway)
  std::string buff;
  std::getline(infile, buff);
  std::getline(infile, buff);

  std::string dummy;
  std::getline(infile, buff);
  std::stringstream ss(buff);
  int count;
  ss >> dummy >> dummy >> count;

  // Output number of Gaussians contained
  std::cout << "Loading " << count << " Gaussian splats" << std::endl;

  while (std::getline(infile, buff))
    if (buff.compare("end_header") == 0)
      break;

  // Read all Gaussians at once (AoS)
  std::vector<RichPoint<D>> points(count);
  infile.read((char *)points.data(), count * sizeof(RichPoint<D>));

  // Resize our SoA data
  pos.resize(count);
  shs.resize(count);
  scales.resize(count);
  rot.resize(count);
  opacities.resize(count);

  // Move data from AoS to SoA
  int SH_N = (D + 1) * (D + 1);
  for (int k = 0; k < count; k++) {
    int i = k;
    pos[k] = points[i].pos;

    // Normalize quaternion
    float length2 = 0;
    for (int j = 0; j < 4; j++)
      length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
    float length = sqrt(length2);
    for (int j = 0; j < 4; j++)
      rot[k].rot[j] = points[i].rot.rot[j] / length;

    // Exponentiate scale
    for (int j = 0; j < 3; j++)
      scales[k].scale[j] = exp(points[i].scale.scale[j]);

    // Activate alpha
    opacities[k] = sigmoid(points[i].opacity);

    shs[k].shs[0] = points[i].shs.shs[0];
    shs[k].shs[1] = points[i].shs.shs[1];
    shs[k].shs[2] = points[i].shs.shs[2];
    for (int j = 1; j < SH_N; j++) {
      shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
      shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
      shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
    }
  }
  return count;
}

std::function<char *(size_t N)> resizeFunctional(void **ptr, size_t &S) {
  auto lambda = [ptr, &S](size_t N) {
    if (N > S) {
      if (*ptr)
        cudaFree(*ptr);
      cudaMalloc(ptr, 2 * N);
      S = 2 * N;
    }
    return reinterpret_cast<char *>(*ptr);
  };
  return lambda;
}

struct InputCamera {
  float fx, fy;
  int width, height;
  float tanfovx, tanfovy;
  Vector3f pos;
  Vector3f rot[3];
};

std::vector<InputCamera> loadJSON(const std::string &jsonPath) {
  std::ifstream json_file(jsonPath, std::ios::in);

  if (!json_file)
    throw std::runtime_error("camera file loading failed");

  std::vector<InputCamera> cameras;

  nlohmann::json jsonData = nlohmann::json::parse(json_file);

  for (auto cam : jsonData) {
    InputCamera inputCam;
    inputCam.width = cam["width"].get<double>();
    inputCam.height = cam["height"].get<double>();
    inputCam.fy = cam["fy"].get<double>();
    inputCam.fx = cam["fx"].get<double>();

    int i = 0;
    for (auto coord : cam["position"])
      inputCam.pos.vals[i++] = coord;

    inputCam.tanfovy = (0.5f * inputCam.height / inputCam.fy);
    inputCam.tanfovx = (0.5f * inputCam.width / inputCam.fx);

    i = 0;
    for (auto r : cam["rotation"]) {
      int j = 0;
      for (auto coord : r)
        inputCam.rot[i].vals[j++] = coord;
      i++;
    }
    cameras.push_back(inputCam);
  }

  return cameras;
}

void projMatrix(float tanfovy, float aspect, float *target) {
  const float yScale = 1.0f / tanfovy;
  const float xScale = yScale / aspect;

  float zn = 0.01f;
  float zf = 1000.0f;
  float vals[] = {xScale,
                  0,
                  0,
                  0,
                  0,
                  yScale,
                  0,
                  0,
                  0,
                  0,
                  -(zn + zf) / (zn - zf),
                  -2 * zn * zf / (zn - zf),
                  0,
                  0,
                  1,
                  0};

  for (int i = 0; i < 16; i++)
    target[i] = vals[i];
}

int main(int argc, char *argv[]) {
  if (argc < 3)
    throw std::runtime_error(
        "Please provide a 3DGS scene dir and cam index as argument");

  std::vector<InputCamera> cameras =
      loadJSON(std::string(argv[1]) + "/cameras.json");

  int index = std::atoi(argv[2]);

  InputCamera cam = cameras[index];

  int WIDTH = 1200;
  int HEIGHT = cam.height / ((float)cam.width) * WIDTH;

  void *geomPtr = nullptr;
  size_t allocdGeom = 0;
  auto geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);

  void *binningPtr = nullptr;
  size_t allocdBinning = 0;
  auto binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);

  void *imgPtr = nullptr;
  size_t allocdImg = 0;
  auto imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);

  std::vector<Pos> pos;
  std::vector<SHs<3>> shs;
  std::vector<float> opacities;
  std::vector<Scale> scales;
  std::vector<Rot> rots;
  size_t num =
      loadPly(std::string(argv[1]) +
                  std::string("/point_cloud/iteration_30000/point_cloud.ply"),
              pos, shs, opacities, scales, rots);

  float *background_cuda;
  float background_color[] = {0, 0, 0};
  cudaMalloc((void **)&background_cuda, sizeof(float) * 3);
  cudaMemcpy(background_cuda, background_color, sizeof(float) * 3,
             cudaMemcpyHostToDevice);

  float *pos_cuda;
  cudaMalloc((void **)&pos_cuda, sizeof(Pos) * num);
  cudaMemcpy(pos_cuda, pos.data(), sizeof(Pos) * num, cudaMemcpyHostToDevice);

  float *shs_cuda;
  cudaMalloc((void **)&shs_cuda, sizeof(SHs<3>) * num);
  cudaMemcpy(shs_cuda, shs.data(), sizeof(SHs<3>) * num,
             cudaMemcpyHostToDevice);

  float *opacities_cuda;
  cudaMalloc((void **)&opacities_cuda, sizeof(float) * num);
  cudaMemcpy(opacities_cuda, opacities.data(), sizeof(float) * num,
             cudaMemcpyHostToDevice);

  float *scales_cuda;
  cudaMalloc((void **)&scales_cuda, sizeof(Scale) * num);
  cudaMemcpy(scales_cuda, scales.data(), sizeof(Scale) * num,
             cudaMemcpyHostToDevice);

  float *rotations_cuda;
  cudaMalloc((void **)&rotations_cuda, sizeof(Rot) * num);
  cudaMemcpy(rotations_cuda, rots.data(), sizeof(Rot) * num,
             cudaMemcpyHostToDevice);

  float *img_cuda;
  cudaMalloc((void **)&img_cuda, sizeof(float) * 3 * WIDTH * HEIGHT);

  float *campos_cuda;
  cudaMalloc((void **)&campos_cuda, sizeof(float) * 3);
  cudaMemcpy(campos_cuda, &cam.pos, sizeof(float) * 3, cudaMemcpyHostToDevice);

  Vector3f transpose[3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      transpose[j].vals[i] = cam.rot[i].vals[j];

  Vector3f invpos = {0};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      invpos.vals[i] += transpose[i].vals[j] * -cam.pos.vals[j];
  }

  float viewmat[16] = {0};
  viewmat[15] = 1.0f;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      if (j == 3)
        viewmat[i * 4 + j] = invpos.vals[i];
      else
        viewmat[i * 4 + j] = transpose[i].vals[j];

  float projmat[16];
  projMatrix(cam.tanfovy, cam.width / ((float)cam.height), projmat);

  float viewprojmat[16] = {0};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++)
        viewprojmat[i * 4 + j] += projmat[i * 4 + k] * viewmat[k * 4 + j];

  float viewmat_transpose[16];
  float viewprojmat_transpose[16];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      viewmat_transpose[i * 4 + j] = viewmat[j * 4 + i];
      viewprojmat_transpose[i * 4 + j] = viewprojmat[j * 4 + i];
    }

  float *viewmat_cuda;
  cudaMalloc((void **)&viewmat_cuda, sizeof(float) * 16);
  cudaMemcpy(viewmat_cuda, viewmat_transpose, sizeof(float) * 16,
             cudaMemcpyHostToDevice);

  float *viewprojmat_cuda;
  cudaMalloc((void **)&viewprojmat_cuda, sizeof(float) * 16);
  cudaMemcpy(viewprojmat_cuda, viewprojmat_transpose, sizeof(float) * 16,
             cudaMemcpyHostToDevice);

  CudaRasterizer::Rasterizer::forward(
      geomBufferFunc, binningBufferFunc, imgBufferFunc, num, 3, 16,
      background_cuda, WIDTH, HEIGHT, pos_cuda, shs_cuda, nullptr,
      opacities_cuda, scales_cuda, 1.0f, rotations_cuda, nullptr, viewmat_cuda,
      viewprojmat_cuda, campos_cuda, cam.tanfovx, cam.tanfovy, false, img_cuda);

  std::vector<float> image_cpu(WIDTH * HEIGHT * 3);
  cudaMemcpy(image_cpu.data(), img_cuda, sizeof(float) * 3 * WIDTH * HEIGHT,
             cudaMemcpyDeviceToHost);

  std::vector<unsigned char> img_cpu(WIDTH * HEIGHT * 3);
  for (int c = 0; c < 3; c++)
    for (int i = 0; i < WIDTH * HEIGHT; i++)
      img_cpu[i * 3 + c] =
          (unsigned char)(image_cpu[c * WIDTH * HEIGHT + i] * 255);

  if (stbi_write_png((std::string(argv[2]) + std::string(".png")).c_str(),
                     WIDTH, HEIGHT, 3, img_cpu.data(), WIDTH * 3) == 0) {
    throw std::runtime_error("Failed to write image");
  }

  std::cout << "Done, wrote image" << std::endl;
}