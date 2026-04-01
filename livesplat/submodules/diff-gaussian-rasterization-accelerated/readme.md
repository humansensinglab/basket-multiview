# Accelerated Differentiable Gaussian Rasterization

High-performance rasterization engine extending [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) with three key optimizations for static scene appearance training:

- **Precomputed ordering**: Depth sorting is performed once after geometry converges and reused across frames, eliminating the primary rasterization bottleneck.
- **Load-balanced tiling**: Tiles with many Gaussians are subdivided into fixed-size subtiles processed by separate thread blocks, maximizing GPU occupancy.
- **CUDA graph capture**: The full forward and backward pass is captured as a static CUDA graph, eliminating CPU-GPU synchronization overhead and kernel launch latency during training.

Together these optimizations achieve from 2x up to 4x speedup over standard 3DGS training for static appearance optimization.

Used as the rasterization engine for the paper "Echoes of the Coliseum: Towards 3D Live Streaming of Sports Events". If you make use of it in your own research, please cite our work and the original 3DGS:
```bibtex
@Article{huang2025echoes,
      author       = {Huang, Junkai and Mallick, Saswat Subhajyoti and Amat, Alejandro and Ruiz Olle, Marc and Mosella-Montoro, Albert and Kerbl, Bernhard and Vicente Carrasco, Francisco and De la Torre, Fernando},
      title        = {Echoes of the Coliseum: Towards 3D Live Streaming of Sports Events},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {44},
      month        = {August},
      year         = {2025},
      doi          = {10.1145/3731214}
}

@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```