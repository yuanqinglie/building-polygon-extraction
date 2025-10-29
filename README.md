
# APOFormer: Adaptive Primitive Optimization Transformer for Building Polygon Extraction from remote sensing imagery

APOFormer is a deep learning-based model for building polygon extraction. It can automatically extract the polygonal contours of buildings from remote sensing images, suitable for scenarios such as urban planning and map updating.

## Model Architecture

The model mainly consists of the following core modules:



1. **Encoder**: Uses ResNet50 as the backbone network to extract multi-scale image features

2. **CFPE Module**: Crossing-Feature Primitives Enhancement

3. **DQO Module**: Dynamic Query Optimizer

4. **MVDA Module**: Multi-vertices Collaborative Deformable Attention

5. **ECTR Module**: Edge-context-guided Topology Reconstruction

### Network Flowchart



```
Input Image → ResNet50 Encoder → Feature Adaptation Layers → CFPE Module → Zf Features

&#x20;                                                              ↓

&#x20;                       DQO Module → Initial Queries Q and Vertex Coordinates Drc ← L4 Features

&#x20;                                                              ↓

&#x20;                     MVDA Layers (Iterative Optimization) → Updated Q and Drc

&#x20;                                                              ↓

&#x20;                       ECTR Module → Adjacency Matrix SA

&#x20;                                                              ↓

&#x20;                    Output → Vertex Coordinates, Adjacency Matrix, Vertex Confidence
```

## Core Module Details

### 1. CFPE Module

Cross-Feature Primitives Enhancement module, responsible for fusing features of different levels:



* Unifies the number of channels across different feature layers

* Fuses shallow features (L2+L3) and deep features (L4+L5)

* Computes feature affinity matrix to realize interactive enhancement of shallow and deep features

* Extracts corner and edge features to enhance contour perception capability

### 2. DQO Module

Dynamic Query Optimizer, generating initial queries and vertex coordinates:



* Adds positional encoding to enhance spatial perception

* Predicts candidate vertex coordinates and instance scores

* Selects Top-K query positions based on spatial score map

* Generates initial query embeddings and vertex coordinates

### 3. MVDA Module

Multi-vertices Collaborative Deformable Attention module:



* Self-attention mechanism realizes positional interaction between vertices

* Bilinear interpolation sampling for vertex features

* ROI Align to extract region-of-interest features

* Cross-attention fuses multi-source features to optimize vertex coordinates

### 4. ECTR Module

Edge-context-guided Topology Reconstruction module:



* Predicts edge confidence and filters key edges

* Multi-scale sampling of edge context features

* Attention mechanism optimizes edge features

* Predicts adjacency matrix to construct polygonal topology

## Usage

### Model Initialization



```
model = BuildingPolygonNet(

&#x20;   encoder\_type="resnet50",

&#x20;   encoder\_channels=\[256, 512, 1024, 2048],

&#x20;   d\_model=256,

&#x20;   num\_queries=100,

&#x20;   num\_vertices=8,

&#x20;   top\_e=16,

&#x20;   pretrained=True

)
```

### Image Preprocessing



```
input\_tensor, raw\_image = preprocess\_image(

&#x20;   image\_path="path/to/image.jpg",&#x20;

&#x20;   img\_size=(512, 512)

)
```

### Model Inference



```
model.eval()

with torch.no\_grad():

&#x20;   vertices, adj\_matrix, confidence = model(input\_tensor)
```

### Result Post-processing and Visualization



```
\# Map coordinates back to original image size

vertices = postprocess\_vertices(vertices)

\# Visualize results

visualize\_results(

&#x20;   raw\_image,&#x20;

&#x20;   vertices,&#x20;

&#x20;   confidence,&#x20;

&#x20;   adj\_matrix,&#x20;

&#x20;   threshold=0.5,&#x20;

&#x20;   top\_k=3

)
```

## Test Script



```
if \_\_name\_\_ == "\_\_main\_\_":

&#x20;   test\_image\_path = "path/to/your/test/image.tif"

&#x20;   checkpoint\_path = "path/to/pretrained/model.pth"  # Optional

&#x20;  &#x20;

&#x20;   results = test\_model(test\_image\_path, checkpoint\_path)
```

## Output Explanation

The model output includes three parts:



1. **Vertex Coordinates**: Shape is (B, K, Nv, 2), where B is batch size, K is number of queries, and Nv is number of vertices per building

2. **Adjacency Matrix**: Shape is (B, K, Nv, Nv), representing the connection relationship between vertices

3. **Vertex Confidence**: Shape is (B, K, Nv, 1), representing the confidence score of each vertex

## Dependencies



* PyTorch >= 1.7.0

* torchvision >= 0.8.1

* numpy >= 1.19.5

* Pillow >= 8.0.1

* matplotlib >= 3.3.3

## Notes



* The recommended input image size is 512x512

* The model supports CUDA acceleration and will automatically use GPU if available

* Pretrained models can significantly improve performance; it is recommended to load pretrained weights

* The visualization will display the top 3 building instances with the highest confidence

You can control the number of vertices of the output polygon by adjusting the `num_vertices` parameter, and control the maximum number of detected buildings by adjusting the `num_queries` parameter.

