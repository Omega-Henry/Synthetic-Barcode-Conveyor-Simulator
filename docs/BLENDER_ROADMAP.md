# Blender 3D Renderer Integration Roadmap

This roadmap documents the architectural strategy for extending the Synthetic Industrial Barcode Conveyor-Belt Simulator to a photorealistic **Blender 3D (Cycles / EEVEE)** rendering backend without modifying the core simulation logic, barcode generation, trajectories, or ground-truth evaluation pipelines.

---

## 1. Architectural Separation

The simulator is built strictly around the **Interface Segregation** principle:

```text
               +----------------------------------+
               |     Simulation Configuration     |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |        Simulation Engine         |
               | (Trajectories, Barcodes, Pkgs)   |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |         SceneDescription         |
               +----------------------------------+
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
        +-----------------+           +-----------------+
        | OpenCV2DRenderer|           | BlenderRenderer |
        |   (Implemented) |           |  (Future Ext)   |
        +-----------------+           +-----------------+
                 |                             |
                 +--------------+--------------+
                                |
                                v
               +----------------------------------+
               |       RGB Frames & Video         |
               +----------------------------------+
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
        +-----------------+           +-----------------+
        |  Ground Truth   |           | Barcode Scanner |
        | (JSONL/COCO/YOLO)           | & Evaluation    |
        +-----------------+           +-----------------+
```

The `Renderer` abstract base class in `src/barcode_simulator/renderers/base.py` defines the contract:

```python
class Renderer(ABC):
    @abstractmethod
    def initialize(self, scene: SceneDescription) -> None:
        pass

    @abstractmethod
    def render_frame(self, scene: SceneDescription) -> np.ndarray:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass
```

The future `BlenderRenderer` simply implements this same class.

---

## 2. 2D to 3D Entity Mapping

| 2D Domain Entity (`core/models.py`) | Blender 3D Counterpart (`bpy`) | Physical / Material Representation |
| :--- | :--- | :--- |
| **Package** (`width`, `height`, `depth`) | `bpy.ops.mesh.primitive_cube_add()` | 3D Box mesh scaled to $(W, H, D)$ in meters, with bevel modifier for rounded corners. |
| **Material** (`cardboard`, `carton`) | Principled BSDF Node Shader | Roughness (0.85), Normal Map (corrugation/fiber), Base Color. |
| **Barcode** (`image_np`) | UV Decal Layer / Image Texture | Barcode PNG loaded into an `ImageTexture` node mapped to the top/front face UV coordinates. |
| **Conveyor Belt** | Flat extruded Mesh + Curve Modifier | Animated conveyor texture offset via Mapping node or moving belt vertices along a curve. |
| **Camera** (`width`, `height`, `zoom`) | `bpy.data.cameras.new()` | Perspective camera with focal length $f = 35\text{mm}$ or $50\text{mm}$, sensor width $36\text{mm}$, depth of field. |
| **Trajectory** ($x(t), y(t)$) | Keyframed Object Location or Rigid Body | Package position $(X(t), Y(t), Z)$ keyframed per frame along conveyor axis. |
| **Lighting** (glare, ambient) | Area Lights + Point Spotlights + HDRI | Overhead rectangular industrial fluorescent tubes with raytraced soft shadows and specular highlights. |

---

## 3. Barcode Texture & Decal Projection in Blender

Because all barcodes in our system are programmatically generated and structurally verified as PNG images, no AI generation is used. In Blender:

1. The package cuboid is UV-unwrapped.
2. A Principled BSDF shader is created with two layers mixed via an alpha mask:
   - **Layer A**: Cardboard / Carton procedural base material.
   - **Layer B**: The high-contrast barcode decal texture positioned at $(u_{\text{rel}}, v_{\text{rel}})$ with scale $s_{\text{rel}}$.
3. The barcode retains its machine-readable contrast while benefiting from physically based rendering (PBR) lighting, shadows, and raytraced reflections.

---

## 4. 3D Camera Projection & Ground Truth Alignment

Ground-truth 2D bounding boxes and 4-corner polygons are computed in Blender by projecting 3D object vertices through the camera's Model-View-Projection matrix:

$$\mathbf{p}_{\text{screen}} = \mathbf{P}_{\text{camera}} \cdot \mathbf{V}_{\text{view}} \cdot \mathbf{M}_{\text{model}} \cdot \mathbf{p}_{\text{3D}}$$

In Blender Python (`bpy_extras`):

```python
import bpy_extras

def get_2d_barcode_polygon(scene, camera, barcode_obj):
    coords_3d = [barcode_obj.matrix_world @ v.co for v in barcode_obj.data.vertices]
    coords_2d = [
        bpy_extras.object_utils.world_to_camera_view(scene, camera, coord)
        for coord in coords_3d
    ]
    # Convert normalized [0, 1] viewport coordinates to pixel coordinates
    render = scene.render
    w, h = render.resolution_x, render.resolution_y
    return [[int(c.x * w), int((1.0 - c.y) * h)] for c in coords_2d]
```

This guarantees that the ground-truth 4-point polygon and bounding box generated from Blender matches the 2D pipeline format identically.

---

## 5. Headless Blender Execution Architecture

Blender can be executed headlessly via Python subprocess:

```bash
blender --background --factory-startup --python render_blender.py -- \
    --scene-data /tmp/scene_dump.json \
    --output-dir /path/to/outputs
```

Or using the `bpy` standalone Python package (`pip install bpy`) within the virtual environment.

---

## 6. Implementation Milestones for Phase 2 (Blender 3D)

1. **Step 1**: Create `renderers/blender_renderer.py` implementing `Renderer`.
2. **Step 2**: Create procedural Blender studio scene builder (`assets/blender/conveyor_scene.blend`).
3. **Step 3**: Implement material shader graph builder (Cardboard, Carton, Plastic).
4. **Step 4**: Implement camera projection helper to compute 2D bounding boxes and polygons from 3D meshes.
5. **Step 5**: Enable EEVEE (fast interactive rendering) and Cycles (photorealistic raytracing with motion blur).
6. **Step 6**: Validate downstream compatibility with `scripts/verify_dataset.py` and `BaselineScanner`.
