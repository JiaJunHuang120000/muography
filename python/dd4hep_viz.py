#!/usr/bin/env python3
"""
Detector Visualizer & Voxelizer
Specialized for the specific XML structure you provided
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import sys
import os
import pickle
import re
from pathlib import Path

def parse_dd4hep_constant(value_str):
    """Parse DD4hep constant values like '10*cm', '5*m', etc."""
    # Remove whitespace
    value_str = str(value_str).strip()
    
    # Unit conversion factors to mm
    units = {
        'mm': 1.0,
        'cm': 10.0,
        'm': 1000.0,
        'inch': 25.4,
        'mil': 0.0254,
        'deg': 1.0,  # for angles
        'rad': 57.2957795  # radians to degrees
    }
    
    # Handle mathematical expressions
    if '*' in value_str:
        parts = value_str.split('*')
        if len(parts) == 2:
            try:
                num = float(parts[0])
                unit = parts[1].lower()
                
                # Check for unit in dictionary
                for key in units:
                    if key in unit:
                        return num * units[key]
                
                # If no unit found, return the number
                return num
            except:
                return float(parts[0]) if parts[0].replace('.', '').isdigit() else 0
    else:
        try:
            return float(value_str)
        except:
            return 0


def read_detectors_from_config_file():
    """
    Read detector positions directly from config.sh (bash arrays).
    Units: meters → millimeters
    """
    detector_path = os.getenv("DETECTOR_PATH")
    if detector_path is None:
        raise RuntimeError("DETECTOR_PATH is not set")

    cfg = Path(detector_path) / "bash" / "config.sh"
    if not cfg.exists():
        raise FileNotFoundError(f"Config file not found: {cfg}")

    text = cfg.read_text()

    def parse_array(var):
        m = re.search(rf"{var}\s*=\s*\(([^)]*)\)", text)
        if not m:
            raise RuntimeError(f"Variable {var} not found in config.sh")
        return [float(x) for x in m.group(1).split()]

    xs = parse_array("detector_pos_x")
    ys = parse_array("detector_pos_y")
    zs = parse_array("detector_pos_z")

    if not (len(xs) == len(ys) == len(zs)):
        raise RuntimeError("detector_pos arrays have inconsistent lengths")

    detectors = []
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        detectors.append({
            "id": i,
            "name": f"detector_{i}",
            "position_mm": {
                "x": x * 1000.0,
                "y": y * 1000.0,
                "z": z * 1000.0,
            },
        })

    return detectors


class DetectorParser:
    def __init__(self, xml_file, resolution=64):
        self.xml_file = xml_file
        self.resolution = resolution   
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        
        # Parse constants first
        self.constants = self._parse_constants()
        
        # Parse detector geometry
        self.detectors = self._parse_detectors()
        
        # Parse world dimensions
        self.world = self._parse_world()
    
    def _parse_constants(self):
        """Parse all <constant> definitions"""
        constants = {}
        
        for const in self.root.findall('.//constant'):
            name = const.get('name', '')
            value = const.get('value', '0')
            
            if name:
                constants[name] = parse_dd4hep_constant(value)
        
        # Also parse from <define><constant> structure
        for define in self.root.findall('.//define'):
            for const in define.findall('constant'):
                name = const.get('name', '')
                value = const.get('value', '0')
                if name:
                    constants[name] = parse_dd4hep_constant(value)
        
        return constants
    
    def _parse_world(self):
        """Parse world volume"""
        world_elem = self.root.find('world')
        pixel_size = float(os.getenv('pixel_size')) # Meters
        world_size = int(pixel_size*int(self.resolution)*1000)
        if world_elem is None:
            return {
                'material': 'Vacuum',
                'shape': 'Box',
                'dimensions': {'dx': world_size, 'dy': world_size, 'dz': world_size},
                'position': {'x': 0, 'y': 0, 'z': 0}
            }
        
        shape = world_elem.find('shape')
        dimensions = {}
        
        if shape is not None:
            for dim in ['dx', 'dy', 'dz']:
                value = shape.get(dim, '0')
                # Replace constants with their values
                for const_name, const_value in self.constants.items():
                    value = value.replace(const_name, str(const_value))
                dimensions[dim] = world_size
        
        return {
            'material': world_elem.get('material', 'Vacuum'),
            'shape': shape.get('type', 'Box') if shape is not None else 'Box',
            'dimensions': dimensions,
            'position': {'x': 0, 'y': 0, 'z': 0}
        }
        
    def _voxelize_cube(x, y, z, dx, dy, dz, pixel_size):
        """
        Voxelize a cube centered at (x, y, z)
    
        dx, dy, dz are full lengths (DD4hep style)
        pixel_size is voxel edge length
        """
    
        hx = dx / 2.0
        hy = dy / 2.0
        hz = dz / 2.0
    
        xs = np.arange(x - hx + pixel_size / 2,
                        x + hx,
                        pixel_size)
        ys = np.arange(y - hy + pixel_size / 2,
                        y + hy,
                        pixel_size)
        zs = np.arange(z - hz + pixel_size / 2,
                        z + hz,
                        pixel_size)
    
        voxels = []
        for xi in xs:
            for yi in ys:
                for zi in zs:
                    voxels.append((xi, yi, zi))
    
        return voxels

    def _voxelize_sphere(x, y, z, r, pixel_size):
        """
        Voxelize a sphere centered at (x, y, z)
        """
    
        r2 = r * r
        voxels = []
    
        xs = np.arange(x - r, x + r, pixel_size)
        ys = np.arange(y - r, y + r, pixel_size)
        zs = np.arange(z - r, z + r, pixel_size)
    
        for xi in xs:
            for yi in ys:
                for zi in zs:
                    if (xi - x)**2 + (yi - y)**2 + (zi - z)**2 <= r2:
                        voxels.append((xi, yi, zi))
    
        return voxels
    
    def _parse_detectors(self):
        """
        Parse detector elements into a flat list of volumes.
    
        Roles:
          - base      : main world / solid detector volume
          - cutout    : vacuum sub-geometry (carves material)
          - overwrite : later solids that can replace vacuum
        """
    
        volumes = []
    
        detectors_elem = self.root.find('.//detectors')
        if detectors_elem is None:
            return volumes
    
        for det in detectors_elem.findall('detector'):
            det_id = int(det.get('id', 0))
            det_name = det.get('name', '')
            det_material = det.get('material', '')
    
            # ------------------------------------------------------------
            # Helper: resolve constants + units
            # ------------------------------------------------------------
            def resolve(value):
                for const_name, const_value in self.constants.items():
                    value = value.replace(const_name, str(const_value))
                return parse_dd4hep_constant(value)
    
            # ------------------------------------------------------------
            # Parse main detector volume (world or solid)
            # ------------------------------------------------------------
            dims = det.find('dimensions')
            pos = det.find('position')
    
            if dims is not None:
                role = "base" if det_id == 2000 else "overwrite"

                main_volume = {
                    "shape": "box",
                    "dimensions": {
                        "x": resolve(dims.get('x', '0')),
                        "y": resolve(dims.get('y', '0')),
                        "z": resolve(dims.get('z', '0')),
                    },
                    "position": {
                        "x": resolve(pos.get('x', '0')) if pos is not None else 0.0,
                        "y": resolve(pos.get('y', '0')) if pos is not None else 0.0,
                        "z": resolve(pos.get('z', '0')) if pos is not None else 0.0,
                    },
                    "material": det_material,
                    "role": role,
                    "detector_id": det_id,
                    "detector_name": det_name,
                }
    
                volumes.append(main_volume)
    
            # ------------------------------------------------------------
            # Determine detector role
            # ------------------------------------------------------------
            if det_id == 2000 and det_name == "RockWithCubeCutout":
                sub_role = "cutout"
                sub_material = "Vacuum"
            else:
                sub_role = "overwrite"
                sub_material = det_material
    
            # ------------------------------------------------------------
            # Parse cube sub-geometries
            # ------------------------------------------------------------
            for cube in det.findall('cube'):
                dims = cube.find('dimensions')
                pos = cube.find('position')
    
                volumes.append({
                    "shape": "box",
                    "dimensions": {
                        "x": resolve(dims.get('x', '0')),
                        "y": resolve(dims.get('y', '0')),
                        "z": resolve(dims.get('z', '0')),
                    },
                    "position": {
                        "x": resolve(pos.get('x', '0')),
                        "y": resolve(pos.get('y', '0')),
                        "z": resolve(pos.get('z', '0')),
                    },
                    "material": sub_material,
                    "role": sub_role,
                    "detector_id": det_id,
                    "detector_name": det_name,
                })
    
            # ------------------------------------------------------------
            # Parse sphere sub-geometries
            # ------------------------------------------------------------
            for sphere in det.findall('sphere'):
                dims = sphere.find('dimensions')
                pos = sphere.find('position')
    
                volumes.append({
                    "shape": "sphere",
                    "dimensions": {
                        "r": resolve(dims.get('r', '0')),
                    },
                    "position": {
                        "x": resolve(pos.get('x', '0')),
                        "y": resolve(pos.get('y', '0')),
                        "z": resolve(pos.get('z', '0')),
                    },
                    "material": sub_material,
                    "role": sub_role,
                    "detector_id": det_id,
                    "detector_name": det_name,
                })
    
        return volumes
    
    def print_summary(self):
        """Print summary of the geometry"""
        print(f"\n{'='*70}")
        print(f" Detector Geometry Analysis")
        print(f"{'='*70}")
        
        print(f"\n📋 Constants defined ({len(self.constants)}):")
        for name, value in sorted(self.constants.items())[:10]:  # Show first 10
            print(f"  {name:30s} = {value}")
        if len(self.constants) > 10:
            print(f"  ... and {len(self.constants)-10} more")
        
        print(f"\n🌍 World volume:")
        print(f"  Material: {self.world['material']}")
        print(f"  Shape: {self.world['shape']}")
        print(f"  Dimensions: {self.world['dimensions']}")
        
        print(f"\n🔬 Detectors ({len(self.detectors)}):")
        for det in self.detectors:
            print(f"  - {det['detector_name']} (ID: {det['detector_id']}, Type: {det['material']})")
            print(f"    Position: ({det['position']['x']:.1f}, "
                  f"{det['position']['y']:.1f}, {det['position']['z']:.1f}) mm")
            if det['shape'] == 'box':
                print(f"    Size: {det['dimensions']['x']:.1f} × "
                  f"{det['dimensions']['y']:.1f} × {det['dimensions']['z']:.1f} mm")
            elif det['shape'] == 'sphere':
                print(f"    Size: {det['dimensions']['r']:.1f} mm")
            print(f"    Material: {det['material']}")
    
    def visualize_2d(self):
        """Create 2D layout plots"""
        if not self.detectors:
            print("No detectors to visualize")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract positions and dimensions
        x_pos = [d['position']['x'] for d in self.detectors]
        y_pos = [d['position']['y'] for d in self.detectors]
        z_pos = [d['position']['z'] for d in self.detectors]
        
        x_sizes = [d['dimensions']['x'] for d in self.detectors]
        y_sizes = [d['dimensions']['y'] for d in self.detectors]
        z_sizes = [d['dimensions']['z'] for d in self.detectors]
        
        # XY view
        for i, (x, y, dx, dy) in enumerate(zip(x_pos, y_pos, x_sizes, y_sizes)):
            axes[0].add_patch(plt.Rectangle((x-dx/2, y-dy/2), dx, dy,
                                           alpha=0.5, label=f"Det{i}" if i<5 else None))
        axes[0].scatter(x_pos, y_pos, c='red', s=20, zorder=5)
        axes[0].set_xlabel('X (mm)')
        axes[0].set_ylabel('Y (mm)')
        axes[0].set_title('XY View')
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # XZ view
        for i, (x, z, dx, dz) in enumerate(zip(x_pos, z_pos, x_sizes, z_sizes)):
            axes[1].add_patch(plt.Rectangle((x-dx/2, z-dz/2), dx, dz,
                                           alpha=0.5))
        axes[1].scatter(x_pos, z_pos, c='red', s=20, zorder=5)
        axes[1].set_xlabel('X (mm)')
        axes[1].set_ylabel('Z (mm)')
        axes[1].set_title('XZ View')
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        # YZ view
        for i, (y, z, dy, dz) in enumerate(zip(y_pos, z_pos, y_sizes, z_sizes)):
            axes[2].add_patch(plt.Rectangle((y-dy/2, z-dz/2), dy, dz,
                                           alpha=0.5))
        axes[2].scatter(y_pos, z_pos, c='red', s=20, zorder=5)
        axes[2].set_xlabel('Y (mm)')
        axes[2].set_ylabel('Z (mm)')
        axes[2].set_title('YZ View')
        axes[2].grid(True, alpha=0.3)
        axes[2].axis('equal')
        
        if len(self.detectors) <= 5:
            axes[0].legend(loc='upper right', fontsize=8)
        
        plt.suptitle(f' Detector Layout: {os.path.basename(self.xml_file)}')
        plt.tight_layout()
        plt.savefig('2d.png',format='png')
        plt.show()
    
    def visualize_3d(self):
        """Create 3D visualization"""
        if not self.detectors:
            print("No detectors to visualize in 3D")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by material
        materials = list(set([d['material'] for d in self.detectors]))
        colors = plt.cm.tab10(np.linspace(0, 1, len(materials)))
        material_color = {mat: color for mat, color in zip(materials, colors)}
        
        for det in self.detectors:
            pos = det['position']
            dims = det['dimensions']
            color = material_color.get(det['material'], 'gray')
            
            # Create box vertices
            x, y, z = pos['x'], pos['y'], pos['z']
            dx, dy, dz = dims['x'], dims['y'], dims['z']
            
            # Vertices of the box
            vertices = np.array([
                [x-dx/2, y-dy/2, z-dz/2],
                [x+dx/2, y-dy/2, z-dz/2],
                [x+dx/2, y+dy/2, z-dz/2],
                [x-dx/2, y+dy/2, z-dz/2],
                [x-dx/2, y-dy/2, z+dz/2],
                [x+dx/2, y-dy/2, z+dz/2],
                [x+dx/2, y+dy/2, z+dz/2],
                [x-dx/2, y+dy/2, z+dz/2]
            ])
            
            # Plot edges
            edges = [
                [0,1], [1,2], [2,3], [3,0],  # bottom
                [4,5], [5,6], [6,7], [7,4],  # top
                [0,4], [1,5], [2,6], [3,7]   # sides
            ]
            
            for edge in edges:
                ax.plot(vertices[edge, 0], vertices[edge, 1], vertices[edge, 2],
                       color=color, alpha=0.8, linewidth=1)
            
            # Plot center point
            ax.scatter([x], [y], [z], color=color, s=50, alpha=0.8,
                      label=f"{det['detector_name']} ({det['material']})" if det == self.detectors[0] else "")
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D  Detector Geometry ({len(self.detectors)} detectors)')
        
        # Add legend
        if self.detectors:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('3d.png',format='png')
        plt.show()


class Voxelizer:
    def __init__(self, parser, resolution=64):
        self.parser = parser
        self.resolution = resolution
        
        # Material densities (g/cm³, approximate)
        self.material_densities = {
            'Steel235': 7.85,      # g/cm³
            'Vacuum': 0.0,
            'Air': 0.001225,
            'Polystyrene': 1.05,
            'Scintillator': 1.05,
            'Iron': 7.87,
            'Lead': 11.34,
            'Copper': 8.96,
            'Aluminum': 2.70,
            'Tungsten': 19.25,
            'Rock': 2.6,
        }

    def _fill_volume(self, voxel_grid, det, bbox, force_density=None):
        pos = det['position']
        dims = det['dimensions']
        material = det['material']
    
        density = (
            force_density
            if force_density is not None
            else self.material_densities.get(material, 1.0)
        )
    
        det_min_x = pos['x'] - dims.get('x', 0)/2
        det_max_x = pos['x'] + dims.get('x', 0)/2
        det_min_y = pos['y'] - dims.get('y', 0)/2
        det_max_y = pos['y'] + dims.get('y', 0)/2
        det_min_z = pos['z'] - dims.get('z', 0)/2
        det_max_z = pos['z'] + dims.get('z', 0)/2
    
        range_x = bbox['x_max'] - bbox['x_min']
        range_y = bbox['y_max'] - bbox['y_min']
        range_z = bbox['z_max'] - bbox['z_min']
    
        i_min = int((det_min_x - bbox['x_min']) / range_x * self.resolution)
        i_max = int((det_max_x - bbox['x_min']) / range_x * self.resolution)
        j_min = int((det_min_y - bbox['y_min']) / range_y * self.resolution)
        j_max = int((det_max_y - bbox['y_min']) / range_y * self.resolution)
        k_min = int((det_min_z - bbox['z_min']) / range_z * self.resolution)
        k_max = int((det_max_z - bbox['z_min']) / range_z * self.resolution)
    
        i_min = max(0, i_min)
        i_max = min(self.resolution-1, i_max)
        j_min = max(0, j_min)
        j_max = min(self.resolution-1, j_max)
        k_min = max(0, k_min)
        k_max = min(self.resolution-1, k_max)
    
        voxel_grid[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] = density


        
    def create_voxel_grid(self):
        """Create voxel grid from the geometry - FIXED VERSION"""
        if not self.parser.detectors:
            print("No detectors to voxelize")
            return np.zeros((self.resolution, self.resolution, self.resolution)), {}
        pixel_size = float(os.getenv('pixel_size')) # Meters
        world_size = int(pixel_size*int(self.resolution)*1000)
        # Use world dimensions directly from your XML
        # From your XML: world_side = 10*m = 10000mm
        world_half = world_size/2  # mm (half of 10m)
        
        bbox = {
            'x_min': -world_half,
            'x_max': world_half,
            'y_min': -world_half,
            'y_max': world_half,
            'z_min': -world_half,
            'z_max': world_half,
        }
        
        print(f"\n📦 Bounding box for voxelization:")
        print(f"   X: [{bbox['x_min']:.1f}, {bbox['x_max']:.1f}] mm")
        print(f"   Y: [{bbox['y_min']:.1f}, {bbox['y_max']:.1f}] mm")
        print(f"   Z: [{bbox['z_min']:.1f}, {bbox['z_max']:.1f}] mm")
        
        # Create empty voxel grid
        voxel_grid = np.zeros((self.resolution, self.resolution, self.resolution))
        
        # Voxel sizes
        voxel_size_x = (bbox['x_max'] - bbox['x_min']) / self.resolution
        voxel_size_y = (bbox['y_max'] - bbox['y_min']) / self.resolution
        voxel_size_z = (bbox['z_max'] - bbox['z_min']) / self.resolution

        xs = np.linspace(
            bbox['x_min'] + voxel_size_x / 2,
            bbox['x_max'] - voxel_size_x / 2,
            self.resolution,
        )
        ys = np.linspace(
            bbox['y_min'] + voxel_size_y / 2,
            bbox['y_max'] - voxel_size_y / 2,
            self.resolution,
        )
        zs = np.linspace(
            bbox['z_min'] + voxel_size_z / 2,
            bbox['z_max'] - voxel_size_z / 2,
            self.resolution,
        )
    
        voxel_coords = np.stack(
            np.meshgrid(xs, ys, zs, indexing="ij"),
            axis=-1,
        )
        
        print(f"\n📐 Voxel size: {voxel_size_x:.1f} × {voxel_size_y:.1f} × {voxel_size_z:.1f} mm")
        print(f"   Total voxels: {self.resolution**3:,}")
        
        # Voxelize each detector
        for det in filter(lambda d: d['role'] == 'base', self.parser.detectors):
            self._fill_volume(voxel_grid, det, bbox)
        
        # 2️⃣ Cutouts (vacuum always wins over base)
        for det in filter(lambda d: d['role'] == 'cutout', self.parser.detectors):
            self._fill_volume(voxel_grid, det, bbox, force_density=0.0)
        
        # 3️⃣ Overwrites (solid replaces vacuum)
        for det in filter(lambda d: d['role'] == 'overwrite', self.parser.detectors):
            self._fill_volume(voxel_grid, det, bbox)
        
        return voxel_grid, voxel_coords, bbox
    
    def plot_voxel_slices(self, voxel_grid):
        """Plot slices through the voxel grid"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        resolution = voxel_grid.shape[0]
        
        # Plot slices at different positions
        slice_positions = [resolution//4, resolution//2, 3*resolution//4]
        
        for idx, slice_pos in enumerate(slice_positions):
            # XY slice
            im1 = axes[0, idx].imshow(voxel_grid[:, :, slice_pos], 
                                     cmap='viridis', 
                                     origin='lower',
                                     aspect='auto',
                                     vmin=0,
                                     vmax=max(1, voxel_grid.max()))
            axes[0, idx].set_title(f'XY Slice at Z={slice_pos}')
            axes[0, idx].set_xlabel('X voxel')
            axes[0, idx].set_ylabel('Y voxel')
            plt.colorbar(im1, ax=axes[0, idx], label='Density (g/cm³)')
            
            # XZ slice
            im2 = axes[1, idx].imshow(voxel_grid[:, slice_pos, :],
                                     cmap='viridis',
                                     origin='lower',
                                     aspect='auto',
                                     vmin=0,
                                     vmax=max(1, voxel_grid.max()))
            axes[1, idx].set_title(f'XZ Slice at Y={slice_pos}')
            axes[1, idx].set_xlabel('X voxel')
            axes[1, idx].set_ylabel('Z voxel')
            plt.colorbar(im2, ax=axes[1, idx], label='Density (g/cm³)')
        
        plt.suptitle(f' Detector Voxel Grid ({resolution}³ resolution)')
        plt.tight_layout()
        plt.show()

    
    def world_to_voxel(self, x, y, z, bbox):
        """Convert world coordinates (mm) to voxel indices (i,j,k)"""
        rx = (x - bbox['x_min']) / (bbox['x_max'] - bbox['x_min'])
        ry = (y - bbox['y_min']) / (bbox['y_max'] - bbox['y_min'])
        rz = (z - bbox['z_min']) / (bbox['z_max'] - bbox['z_min'])
    
        i = int(rx * self.resolution)
        j = int(ry * self.resolution)
        k = int(rz * self.resolution)
    
        # Clamp safely
        i = max(0, min(self.resolution - 1, i))
        j = max(0, min(self.resolution - 1, j))
        k = max(0, min(self.resolution - 1, k))
    
        return i, j, k

        
    def save_results(self, voxel_grid, voxel_coords, bbox, prefix='_Detector'):
        """Save voxel grid and metadata"""
        # Save voxel grid
        voxel_file = f'{prefix}_voxels.npy'
        #np.save(voxel_file, voxel_grid)

        voxel_file_pkl = f'{prefix}_voxels.pkl'
        with open(voxel_file_pkl, 'wb') as f:
            pickle.dump(voxel_grid, f)

        with open(f"{prefix}_voxel_coords.pkl", "wb") as f:
            pickle.dump(voxel_coords, f)
            
        # ---- read detectors from config.sh (text parsing) ----
        detectors = read_detectors_from_config_file()
        
        detector_voxels = []
        for det in detectors:
            p = det["position_mm"]
        
            i, j, k = self.world_to_voxel(
                p["x"], p["y"], p["z"], bbox
            )
        
            flat = (
                i * self.resolution * self.resolution
                + j * self.resolution
                + k
            )
        
            detector_voxels.append({
                "id": det["id"],
                "name": det["name"],
                "position_mm": p,
                "voxel_ijk": (i, j, k),
                "voxel_flat": flat,
            })
        
                
        # Save metadata
        metadata = {
            'voxel_grid_shape': voxel_grid.shape,
            'bounding_box': bbox,
            'resolution': self.resolution,
            'material_voxels': int(np.count_nonzero(voxel_grid)),
            'total_voxels': voxel_grid.size,
            'max_density': float(voxel_grid.max()),
            'min_density': float(voxel_grid.min()),
            'targets': self.parser.detectors,
            'detectors': detector_voxels,
        }

        
        metadata_file = f'{prefix}_metadata.npy'
        np.save(metadata_file, metadata, allow_pickle=True)
        
        # Also save as text
        with open(f'{prefix}_metadata.txt', 'w') as f:
            f.write(f" Detector Voxelization Results\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"Voxel grid shape: {voxel_grid.shape}\n")
            f.write(f"Resolution: {self.resolution}\n")
            f.write(f"Total voxels: {voxel_grid.size:,}\n")
            f.write(f"Material voxels: {int(np.count_nonzero(voxel_grid)):,}\n")
            f.write(f"Empty voxels: {int(np.sum(voxel_grid == 0)):,}\n")
            f.write(f"Bounding box (mm):\n")
            f.write(f"  X: [{bbox['x_min']:.1f}, {bbox['x_max']:.1f}]\n")
            f.write(f"  Y: [{bbox['y_min']:.1f}, {bbox['y_max']:.1f}]\n")
            f.write(f"  Z: [{bbox['z_min']:.1f}, {bbox['z_max']:.1f}]\n\n")
            
            f.write(f"Targets:\n")
            for det in self.parser.detectors:
                f.write(f"  - {det['detector_name']} ({det['material']}) ({det['role']}):\n")
                f.write(f"    Position: ({det['position']['x']:.1f}, "
                       f"{det['position']['y']:.1f}, {det['position']['z']:.1f}) mm\n")
                if det['shape'] == 'box':
                    f.write(f"    Size: {det['dimensions']['x']:.1f} × "
                           f"{det['dimensions']['y']:.1f} × {det['dimensions']['z']:.1f} mm\n")
                elif det['shape'] == 'sphere':
                    f.write(f"    Size: {det['dimensions']['r']:.1f} mm\n")

            f.write("\nDetector positions (from config.sh):\n")
            f.write("=" * 50 + "\n")
            
            for d in detector_voxels:
                f.write(f"Detector {d['id']}:\n")
                f.write(
                    f"  Position (mm): "
                    f"x={d['position_mm']['x']:.1f}, "
                    f"y={d['position_mm']['y']:.1f}, "
                    f"z={d['position_mm']['z']:.1f}\n"
                )
                f.write(
                    f"  Voxel index (i,j,k): {d['voxel_ijk']}\n"
                )
                f.write(
                    f"  Flat voxel number: {d['voxel_flat']}\n\n"
                )
        
        print(f"\n💾 Results saved:")
        print(f"   Voxel grid: {voxel_file}")
        print(f"   Metadata (binary): {metadata_file}")
        print(f"   Metadata (text): {prefix}_metadata.txt")
        
        return voxel_file, metadata_file




def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python _viz.py <xml_file>")
        print("Example: python _viz.py _Detector.xml")
        sys.exit(1)
    
    xml_file = sys.argv[1]
    
    if not os.path.exists(xml_file):
        print(f"Error: File '{xml_file}' not found!")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"🚀  Detector Visualizer & Voxelizer")
    print(f"{'='*70}")
    print(f"📁 Input file: {xml_file}")

    # Ask for resolution
    res_choice = input(f"Voxel resolution (default 64, max 256): ").strip()
    if res_choice and res_choice.isdigit():
        resolution = min(int(res_choice), 256)
    else:
        resolution = 64
    
    print(f"Using resolution: {resolution}³")
    
    # Parse the XML
    parser = DetectorParser(xml_file, resolution)
    
    # Print summary
    parser.print_summary()
    
    # Ask user what to do
    print("\n🔧 Options:")
    print("   1. 2D Layout Visualization")
    print("   2. 3D Geometry Visualization")
    print("   3. Create Voxel Grid (64x64x64)")
    print("   4. All of the above")
    print("   5. Exit")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice in ['1', '4']:
            print("\n📊 Creating 2D layout plot...")
            parser.visualize_2d()
        
        if choice in ['2', '4']:
            print("\n📊 Creating 3D geometry plot...")
            parser.visualize_3d()
        
        if choice in ['3', '4']:
            print("\n🛠️ Creating voxel grid...")
            
            # Create voxelizer
            voxelizer = Voxelizer(parser, resolution=resolution)
            
            # Create voxel grid
            voxel_grid, voxel_coords, bbox = voxelizer.create_voxel_grid()
            
            # Plot slices
            voxelizer.plot_voxel_slices(voxel_grid)
            
            # Save results
            prefix = os.path.splitext(os.path.basename(xml_file))[0]
            voxelizer.save_results(voxel_grid, voxel_coords, bbox, prefix=prefix)
        
        if choice == '5':
            print("\n👋 Goodbye!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def quick_visualize(xml_file):
    """Quick one-liner visualization"""
    parser = DetectorParser(xml_file, resolution)
    parser.print_summary()
    parser.visualize_2d()
    
    # Quick voxelization
    voxelizer = Voxelizer(parser, resolution=32)
    voxel_grid, _ = voxelizer.create_voxel_grid()
    
    # Show summary
    print(f"\n📊 Voxel grid summary:")
    print(f"   Shape: {voxel_grid.shape}")
    print(f"   Non-zero voxels: {np.count_nonzero(voxel_grid)}")
    print(f"   Max density: {voxel_grid.max():.2f} g/cm³")
    
    return parser, voxel_grid


if __name__ == "__main__":
    main()