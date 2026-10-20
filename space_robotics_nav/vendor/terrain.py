from __future__ import annotations

import numpy as np
import torch

class HeightmapManager:
    """
    A professional heightmap manager for converting 3D mesh data to 2D heightmaps.
    
    This class handles the conversion of 3D mesh vertices and faces into a structured
    2D heightmap representation, enabling efficient height queries and terrain analysis.
    Supports both CPU and CUDA tensor operations for performance optimization.
    
    Attributes:
        resolution_in_m (float): The resolution of the heightmap in meters per pixel.
        device (str): The device to use for tensor operations ('cpu', 'cuda', or 'cuda:0').
        heightmap (np.ndarray): The 2D heightmap as a numpy array.
        min_x (float): Minimum X coordinate of the heightmap bounds.
        min_y (float): Minimum Y coordinate of the heightmap bounds.
        max_x (float): Maximum X coordinate of the heightmap bounds.
        max_y (float): Maximum Y coordinate of the heightmap bounds.
        heightmap_tensor (torch.Tensor): GPU/CPU tensor version of the heightmap.
        offset_tensor (torch.Tensor): Offset tensor for coordinate transformations.
    """

    def __init__(self, resolution_in_m: float, vertices: np.ndarray, faces: np.ndarray, device: str = 'cpu') -> None:
        """
        Initialize the HeightmapManager with mesh data and configuration.
        
        Args:
            resolution_in_m: The resolution of the heightmap in meters per pixel.
            vertices: 3D vertex array of shape (N, 3) containing [x, y, z] coordinates.
            faces: Face indices array of shape (M, 3) referencing vertices.
            device: Device for tensor operations ('cpu', 'cuda', or 'cuda:0').
            
        Raises:
            ValueError: If vertices or faces arrays have invalid shapes.
            RuntimeError: If CUDA is requested but not available.
        """
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Vertices must have shape (N, 3), got {vertices.shape}")
        if len(faces.shape) != 2 or faces.shape[1] != 3:
            raise ValueError(f"Faces must have shape (M, 3), got {faces.shape}")
        
        self.resolution_in_m = resolution_in_m
        self.device = device
        
        # Generate heightmap from mesh data
        self.heightmap, self.min_x, self.min_y, self.max_x, self.max_y = self._mesh_to_heightmap(vertices, faces)
        
        # Initialize tensors based on device
        self._initialize_tensors()
    
    def _initialize_tensors(self) -> None:
        """Initialize heightmap and offset tensors on the specified device."""
        if self.device in ['cuda', 'cuda:0']:
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA requested but not available. Falling back to CPU.")
            self.heightmap_tensor = torch.from_numpy(self.heightmap).cuda()
            self.offset_tensor = torch.tensor([self.min_x, self.min_y], dtype=torch.float32).cuda()
        else:
            self.heightmap_tensor = torch.from_numpy(self.heightmap)
            self.offset_tensor = torch.tensor([self.min_x, self.min_y], dtype=torch.float32)

    def _mesh_to_heightmap(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Convert 3D mesh data to a 2D heightmap representation.
        
        This method projects 3D triangular mesh faces onto a 2D grid, recording the maximum
        height at each grid cell. The resulting heightmap provides an efficient way to
        query terrain heights at arbitrary 2D positions.
        
        Args:
            vertices: 3D vertex coordinates array of shape (N, 3).
            faces: Triangle face indices array of shape (M, 3).
            
        Returns:
            A tuple containing:
                - heightmap (np.ndarray): 2D height grid of shape (height, width).
                - min_x (float): Minimum X coordinate of the heightmap bounds.
                - min_y (float): Minimum Y coordinate of the heightmap bounds.
                - max_x (float): Maximum X coordinate of the heightmap bounds.
                - max_y (float): Maximum Y coordinate of the heightmap bounds.
                
        Note:
            A border margin of 1.0 meter is applied to prevent edge effects.
            Grid cells without mesh coverage are initialized to negative infinity.
        """
        # Apply border margin to prevent edge effects
        border_margin = 1.0
        
        # Calculate bounding box with margin
        min_x, min_y, _ = np.min(vertices, axis=0) + border_margin
        max_x, max_y, _ = np.max(vertices, axis=0) - border_margin

        # Calculate grid dimensions
        grid_size_x = (max_x - min_x) / self.resolution_in_m
        grid_size_y = (max_y - min_y) / self.resolution_in_m
        
        # Grid dimensions (add 1 for inclusive bounds)
        grid_width = int(grid_size_x + 1)
        grid_height = int(grid_size_y + 1)

        # Initialize below every valid elevation so negative-altitude terrain can populate the heightmap.
        heightmap = np.full((grid_height, grid_width), -np.inf, dtype=np.float32)

        # Calculate cell sizes
        cell_size_x = (max_x - min_x) / grid_size_x
        cell_size_y = (max_y - min_y) / grid_size_y

        if len(faces) > 0:
            # Vectorized processing of all triangles
            face_vertices = vertices[faces]
            
            # Extract coordinate components
            x_coords = face_vertices[:, :, 0]
            y_coords = face_vertices[:, :, 1]
            z_coords = face_vertices[:, :, 2]
            
            # Calculate bounding boxes for all triangles
            min_x_tri = np.min(x_coords, axis=1)
            max_x_tri = np.max(x_coords, axis=1)
            min_y_tri = np.min(y_coords, axis=1)
            max_y_tri = np.max(y_coords, axis=1)
            max_z_tri = np.max(z_coords, axis=1)
            
            # Convert world coordinates to grid coordinates
            min_i = np.maximum(0, ((min_x_tri - min_x) / cell_size_x).astype(int))
            max_i = np.minimum(grid_width - 1, ((max_x_tri - min_x) / cell_size_x).astype(int))
            min_j = np.maximum(0, ((min_y_tri - min_y) / cell_size_y).astype(int))
            max_j = np.minimum(grid_height - 1, ((max_y_tri - min_y) / cell_size_y).astype(int))
            
            # Project triangles onto heightmap
            for idx in range(len(faces)):
                i_range = max_i[idx] - min_i[idx] + 1
                j_range = max_j[idx] - min_j[idx] + 1
                
                if i_range > 0 and j_range > 0:
                    # Update heightmap with maximum height in each cell
                    heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1] = np.maximum(
                        heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1],
                        max_z_tri[idx]
                    )

        return heightmap, min_x, min_y, max_x, max_y

    def get_height_at(self, position: torch.Tensor) -> torch.Tensor:
        """
        Query the height at specified 2D positions using bilinear interpolation.
        
        This method efficiently retrieves terrain heights at arbitrary 2D world coordinates
        by mapping them to the heightmap grid and performing lookups. Input positions are
        automatically clamped to the heightmap bounds to prevent out-of-bounds access.
        
        Args:
            position: 2D world coordinates tensor of shape (N, 2) containing [x, y] positions.
                     Must be on the same device as the HeightmapManager.
                     
        Returns:
            Height values tensor of shape (N,) corresponding to each input position.
            
        Raises:
            RuntimeError: If position tensor is not on the same device as the heightmap.
            
        Example:
            >>> positions = torch.tensor([[10.0, 15.0], [20.0, 25.0]], device='cuda')
            >>> heights = heightmap_manager.get_height_at(positions)
            >>> print(heights.shape)  # torch.Size([2])
            
        Note:
            Positions outside the heightmap bounds are clamped to the nearest valid grid cell.
            This prevents extrapolation and ensures consistent behavior at terrain edges.
        """
        if position.device != self.heightmap_tensor.device:
            raise RuntimeError(f"Position tensor device ({position.device}) must match "
                             f"heightmap device ({self.heightmap_tensor.device})")
        
        if position.shape[-1] != 2:
            raise ValueError(f"Position tensor must have shape (..., 2), got {position.shape}")
        
        grid_cell = self.world_to_grid(position)

        # Clamp indices to valid heightmap bounds
        grid_cell[:, 0] = torch.clamp(grid_cell[:, 0], 0, self.heightmap_tensor.shape[1] - 1)
        grid_cell[:, 1] = torch.clamp(grid_cell[:, 1], 0, self.heightmap_tensor.shape[0] - 1)

        # Retrieve heights at the specified grid positions
        return self.heightmap_tensor[grid_cell[:, 1], grid_cell[:, 0]]

    def world_to_grid(self, position: torch.Tensor) -> torch.Tensor:
        """Convert world XY coordinates to integer heightmap indices."""
        return ((position - self.offset_tensor) / self.resolution_in_m).long()
    
    def get_heightmap_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the world coordinate bounds of the heightmap.
        
        Returns:
            A tuple containing (min_x, min_y, max_x, max_y) in world coordinates.
        """
        return self.min_x, self.min_y, self.max_x, self.max_y
    
    def get_heightmap_shape(self) -> Tuple[int, int]:
        """
        Get the dimensions of the heightmap grid.
        
        Returns:
            A tuple containing (height, width) of the heightmap in grid cells.
        """
        return self.heightmap.shape
    
    def is_position_valid(self, position: torch.Tensor) -> torch.Tensor:
        """
        Check if positions are within the valid heightmap bounds.
        
        Args:
            position: 2D world coordinates tensor of shape (N, 2).
            
        Returns:
            Boolean tensor of shape (N,) indicating valid positions.
        """
        within_x_bounds = (position[:, 0] >= self.min_x) & (position[:, 0] <= self.max_x)
        within_y_bounds = (position[:, 1] >= self.min_y) & (position[:, 1] <= self.max_y)
        return within_x_bounds & within_y_bounds


class TerrainOps:
    def check_if_target_is_valid(
            self,
            env_ids: torch.Tensor,
            target_positions: torch.Tensor,
            device: str = "cuda:0"
    ) -> Tuple[torch.Tensor, int]:
        """
        Validate target positions against terrain safety constraints.
        
        This method checks if target positions are located in safe areas (not on rocks
        or obstacles) and returns the environment IDs that require reset due to
        unsafe target placement. Additionally checks if targets are within valid boundary
        distance from the map edges.
        
        Note: Targets are allowed on steep terrain - only rock/obstacle constraints apply.
        Rovers can navigate to targets on steep terrain, but shouldn't spawn on them.
        
        Args:
            env_ids: Environment IDs to check, tensor of shape (N,).
            target_positions: Target positions to validate, tensor of shape (N, 2 or 3).
            device: Device for tensor operations (legacy parameter, uses manager's device).
            
        Returns:
            A tuple containing:
                - env_ids_to_reset (torch.Tensor): Environment IDs requiring reset.
                - num_resets (int): Number of environments requiring reset.
                
        Note:
            Only uses the first 2 dimensions (x, y) of target_positions for validation.
            The safety check is performed against rock safety masks only (not gradient masks)
            and boundary distance constraints using target_distance_to_boundary parameter.
        """
        # Check 1: Boundary distance constraints
        # Ensure targets are not too close to the map boundaries
        x_coords = target_positions[:, 0]
        y_coords = target_positions[:, 1]
        
        # Get heightmap bounds
        min_x, min_y, max_x, max_y = self._heightmap_manager.get_heightmap_bounds()
        
        # Check if targets are within valid boundary distance
        boundary_margin = self.target_distance_to_boundary
        boundary_violations = (
            (x_coords < min_x + boundary_margin) |
            (x_coords > max_x - boundary_margin) |
            (y_coords < min_y + boundary_margin) |
            (y_coords > max_y - boundary_margin)
        )
        
        # Check 2: Rock/obstacle safety constraints (targets allowed on steep terrain)
        # Convert to integer grid indices and clamp to valid bounds
        grid_cell = self._heightmap_manager.world_to_grid(target_positions[:, 0:2])
        grid_cell[:, 0] = torch.clamp(grid_cell[:, 0], 0, self._heightmap_manager.heightmap_tensor.shape[1] - 1)
        grid_cell[:, 1] = torch.clamp(grid_cell[:, 1], 0, self._heightmap_manager.heightmap_tensor.shape[0] - 1)

        # Check rock safety mask only (targets can be on steep terrain, just not on rocks): 1 indicates unsafe areas requiring reset
        safety_violations = torch.where(self.safe_rock_mask_tensor[grid_cell[:, 1], grid_cell[:, 0]] == 1, 1, 0).squeeze(-1)
        terrain_violations = ~torch.isfinite(self._heightmap_manager.heightmap_tensor[grid_cell[:, 1], grid_cell[:, 0]])
        
        # Combine checks: reset if the target is outside the map, on an obstacle, or over a heightmap gap.
        combined_violations = boundary_violations | (safety_violations == 1) | terrain_violations
        env_ids_to_reset = env_ids[combined_violations]
        
        return env_ids_to_reset, len(env_ids_to_reset)

    def project_rocks_to_heightmap(self, rock_vertices: np.ndarray, rock_faces: np.ndarray, safety_margin: float = 2.0):
        """Project rock mesh triangles onto XY plane to create rock masks"""
        import cv2
        from scipy import ndimage
        
        # Get heightmap dimensions and bounds from the combined heightmap
        height, width = self._heightmap_manager.heightmap.shape
        min_x = self._heightmap_manager.min_x
        min_y = self._heightmap_manager.min_y
        
        # Initialize rock mask
        rock_mask = np.zeros((height, width), dtype=np.uint8)
        
        self.log(f'Projecting {len(rock_faces)} rock triangles onto XY plane', level='debug', block=False)
        
        if len(rock_faces) > 0:
            # Get all triangle vertices and project to 2D
            face_vertices = rock_vertices[rock_faces]
            
            # Extract XY coordinates
            x_coords = face_vertices[:, :, 0]
            y_coords = face_vertices[:, :, 1]
            
            # Find bounding box for each triangle
            min_x_tri = np.min(x_coords, axis=1)
            max_x_tri = np.max(x_coords, axis=1)
            min_y_tri = np.min(y_coords, axis=1)
            max_y_tri = np.max(y_coords, axis=1)
            
            # Convert to grid coordinates
            cell_size_x = (self._heightmap_manager.max_x - min_x) / (width - 1)
            cell_size_y = (self._heightmap_manager.max_y - min_y) / (height - 1)
            
            min_i = np.maximum(0, ((min_x_tri - min_x) / cell_size_x).astype(int))
            max_i = np.minimum(width - 1, ((max_x_tri - min_x) / cell_size_x).astype(int))
            min_j = np.maximum(0, ((min_y_tri - min_y) / cell_size_y).astype(int))
            max_j = np.minimum(height - 1, ((max_y_tri - min_y) / cell_size_y).astype(int))
            
            # Mark triangles on the mask
            for idx in range(len(rock_faces)):
                i_range = max_i[idx] - min_i[idx] + 1
                j_range = max_j[idx] - min_j[idx] + 1
                
                if i_range > 0 and j_range > 0:
                    rock_mask[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1] = 1
        
        self.log(f'Rocks projected onto heightmap: {np.sum(rock_mask)} cells marked as rocks', level='debug', block=True)
        self.log("Rock projection completed. Applying morphological operations...", level='debug', block=False)

        # Apply morphological operations to clean up the mask
        kernel_small = np.ones((3, 3), np.uint8)
        rock_mask = cv2.morphologyEx(rock_mask, cv2.MORPH_CLOSE, kernel_small)
        rock_mask = ndimage.binary_fill_holes(rock_mask).astype(np.uint8)
        
        # Keep small components: minimum-diameter lethal rocks still need exclusion zones.
        # Dilate slightly to account for rock boundaries
        kernel_dilate = np.ones((7, 7), np.uint8)
        rock_mask = cv2.dilate(rock_mask, kernel_dilate, iterations=1)
        
        # Create safety margin for spawn locations
        # This creates a larger exclusion zone around rocks for safer navigation
        safety_margin_size = int(safety_margin / self.resolution_in_m)  # 2 meter safety margin
        kernel_safety = np.ones((safety_margin_size, safety_margin_size), np.uint8)
        safe_rock_mask = cv2.dilate(rock_mask, kernel_safety, iterations=1)

        self.log(
            f"Rock mask created: {np.sum(rock_mask)} cells marked as rocks\n"
            f"Rock safety mask created: {np.sum(safe_rock_mask)} cells marked as unsafe",
            level='debug', block=True
        )

        return rock_mask.astype(np.int32), safe_rock_mask.astype(np.int32)

    def compute_gradient_masks(self, heightmap, threshold=0.1, safety_margin=2.0):
        """
        Compute gradient masks for terrain steepness (similar to find_rocks_in_heightmap but only for terrain)
        """
        import cv2
        from scipy import ndimage
        from scipy.signal import convolve2d

        # Sobel operators for gradient in x and y directions
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        valid_height_mask = np.isfinite(heightmap)
        heightmap_for_gradient = np.where(valid_height_mask, heightmap, 0.0)

        # Compute the gradient components
        grad_x = convolve2d(heightmap_for_gradient, sobel_x, mode='same', boundary='wrap')
        grad_y = convolve2d(heightmap_for_gradient, sobel_y, mode='same', boundary='wrap')

        # Compute the overall gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Create mask for steep areas
        gradient_mask = np.zeros_like(heightmap, dtype=np.int32)
        gradient_mask[grad_magnitude > threshold] = 1
        gradient_mask[~valid_height_mask] = 1
        gradient_mask = gradient_mask.astype(np.uint8)
        
        # Apply morphological operations to clean up the mask
        kernel_small = np.ones((3, 3), np.uint8)
        gradient_mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_CLOSE, kernel_small)
        gradient_mask = ndimage.binary_fill_holes(gradient_mask).astype(np.uint8)
        
        # Remove very small isolated regions
        kernel_open = np.ones((5, 5), np.uint8)
        gradient_mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Dilate slightly to account for gradient boundaries
        kernel_dilate = np.ones((7, 7), np.uint8)
        gradient_mask = cv2.dilate(gradient_mask, kernel_dilate, iterations=1)
        
        # Create safety margin for gradient areas
        # This creates a larger exclusion zone around steep areas for safer navigation
        safety_margin_size = int(safety_margin / self.resolution_in_m)  # default 2.0 meter safety margin
        kernel_safety = np.ones((safety_margin_size, safety_margin_size), np.uint8)
        safe_gradient_mask = cv2.dilate(gradient_mask, kernel_safety, iterations=1)

        self.log(
            f"Gradient mask created: {np.sum(gradient_mask)} cells marked as steep\n"
            f"Gradient safety mask created: {np.sum(safe_gradient_mask)} cells marked as unsafe",
            level='debug', block=True
        )

        return gradient_mask.astype(np.int32), safe_gradient_mask.astype(np.int32)

    def random_rover_spawns(
            self,
            safe_mask: np.ndarray,
            heightmap: np.ndarray, 
            n_spawns: int = 100,
            border_offset: float = 20.0,
            seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate safe random spawn locations for rover deployment.
        
        This method creates a specified number of spawn locations by randomly sampling
        positions within the heightmap bounds and validating them against safety constraints.
        Generated positions avoid rocks, steep terrain, and maintain border margins.
        
        Args:
            safe_mask: Binary safety mask where 0=safe, 1=unsafe areas.
            heightmap: Height data for Z-coordinate assignment.
            n_spawns: Number of spawn locations to generate.
            border_offset: Safety margin from heightmap edges in meters.
            seed: Random seed for reproducible generation.
            
        Returns:
            Array of spawn locations with shape (n_spawns, 3) containing [x, y, z] coordinates
            in world space.
            
        Raises:
            ValueError: If the border offset is too large or no valid spawn cells are available.
        """
        if seed is not None:
            np.random.seed(seed)

        height, width = safe_mask.shape
        border_cells = int(border_offset / self.resolution_in_m)
        if height <= 2 * border_cells or width <= 2 * border_cells:
            raise ValueError(
                f"Border offset {border_offset}m is too large for heightmap shape {(height, width)} "
                f"at {self.resolution_in_m}m resolution."
            )

        valid_mask = (safe_mask == 0) & np.isfinite(heightmap)
        interior_mask = np.zeros_like(valid_mask)
        interior_mask[border_cells:height - border_cells, border_cells:width - border_cells] = True
        valid_cells = np.argwhere(valid_mask & interior_mask)
        if len(valid_cells) == 0:
            raise ValueError("No valid rover spawn cells remain after terrain safety filtering.")

        sampled_cells = valid_cells[np.random.randint(0, len(valid_cells), size=n_spawns)]
        spawn_locations = np.empty((n_spawns, 3), dtype=np.float32)
        spawn_locations[:, 0] = sampled_cells[:, 1]
        spawn_locations[:, 1] = sampled_cells[:, 0]
        spawn_locations[:, 2] = heightmap[sampled_cells[:, 0], sampled_cells[:, 1]]

        # Convert grid coordinates to world coordinates
        spawn_locations[:, 0] = spawn_locations[:, 0] * self.resolution_in_m + self._heightmap_manager.min_x
        spawn_locations[:, 1] = spawn_locations[:, 1] * self.resolution_in_m + self._heightmap_manager.min_y

        return spawn_locations

    def resize_terrain_heightmap_to_match_combined(self, terrain_vertices: np.ndarray, terrain_faces: np.ndarray, reference_heightmap_manager):
        """Resize terrain-only heightmap to match combined heightmap dimensions"""
        # Get target dimensions from combined heightmap
        target_height, target_width = reference_heightmap_manager.heightmap.shape
        target_min_x = reference_heightmap_manager.min_x
        target_min_y = reference_heightmap_manager.min_y
        target_max_x = reference_heightmap_manager.max_x
        target_max_y = reference_heightmap_manager.max_y
        
        # Initialize terrain-only heightmap with same dimensions and bounds
        terrain_heightmap = np.full((target_height, target_width), -np.inf, dtype=np.float32)
        
        # Calculate cell size
        cell_size_x = (target_max_x - target_min_x) / (target_width - 1)
        cell_size_y = (target_max_y - target_min_y) / (target_height - 1)
        
        if len(terrain_faces) > 0:
            # Get all triangle vertices from terrain only
            face_vertices = terrain_vertices[terrain_faces]
            
            # Extract coordinates
            x_coords = face_vertices[:, :, 0]
            y_coords = face_vertices[:, :, 1]
            z_coords = face_vertices[:, :, 2]
            
            # Find bounding box for each triangle
            min_x_tri = np.min(x_coords, axis=1)
            max_x_tri = np.max(x_coords, axis=1)
            min_y_tri = np.min(y_coords, axis=1)
            max_y_tri = np.max(y_coords, axis=1)
            max_z_tri = np.max(z_coords, axis=1)
            
            # Convert to grid coordinates using target bounds
            min_i = np.maximum(0, ((min_x_tri - target_min_x) / cell_size_x).astype(int))
            max_i = np.minimum(target_width - 1, ((max_x_tri - target_min_x) / cell_size_x).astype(int))
            min_j = np.maximum(0, ((min_y_tri - target_min_y) / cell_size_y).astype(int))
            max_j = np.minimum(target_height - 1, ((max_y_tri - target_min_y) / cell_size_y).astype(int))
            
            # Process triangles
            for idx in range(len(terrain_faces)):
                i_range = max_i[idx] - min_i[idx] + 1
                j_range = max_j[idx] - min_j[idx] + 1
                
                if i_range > 0 and j_range > 0:
                    terrain_heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1] = np.maximum(
                        terrain_heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1],
                        max_z_tri[idx]
                    )
        
        self.log(f"Resizing terrain-only heightmap to match combined heightmap dimensions: {target_height}x{target_width}", level='debug', block=False)

        
        # Create a new HeightmapManager with the resized terrain heightmap
        # Use same bounds as the reference heightmap
        resized_heightmap_manager = HeightmapManager.__new__(HeightmapManager)
        resized_heightmap_manager.resolution_in_m = self.resolution_in_m
        resized_heightmap_manager.device = self.device
        resized_heightmap_manager.heightmap = terrain_heightmap
        resized_heightmap_manager.min_x = target_min_x
        resized_heightmap_manager.min_y = target_min_y
        resized_heightmap_manager.max_x = target_max_x
        resized_heightmap_manager.max_y = target_max_y
        
        # Set up tensor attributes
        if self.device == 'cuda' or self.device == 'cuda:0':
            resized_heightmap_manager.heightmap_tensor = torch.from_numpy(terrain_heightmap).cuda()
            resized_heightmap_manager.offset_tensor = torch.tensor([target_min_x, target_min_y]).cuda()
        else:
            resized_heightmap_manager.heightmap_tensor = torch.from_numpy(terrain_heightmap)
            resized_heightmap_manager.offset_tensor = torch.tensor([target_min_x, target_min_y])
        
        return resized_heightmap_manager
