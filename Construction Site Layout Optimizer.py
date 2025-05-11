"""
Construction Site Layout Optimizer
A sophisticated GUI application for optimizing construction site facilities layout
"""

import sys
import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

from queue import PriorityQueue
from typing import List, Dict, Tuple, Optional, Type, Union
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QLineEdit, QPushButton,
                            QComboBox, QTabWidget, QGridLayout, QGroupBox,
                            QScrollArea, QSplitter, QFrame, QFileDialog,
                            QSlider, QCheckBox, QSpinBox, QDoubleSpinBox,
                            QMessageBox, QTableWidget, QTableWidgetItem,
                            QHeaderView, QProgressBar, QRadioButton, QButtonGroup,
                            QColorDialog, QAbstractSpinBox, QAction, QToolBar, QMenu,
                            QSpacerItem, QSizePolicy)
from PyQt5.QtCore import (Qt, QRect, QRectF, QPoint, QPointF, QSize, QSizeF,
                         pyqtSignal, QTimer, QThread, QSettings, QDate,
                         QLineF, QDir, QFileInfo)
from PyQt5.QtGui import (QPainter, QColor, QBrush, QPen, QFont, QPainterPath,
                        QPixmap, QPalette, QIcon, QTransform, QImage, QPolygonF,
                        QLinearGradient, QFontMetricsF, QCursor, # <--- ADDED QCursor HERE
                        QPaintEvent, QMouseEvent, QWheelEvent, QKeyEvent, QCloseEvent, QResizeEvent, QContextMenuEvent)

DARK_THEME_TEXT_COLOR = QColor(220, 220, 220)
DARK_THEME_INFO_TEXT_COLOR = QColor(180, 180, 180) # Lighter gray for info text on dark background
DARK_THEME_BORDER_COLOR = QColor(85, 85, 85)     # Darker gray for borders on dark background
DARK_THEME_GROUPBOX_BG_COLOR = QColor(55, 55, 55) # Slightly lighter than main window for groupboxes
DARK_THEME_BUTTON_BG_COLOR = QColor(70, 70, 70)
DARK_THEME_BUTTON_HOVER_BG_COLOR = QColor(90, 90, 90)
DARK_THEME_BUTTON_PRESSED_BG_COLOR = QColor(60, 60, 60)
DARK_THEME_SUCCESS_BUTTON_BG_COLOR = QColor(40, 120, 65) # Darker green
DARK_THEME_SUCCESS_BUTTON_HOVER_BG_COLOR = QColor(50, 140, 75)
DARK_THEME_SUCCESS_BUTTON_PRESSED_BG_COLOR = QColor(30, 100, 55)

# --- Global Default Configurations ---
SITE_ELEMENTS = {
    'Building': {'color': QColor(200, 200, 220), 'movable': True, 'priority': 1, 'placeable_inside_building': False, 'resizable': False},
    'Contractor Office': {'color': QColor(100, 149, 237), 'movable': True, 'priority': 3, 'placeable_inside_building': True, 'resizable': True, 'aspect_ratio': 1.5},
    'Consultant Office': {'color': QColor(65, 105, 225), 'movable': True, 'priority': 3, 'placeable_inside_building': True, 'resizable': True, 'aspect_ratio': 1.3},
    'Client Office': {'color': QColor(30, 144, 255), 'movable': True, 'priority': 3, 'placeable_inside_building': True, 'resizable': True, 'aspect_ratio': 1.2},
    'Workers Dormitory': {'color': QColor(70, 130, 180), 'movable': True, 'priority': 2, 'placeable_inside_building': True, 'resizable': True, 'aspect_ratio': 2.0},
    'Security Post': {'color': QColor(0, 128, 0), 'movable': True, 'priority': 4, 'placeable_inside_building': False, 'resizable': False, 'fixed_width': 3.0, 'fixed_height': 4.0},
    'Sanitary Facilities': {'color': QColor(144, 238, 144), 'movable': True, 'priority': 4, 'placeable_inside_building': True, 'resizable': True, 'aspect_ratio': 1.0},
    'Material Storage': {'color': QColor(210, 180, 140), 'movable': True, 'priority': 2, 'placeable_inside_building': True, 'resizable': True, 'aspect_ratio': 2.5},
    'Welding Workshop': {'color': QColor(255, 140, 0), 'movable': True, 'priority': 2, 'placeable_inside_building': False, 'resizable': True, 'aspect_ratio': 1.8},
    'Tower Crane': {'color': QColor(128, 0, 0), 'movable': True, 'priority': 1, 'placeable_inside_building': False, 'resizable': True, 'fixed_width': 7.0, 'fixed_height': 7.0},
    'Fuel Tank': {'color': QColor(255, 0, 0), 'movable': True, 'priority': 5, 'placeable_inside_building': False, 'resizable': False, 'fixed_width': 4.0, 'fixed_height': 4.0},
    'Machinery Parking': {'color': QColor(169, 169, 169), 'movable': True, 'priority': 4, 'placeable_inside_building': False, 'resizable': True, 'aspect_ratio': 2.0}
}
DEFAULT_SPACE_REQUIREMENTS = { # Example values
    'Building': 2400.0, 'Contractor Office': 35.0, 'Consultant Office': 25.0, 'Client Office': 15.0,
    'Workers Dormitory': 100.0, 'Security Post': 6.0, 'Sanitary Facilities': 20.0,
    'Material Storage': 150.0, 'Welding Workshop': 75.0, 'Tower Crane': 36.0,
    'Fuel Tank': 8.0, 'Machinery Parking': 120.0
}
DEFAULT_CONSTRAINTS = {
    'min_distance_fuel_dormitory': 25.0, 'min_distance_fuel_welding': 20.0,
    'max_distance_storage_welding': 20.0, 'crane_reach': 45.0,
    'min_distance_offices_welding': 18.0, 'min_distance_offices_machinery': 12.0,
    'min_path_width': 3.0, 'safety_buffer': 1.5, 'building_safety_buffer': 1.0,
    'buffer_building_override': True
}
# For the personnel_config_data in SpaceRequirementsWidget
DEFAULT_PERSONNEL_CONFIG = {
    'Contractor Office': {'count': 10, 'area_per_person': 7.0, 'default_total_area': 70.0},
    'Consultant Office': {'count': 6, 'area_per_person': 7.0, 'default_total_area': 42.0},
    'Client Office':   {'count': 3,  'area_per_person': 8.0, 'default_total_area': 24.0},
    'Workers Dormitory': {'count': 30, 'area_per_person': 6.0, 'default_total_area': 180.0},
}
# SA Constants for OptimizationControlWidget
SA_MIN_TEMPERATURE = 0.01
SA_REHEAT_FACTOR = 1.8
SA_REHEAT_THRESHOLD_TEMP = 5.0
APP_VERSION = "1.2.0"
ORGANIZATION_NAME = "HamoonSoleimani" # For QSettings
APPLICATION_NAME = "ConstructionSiteOptimizer" # For QSettings
# --- Site Element Definition ---
class SiteElement:
    """Class representing a construction site element with position, size, and properties"""

    def __init__(self, name, x=0.0, y=0.0, width=10.0, height=10.0, rotation=0.0):
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)
        self.rotation = float(rotation)  # in degrees

        element_defaults = SITE_ELEMENTS.get(name, {})
        raw_color_input = element_defaults.get('color', QColor(200, 200, 200))

        # Ensure self.color is a new QColor instance, robust to input type
        if isinstance(raw_color_input, QColor):
            self.color = QColor(raw_color_input)
        elif isinstance(raw_color_input, (tuple, list)) and len(raw_color_input) in [3, 4]:
            self.color = QColor(*raw_color_input)
        elif isinstance(raw_color_input, str):
            try:
                self.color = QColor(raw_color_input)
            except Exception:
                 logger.warning(f"Invalid color string '{raw_color_input}' for {name}. Using default.")
                 self.color = QColor(200, 200, 200)
        else:
            logger.warning(f"Unexpected color type '{type(raw_color_input)}' for {name}. Using default.")
            self.color = QColor(200, 200, 200)

        self.movable = element_defaults.get('movable', True)
        self.priority = element_defaults.get('priority', 5)
        self.placeable_inside_building = element_defaults.get('placeable_inside_building', False) # New attribute

        self.selected = False
        self.connected_elements = []

        # New attributes for internal placement
        self.is_placed_inside_building = False
        self.internal_placement_info = None # Could store e.g. {'floor': 0, 'local_x': ..., 'local_y': ...} if multi-floor internal placement was modeled
        self.original_external_x = None
        self.original_external_y = None
        self.original_external_width = None
        self.original_external_height = None
        self.original_external_rotation = None

    def set_as_internally_placed(self, building_element: 'SiteElement', internal_x_offset_from_building_corner=0.0, internal_y_offset_from_building_corner=0.0):
        if not self.is_placed_inside_building: # Store original only once
            self.original_external_x = self.x
            self.original_external_y = self.y
            self.original_external_width = self.width
            self.original_external_height = self.height
            self.original_external_rotation = self.rotation

        self.is_placed_inside_building = True
        # For visualization and high-level logic, place it *at* a corner of the building
        # A more complex internal layout algorithm would be needed for precise internal packing.
        # For now, we'll place it at the building's origin + small offset to signify it's "absorbed"
        # The actual area it occupies comes from building_element's available internal area.
        self.x = building_element.x + internal_x_offset_from_building_corner # Or specific calculated internal position
        self.y = building_element.y + internal_y_offset_from_building_corner
        # Its "external" footprint becomes negligible or handled differently in overlap/distance.
        # For simplicity in overlap, we could make width/height very small, or rely on a flag.
        # Let's keep its original width/height for area calculations, but its position indicates it's inside.
        # The OptimizationEngine's overlap check will need to be aware of `is_placed_inside_building`.
        # self.width = 0.1 # Option: make it externally tiny
        # self.height = 0.1 # Option: make it externally tiny
        self.movable = False # Once inside, it's not movable by the general algorithm in the same way
        logger.debug(f"Element {self.name} marked as internally placed within {building_element.name}.")

    def set_as_externally_placed(self):
        """Restores the element to its external placement properties."""
        if self.is_placed_inside_building:
            if self.original_external_x is not None: # Check if originals were stored
                self.x = self.original_external_x
                self.y = self.original_external_y
                self.width = self.original_external_width
                self.height = self.original_external_height
                self.rotation = self.original_external_rotation
            else: # Fallback if somehow originals weren't stored (e.g. started as internal)
                  # This case needs careful thought if elements can *start* internally without prior external state.
                  # For now, assume elements start externally and can be moved inside.
                pass
            self.is_placed_inside_building = False
            self.movable = SITE_ELEMENTS.get(self.name, {}).get('movable', True) # Reset movability
            logger.debug(f"Element {self.name} marked as externally placed.")

    def area(self):
        """Return the area of the element in square meters"""
        return self.width * self.height

    def contains_point(self, point_x, point_y):
        """Check if the element contains the given point, considering rotation."""
        # Transform point to element's local coordinate system
        # (where element is at origin, unrotated)
        cx, cy = self.x + self.width/2, self.y + self.height/2 # Element center

        # Translate point so that element center is origin
        translated_x = point_x - cx
        translated_y = point_y - cy

        # Rotate point inversely around origin
        if self.rotation == 0:
            rx, ry = translated_x, translated_y
        else:
            rad = math.radians(-self.rotation) # Inverse rotation
            cos_rad = math.cos(rad)
            sin_rad = math.sin(rad)
            rx = translated_x * cos_rad - translated_y * sin_rad
            ry = translated_x * sin_rad + translated_y * cos_rad

        # Check if rotated point is within the unrotated rectangle (now centered at origin)
        half_width = self.width / 2
        half_height = self.height / 2
        return (-half_width <= rx <= half_width and
                -half_height <= ry <= half_height)

    def distance_to(self, other_element):
        """Calculate the minimum distance to another element (edge to edge)."""
        if self.overlaps(other_element): # If they overlap, distance is 0
            return 0.0

        # For non-rotated AABBs (Axis-Aligned Bounding Boxes)
        if self.rotation == 0 and other_element.rotation == 0:
            dx = 0.0
            if self.x + self.width < other_element.x: # self is to the left of other
                dx = other_element.x - (self.x + self.width)
            elif other_element.x + other_element.width < self.x: # other is to the left of self
                dx = self.x - (other_element.x + other_element.width)

            dy = 0.0
            if self.y + self.height < other_element.y: # self is above other
                dy = other_element.y - (self.y + self.height)
            elif other_element.y + other_element.height < self.y: # other is above self
                dy = self.y - (other_element.y + other_element.height)

            if dx == 0 and dy == 0: # Edges touch or overlap (should be caught by self.overlaps)
                 return 0.0
            return math.sqrt(dx**2 + dy**2)

        # Approximation for rotated elements: center-to-center minus "average radii"
        # This is a heuristic. Accurate distance for rotated polygons is complex (e.g., GJK algorithm).
        my_center_x, my_center_y = self.get_center()
        other_center_x, other_center_y = other_element.get_center()
        center_distance = math.hypot(my_center_x - other_center_x, my_center_y - other_center_y)

        # Approximate "radius" as half the diagonal (overestimate for distance calculation)
        # This aims to ensure that if this approx distance is > 0, they are definitely not overlapping
        my_radius_approx = math.hypot(self.width, self.height) / 2.0
        other_radius_approx = math.hypot(other_element.width, other_element.height) / 2.0

        edge_to_edge_approx = center_distance - (my_radius_approx + other_radius_approx)
        return max(0.0, edge_to_edge_approx) # Distance cannot be negative

    def overlaps(self, other_element, buffer=0.0):
        """
        Check if this element overlaps with another element, considering a buffer.
        Uses Separating Axis Theorem (SAT) for accurate rotated rectangle collision.
        The buffer is applied as an AABB expansion/contraction to the elements before SAT.
        """
        # If buffer is 0 and no rotation, use simple AABB check for speed
        if buffer == 0 and self.rotation == 0 and other_element.rotation == 0:
            return not (self.x + self.width <= other_element.x or # element1 is to the left of element2
                       other_element.x + other_element.width <= self.x or # element2 is to the left of element1
                       self.y + self.height <= other_element.y or # element1 is above element2
                       other_element.y + other_element.height <= self.y)  # element2 is above element1

        # Create "buffered" versions of elements for SAT.
        # self.expand() creates new SiteElement instances with adjusted AABB dimensions.
        # If buffer is positive, elements are made larger.
        # If buffer is negative (e.g., to allow "settling"), elements are made smaller.
        # The SAT is then performed on these (potentially AABB-expanded) elements.
        
        # Use the original element if buffer is zero to avoid object creation
        buffered_self = self.expand(buffer) if buffer != 0.0 else self
        buffered_other = other_element.expand(buffer) if buffer != 0.0 else other_element
        
        # If expand resulted in invalid (e.g. negative dim) elements due to large negative buffer,
        # it might lead to issues. expand() tries to keep dims >= 0.1.
        if buffered_self.width <=0 or buffered_self.height <=0 or \
           buffered_other.width <=0 or buffered_other.height <=0:
            logger.warning(f"Overlap check with invalid dimension after buffering for {self.name} or {other_element.name}. Defaulting to no overlap.")
            return False # Cannot reliably check overlap with zero/negative dimension element

        polygons_vertices = [buffered_self.get_corners_as_vectors(), buffered_other.get_corners_as_vectors()]

        # Iterate through each polygon (self and other)
        for poly_idx in range(len(polygons_vertices)):
            current_polygon_vertices = polygons_vertices[poly_idx]
            
            # Iterate through each edge of the current polygon
            for i1 in range(len(current_polygon_vertices)):
                i2 = (i1 + 1) % len(current_polygon_vertices) # Next vertex, wraps around
                p1_vec = current_polygon_vertices[i1]
                p2_vec = current_polygon_vertices[i2]

                # Calculate the edge vector
                edge_x = p2_vec[0] - p1_vec[0]
                edge_y = p2_vec[1] - p1_vec[1]

                # Calculate the perpendicular vector (normal to the edge)
                # For a 2D vector (dx, dy), a perpendicular vector is (-dy, dx)
                normal_x = -edge_y
                normal_y = edge_x
                
                # --- Project all vertices of *both* polygons onto this normal ---
                # (Original code projected polygon[0] (self) onto its own normals, then polygon[1] (other) onto its own normals.
                #  SAT requires projecting *both* polygons onto *each* unique axis derived from *both* polygons' edges.)
                #  The current loop structure correctly iterates through axes from both polygons.

                # Project polygon A (buffered_self)
                minA, maxA = None, None
                for vertex_polyA in polygons_vertices[0]: 
                    projected_val = normal_x * vertex_polyA[0] + normal_y * vertex_polyA[1]
                    if minA is None or projected_val < minA: minA = projected_val
                    if maxA is None or projected_val > maxA: maxA = projected_val
                
                # Project polygon B (buffered_other)
                minB, maxB = None, None
                for vertex_polyB in polygons_vertices[1]:
                    projected_val = normal_x * vertex_polyB[0] + normal_y * vertex_polyB[1]
                    if minB is None or projected_val < minB: minB = projected_val
                    if maxB is None or projected_val > maxB: maxB = projected_val

                # Check for separation on this axis
                # If the projections do not overlap, we found a separating axis
                if maxA < minB or maxB < minA:
                    return False # No overlap, separating axis found

        # If no separating axis was found after checking all edge normals of both polygons
        return True # Polygons overlap

    def get_corners_as_vectors(self):
        """Return corners as a list of [x,y] lists (vectors) for SAT."""
        corners_tuples = self.get_corners()
        return [[c[0], c[1]] for c in corners_tuples]

    def get_corners(self):
        """Return the four corners of the element considering rotation as (x,y) tuples."""
        center_x = self.x + self.width/2
        center_y = self.y + self.height/2
        hw, hh = self.width/2, self.height/2
        
        # Corners relative to center, before rotation
        # Order: top-left, top-right, bottom-right, bottom-left (when unrotated, y points down)
        # Or: bottom-left, bottom-right, top-right, top-left (if y points up as in standard math)
        # Let's assume: (-hw, -hh) is one corner, (hw, -hh) another, etc. in local frame.
        # Standard drawing often has Y increasing downwards from top-left.
        # Matplotlib plotting typically has Y increasing upwards from bottom-left.
        # Let's be consistent: assuming (x,y) is top-left for the unrotated rectangle.
        # Corners relative to (0,0) if element was at origin, then add center translation.
        corners_rel_to_origin_unrotated = [
            (0, 0), (self.width, 0), (self.width, self.height), (0, self.height)
        ]
        # Transform these to be relative to the element's actual center for rotation
        corners_rel_to_center = [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
        ]

        rotated_corners_tuples = []

        if self.rotation == 0:
            for rel_x, rel_y in corners_rel_to_center:
                rotated_corners_tuples.append((center_x + rel_x, center_y + rel_y))
        else:
            rad = math.radians(self.rotation)
            cos_val, sin_val = math.cos(rad), math.sin(rad)
            for rel_x, rel_y in corners_rel_to_center:
                # Standard 2D rotation:
                # x' = x*cos(theta) - y*sin(theta)
                # y' = x*sin(theta) + y*cos(theta)
                rotated_rel_x = rel_x * cos_val - rel_y * sin_val
                rotated_rel_y = rel_x * sin_val + rel_y * cos_val
                rotated_corners_tuples.append((center_x + rotated_rel_x, center_y + rotated_rel_y))
        return rotated_corners_tuples

    def get_center(self):
        """Return the center point (x,y) of the element."""
        return (self.x + self.width/2, self.y + self.height/2)

    def expand(self, buffer):
        """
        Return a new element whose Axis-Aligned Bounding Box (AABB) is expanded
        or contracted by the buffer amount *before* rotation is considered.
        The returned element will have the same rotation as the original.
        The width and height of the new element will not be less than 0.1.
        """
        new_x = self.x - buffer
        new_y = self.y - buffer
        new_width = self.width + 2 * buffer
        new_height = self.height + 2 * buffer

        # Ensure width and height don't become effectively zero or negative
        # This is important if buffer is negative and large.
        if new_width < 0.1: new_width = 0.1
        if new_height < 0.1: new_height = 0.1
        
        # Use self.__class__ to allow for correct instantiation if SiteElement is subclassed
        expanded = self.__class__( 
            self.name,
            new_x, new_y,
            new_width, new_height,
            self.rotation # Rotation is maintained
        )
        # Copy other relevant properties
        expanded.color = QColor(self.color) # QColor is a value type, so this is a copy
        expanded.movable = self.movable
        expanded.priority = self.priority
        expanded.selected = self.selected
        # expanded.connected_elements = list(self.connected_elements) # If deep copy needed for connections
        return expanded
    
# --- Path Planner ---
class PathPlanner:
    """
    Class responsible for planning and managing routes between construction site elements.
    It uses an A* algorithm on a grid representation of the site to find paths
    and considers obstacles, safety buffers, and different types of routes.
    It distinguishes between external elements (obstacles) and internal elements (whose
    path connections might originate from the main building).
    """

    def __init__(self,
                 plot_width: float,
                 plot_height: float,
                 all_site_elements: List['SiteElement'], # All elements, including internal ones
                 constraints: Dict,
                 grid_resolution: float = 1.0,
                 default_path_width_vehicle: float = 3.5,
                 default_path_width_pedestrian: float = 1.5):
        """
        Initialize the PathPlanner.

        Args:
            plot_width: Total width of the construction site plot in meters.
            plot_height: Total height of the construction site plot in meters.
            all_site_elements: A list of ALL SiteElement objects (internal and external).
                               PathPlanner will filter these for obstruction grid creation.
            constraints: Dictionary of constraint values (e.g., 'safety_buffer', 'min_path_width').
            grid_resolution: Size of each cell in the pathfinding grid (meters).
            default_path_width_vehicle: Default width for vehicle paths for visualization.
            default_path_width_pedestrian: Default width for pedestrian paths for visualization.
        """
        self.plot_width = float(plot_width)
        self.plot_height = float(plot_height)
        self.all_elements = list(all_site_elements) # Store a copy of all elements
        self.constraints = constraints.copy()

        if grid_resolution <= 0.1: # Ensure a minimum practical resolution
            logger.warning(f"Grid resolution {grid_resolution} too small. Clamping to 0.1m.")
            self.grid_resolution = 0.1
        else:
            self.grid_resolution = float(grid_resolution)

        self.default_path_width_vehicle = default_path_width_vehicle
        self.default_path_width_pedestrian = default_path_width_pedestrian

        self.vehicle_routes: List[Dict] = []
        self.pedestrian_routes: List[Dict] = []
        self.obstruction_grid: Optional[np.ndarray] = None # Cache the grid

        # Define a standard site entry point (e.g., bottom-center of the plot)
        self.site_entry_world_coord = self._get_site_entry_point()

        logger.info(f"PathPlanner initialized for plot {self.plot_width}x{self.plot_height}m, "
                    f"grid res: {self.grid_resolution}m, {len(self.all_elements)} total elements.")

    def _get_site_entry_point(self) -> Dict[str, float]:
        """Defines the primary site entry point coordinates."""
        # Could be made more flexible, e.g., user-defined or a "SiteEntrance" element
        return {
            'x': self.plot_width / 2.0,
            'y': self.plot_height - (self.grid_resolution * 0.5) # Slightly inside the bottom edge
        }

    def update_elements(self, new_all_site_elements: List['SiteElement']):
        """Updates the elements list and clears cached grid."""
        self.all_elements = list(new_all_site_elements)
        self.obstruction_grid = None # Invalidate cached grid
        logger.debug("PathPlanner elements updated, obstruction grid invalidated.")


    def create_routes(self, force_rebuild_grid: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """
        Generates vehicle and pedestrian routes based on predefined connections
        and the current layout of site elements.

        Args:
            force_rebuild_grid: If True, rebuilds the obstruction grid even if one exists.

        Returns:
            A tuple (vehicle_routes, pedestrian_routes). Each route is a dict:
            {'from_name', 'to_name', 'from_coord', 'to_coord', 'path', 'width', 'type'}.
        """
        self.vehicle_routes = []
        self.pedestrian_routes = []

        if not self.all_elements or self.plot_width <= 0 or self.plot_height <= 0:
            logger.warning("Cannot create routes: No elements or invalid plot dimensions.")
            return [], []

        if self.obstruction_grid is None or force_rebuild_grid:
            self.obstruction_grid = self._create_obstruction_grid()

        if self.obstruction_grid is None: # Still None after trying to create
            logger.error("Failed to create obstruction grid. Cannot generate routes.")
            return [], []

        elements_by_name = {el.name: el for el in self.all_elements}
        building_element = elements_by_name.get('Building')


        # --- Route Connection Definitions ---
        # These could be loaded from config or dynamically determined.
        # 'target_type' can be 'center', 'edge_closest_to_source', 'specific_point_on_target'
        vehicle_connections = [
            {'from': 'site_entry', 'to': 'Material Storage', 'width_factor': 1.2, 'target_type': 'center'},
            {'from': 'site_entry', 'to': 'Machinery Parking', 'width_factor': 1.2, 'target_type': 'center'},
            {'from': 'Material Storage', 'to': 'Building', 'width_factor': 1.0, 'target_type': 'edge_closest_to_source'},
            {'from': 'Material Storage', 'to': 'Welding Workshop', 'width_factor': 1.0, 'target_type': 'center'},
            {'from': 'Machinery Parking', 'to': 'Building', 'width_factor': 1.0, 'target_type': 'edge_closest_to_source'},
            # Add more connections as needed
        ]
        pedestrian_connections = [
            {'from': 'Building', 'to': 'Contractor Office', 'target_type': 'center'},
            {'from': 'Building', 'to': 'Consultant Office', 'target_type': 'center'},
            {'from': 'Building', 'to': 'Client Office', 'target_type': 'center'},
            {'from': 'Workers Dormitory', 'to': 'Building', 'target_type': 'edge_closest_to_source'},
            {'from': 'Workers Dormitory', 'to': 'Sanitary Facilities', 'target_type': 'center'},
            {'from': 'Security Post', 'to': 'site_entry', 'target_type': 'center'},
            # Example: Office to Office paths
            {'from': 'Contractor Office', 'to': 'Consultant Office', 'target_type': 'center'},
        ]
        # Add inter-office and office-to-sanitary connections dynamically
        office_names = [name for name in SITE_ELEMENTS if "Office" in name]
        for i, office1_name in enumerate(office_names):
            pedestrian_connections.append({'from': office1_name, 'to': 'Sanitary Facilities', 'target_type': 'center'})
            for office2_name in office_names[i+1:]:
                pedestrian_connections.append({'from': office1_name, 'to': office2_name, 'target_type': 'center'})


        base_path_width = self.constraints.get('min_path_width', 3.0)

        connection_sets_config = [
            {'defs': vehicle_connections, 'store': self.vehicle_routes, 'type': 'vehicle', 'base_width': base_path_width * 1.2},
            {'defs': pedestrian_connections, 'store': self.pedestrian_routes, 'type': 'pedestrian', 'base_width': base_path_width * 0.8}
        ]

        for conn_set_cfg in connection_sets_config:
            logger.debug(f"Processing {conn_set_cfg['type']} connections.")
            for conn_def in conn_set_cfg['defs']:
                start_el_name, end_el_name = conn_def['from'], conn_def['to']

                start_coord_wc = self._get_connection_point_for_element(start_el_name, elements_by_name, building_element, 'source', None)
                # For target point, we might need source to determine 'edge_closest_to_source'
                end_coord_wc = self._get_connection_point_for_element(end_el_name, elements_by_name, building_element, 'target', start_coord_wc, conn_def.get('target_type', 'center'))


                if not start_coord_wc:
                    logger.warning(f"Could not resolve start point for '{start_el_name}' in {conn_set_cfg['type']} path.")
                    continue
                if not end_coord_wc:
                    logger.warning(f"Could not resolve end point for '{end_el_name}' in {conn_set_cfg['type']} path.")
                    continue

                # Avoid pathfinding if start and end are identical or extremely close
                if math.hypot(start_coord_wc['x'] - end_coord_wc['x'], start_coord_wc['y'] - end_coord_wc['y']) < self.grid_resolution:
                    logger.debug(f"Skipping path for {conn_set_cfg['type']}: '{start_el_name}' -> '{end_el_name}' (start/end too close).")
                    # Add a direct line segment as path
                    trivial_path = [start_coord_wc, end_coord_wc]
                    conn_set_cfg['store'].append({
                        'from_name': start_el_name, 'to_name': end_el_name,
                        'from_coord': start_coord_wc, 'to_coord': end_coord_wc,
                        'path': trivial_path,
                        'width': conn_set_cfg['base_width'] * conn_def.get('width_factor', 1.0),
                        'type': conn_set_cfg['type']
                    })
                    continue

                logger.debug(f"Attempting to find {conn_set_cfg['type']} path: '{start_el_name}' ({start_coord_wc['x']:.1f},{start_coord_wc['y']:.1f}) to "
                             f"'{end_el_name}' ({end_coord_wc['x']:.1f},{end_coord_wc['y']:.1f}).")

                path_segment_wc = self._find_path(start_coord_wc, end_coord_wc, self.obstruction_grid, conn_set_cfg['type'])

                if path_segment_wc:
                    conn_set_cfg['store'].append({
                        'from_name': start_el_name, 'to_name': end_el_name,
                        'from_coord': start_coord_wc, 'to_coord': end_coord_wc,
                        'path': path_segment_wc,
                        'width': conn_set_cfg['base_width'] * conn_def.get('width_factor', 1.0),
                        'type': conn_set_cfg['type']
                    })
                    logger.debug(f"Path FOUND for {conn_set_cfg['type']}: '{start_el_name}' -> '{end_el_name}', {len(path_segment_wc)} points.")
                else:
                    logger.warning(f"Path NOT FOUND for {conn_set_cfg['type']}: '{start_el_name}' -> '{end_el_name}'.")

        return self.vehicle_routes, self.pedestrian_routes

    def _get_connection_point_for_element(self, el_name: str, elements_by_name: Dict[str, 'SiteElement'],
                                          building_element: Optional['SiteElement'],
                                          point_role: str, # 'source' or 'target'
                                          other_point_wc: Optional[Dict[str,float]] = None, # Used for 'edge_closest_to_source'
                                          target_type: str = 'center'
                                          ) -> Optional[Dict[str, float]]:
        """
        Resolves an element name to its world coordinate connection point.
        Handles 'site_entry', internally placed elements (using Building as proxy),
        and different target_type strategies.
        """
        if el_name == 'site_entry':
            return self.site_entry_world_coord.copy()

        element = elements_by_name.get(el_name)
        if not element:
            logger.warning(f"Element '{el_name}' not found in layout for path connection.")
            return None

        # If element is placed inside the building, its connection point is on the building itself.
        if getattr(element, 'is_placed_inside_building', False):
            if not building_element:
                logger.error(f"Building element missing, cannot resolve connection point for internal element '{el_name}'.")
                return None
            # Use the building as the effective element for path connection
            effective_element = building_element
        else:
            effective_element = element

        # Now determine the point on the effective_element based on target_type
        if point_role == 'target' and target_type == 'edge_closest_to_source' and other_point_wc:
            return self._get_closest_point_on_element_edge(effective_element, other_point_wc)
        # elif point_role == 'target' and target_type == 'specific_access_point':
            # return self._get_defined_access_point(effective_element) # Needs more logic
        else: # Default to center ('center' or source point)
            center_x, center_y = effective_element.get_center()
            return {'x': center_x, 'y': center_y}

    def _get_closest_point_on_element_edge(self, element: 'SiteElement', external_point_wc: Dict[str, float]) -> Dict[str, float]:
        """
        Finds the point on the perimeter of the (unrotated, AABB) element
        that is closest to the external_point_wc.
        For rotated elements, this is an approximation using AABB. A more accurate method
        would project onto the rotated polygon's edges.
        """
        ex, ey = external_point_wc['x'], external_point_wc['y']
        el_x, el_y = element.x, element.y
        el_w, el_h = element.width, element.height

        # Clamp external point to element's AABB boundary
        closest_x = max(el_x, min(ex, el_x + el_w))
        closest_y = max(el_y, min(ey, el_y + el_h))
        
        # If the point is already inside, this will snap to it.
        # We need to push it to an edge if it's inside.
        # This simple clamping works if the external point is outside.
        # If external_point_wc is inside element, this needs refinement to pick an edge.
        # For now, assuming external_point_wc is typically outside for "edge_closest_to_source".

        # A more robust way for AABB:
        # Find distances to each of the four infinitely extended lines of the AABB edges
        # and project. Then check if projection is within segment.
        # Or, simpler: find which region the point is in (Voronoi regions of AABB)
        # and project to nearest feature (vertex or edge).

        # Current simple clamp is okay as a starting point.
        # To make it always be on the perimeter:
        if el_x < closest_x < el_x + el_w and el_y < closest_y < el_y + el_h : # Point was inside, project to nearest edge
            dist_to_left = ex - el_x
            dist_to_right = (el_x + el_w) - ex
            dist_to_top = ey - el_y
            dist_to_bottom = (el_y + el_h) - ey
            min_dist_edge = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
            if min_dist_edge == dist_to_left: closest_x = el_x
            elif min_dist_edge == dist_to_right: closest_x = el_x + el_w
            elif min_dist_edge == dist_to_top: closest_y = el_y
            else: closest_y = el_y + el_h # dist_to_bottom

        return {'x': closest_x, 'y': closest_y}


    def _create_obstruction_grid(self) -> Optional[np.ndarray]:
        """
        Creates a 2D grid representing the site, marking cells as obstructed (1) or free (0).
        Only considers EXTERNALLY placed elements as obstacles.
        The main 'Building' element is also treated as an obstacle.
        """
        if self.plot_width <= 0 or self.plot_height <= 0 or self.grid_resolution <= 0:
            logger.error("Invalid parameters for obstruction grid creation.")
            return None

        grid_cols = int(math.ceil(self.plot_width / self.grid_resolution))
        grid_rows = int(math.ceil(self.plot_height / self.grid_resolution))
        if grid_cols == 0: grid_cols = 1
        if grid_rows == 0: grid_rows = 1

        grid = np.zeros((grid_rows, grid_cols), dtype=np.int8)
        safety_buffer = self.constraints.get('safety_buffer', 1.0)

        # Filter for elements that are physically present on the external site
        external_obstacles = [
            el for el in self.all_elements if not getattr(el, 'is_placed_inside_building', False)
        ]
        # The main 'Building' is always an external obstacle. Ensure it's in the list if not already.
        building_el = next((el for el in self.all_elements if el.name == 'Building'), None)
        if building_el and building_el not in external_obstacles: # Should be if logic is correct
             # This case is unlikely if external_obstacles is derived from all_elements where Building is not internal
             pass


        logger.debug(f"Creating obstruction grid ({grid_rows}x{grid_cols}) with safety buffer: {safety_buffer}m "
                     f"from {len(external_obstacles)} external elements.")

        for element in external_obstacles:
            # Consider elements expanded by the safety buffer for pathfinding.
            # The SiteElement.expand() method should create a new element with adjusted AABB.
            try:
                # For the Building, we might want a slightly smaller buffer or no buffer if paths need to go right to its edge.
                # For now, apply buffer uniformly.
                current_buffer = safety_buffer
                if element.name == "Building" and self.constraints.get('buffer_building_override', False):
                    current_buffer = self.constraints.get('building_safety_buffer', safety_buffer * 0.5) # e.g. smaller buffer for building

                buffered_element = element.expand(current_buffer)
            except Exception as e:
                logger.error(f"Error expanding element {element.name} for obstruction grid: {e}")
                continue

            corners_wc = buffered_element.get_corners()
            if not corners_wc:
                logger.warning(f"Could not get corners for buffered external element: {buffered_element.name}")
                continue

            # Optimized iteration over grid cells within the AABB of the buffered element
            min_x_wc = min(c[0] for c in corners_wc); max_x_wc = max(c[0] for c in corners_wc)
            min_y_wc = min(c[1] for c in corners_wc); max_y_wc = max(c[1] for c in corners_wc)

            min_gx = int(min_x_wc / self.grid_resolution); max_gx = int(math.ceil(max_x_wc / self.grid_resolution))
            min_gy = int(min_y_wc / self.grid_resolution); max_gy = int(math.ceil(max_y_wc / self.grid_resolution))

            for gy_idx in range(max(0, min_gy), min(grid_rows, max_gy)):
                for gx_idx in range(max(0, min_gx), min(grid_cols, max_gx)):
                    if grid[gy_idx, gx_idx] == 1: continue # Already an obstacle

                    cell_center_x_wc = (gx_idx + 0.5) * self.grid_resolution
                    cell_center_y_wc = (gy_idx + 0.5) * self.grid_resolution

                    # Use buffered_element.contains_point for accurate check with rotation
                    if buffered_element.contains_point(cell_center_x_wc, cell_center_y_wc):
                        grid[gy_idx, gx_idx] = 1
        self.obstruction_grid = grid
        return grid

    def _find_path(self, start_wc: Dict[str, float], end_wc: Dict[str, float],
                   grid: np.ndarray, path_type: str) -> Optional[List[Dict[str, float]]]:
        """
        Implements A* pathfinding on the grid.
        Args are world coordinates. Returns path in world coordinates.
        """
        grid_rows, grid_cols = grid.shape

        # Convert world start/end to grid coordinates, clamping to grid boundaries
        start_gx = max(0, min(int(start_wc['x'] / self.grid_resolution), grid_cols - 1))
        start_gy = max(0, min(int(start_wc['y'] / self.grid_resolution), grid_rows - 1))
        end_gx = max(0, min(int(end_wc['x'] / self.grid_resolution), grid_cols - 1))
        end_gy = max(0, min(int(end_wc['y'] / self.grid_resolution), grid_rows - 1))

        start_node_gc = (start_gx, start_gy)
        end_node_gc = (end_gx, end_gy)

        logger.debug(f"[{path_type.upper()}] A* World Start ({start_wc['x']:.1f},{start_wc['y']:.1f}) -> Grid Start {start_node_gc}")
        logger.debug(f"[{path_type.upper()}] A* World End ({end_wc['x']:.1f},{end_wc['y']:.1f}) -> Grid End {end_node_gc}")

        # Check if start/end grid nodes are obstructed; if so, find nearest valid point.
        # This is crucial if connection points are centers of large elements.
        if grid[start_node_gc[1], start_node_gc[0]] == 1:
            logger.debug(f"[{path_type.upper()}] Start node {start_node_gc} is obstacle. Finding nearest valid...")
            valid_start_gc = self._find_nearest_valid_point(start_node_gc, grid, max_search_radius_cells=10)
            if valid_start_gc:
                start_node_gc = valid_start_gc
                logger.debug(f"[{path_type.upper()}] Relocated start to valid grid point: {start_node_gc}")
            else:
                logger.warning(f"[{path_type.upper()}] No valid start point near {start_node_gc} for path from {start_wc}.")
                return None

        if grid[end_node_gc[1], end_node_gc[0]] == 1:
            logger.debug(f"[{path_type.upper()}] End node {end_node_gc} is obstacle. Finding nearest valid...")
            valid_end_gc = self._find_nearest_valid_point(end_node_gc, grid, max_search_radius_cells=10)
            if valid_end_gc:
                end_node_gc = valid_end_gc
                logger.debug(f"[{path_type.upper()}] Relocated end to valid grid point: {end_node_gc}")
            else:
                logger.warning(f"[{path_type.upper()}] No valid end point near {end_node_gc} for path to {end_wc}.")
                return None

        if start_node_gc == end_node_gc:
            logger.debug(f"[{path_type.upper()}] Start and end nodes are identical ({start_node_gc}). Path is trivial.")
            return [start_wc.copy(), end_wc.copy()] # Return path with original world start/end

        # A* Algorithm
        open_set = PriorityQueue() # Stores (f_score, tie_breaker, node_gc_tuple)
        tie_breaker_count = 0
        open_set.put((0, tie_breaker_count, start_node_gc))
        
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_node_gc: None}
        g_score: Dict[Tuple[int, int], float] = {start_node_gc: 0.0}
        
        heuristic = lambda p1_gc, p2_gc: math.hypot(p1_gc[0] - p2_gc[0], p1_gc[1] - p2_gc[1]) * self.grid_resolution # Heuristic in meters
        f_score: Dict[Tuple[int, int], float] = {start_node_gc: heuristic(start_node_gc, end_node_gc)}

        # Movement: (dx, dy, cost_multiplier for grid_resolution)
        # Allow diagonal movement with higher cost.
        neighbors_deltas = [
            (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),  # Cardinal
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),          # Diagonal
            (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))
        ]

        while not open_set.empty():
            _, _, current_gc = open_set.get() # Get node with the lowest f_score

            if current_gc == end_node_gc: # Goal reached
                path_gc_list = []
                temp_gc: Optional[Tuple[int,int]] = current_gc
                while temp_gc is not None:
                    path_gc_list.append(temp_gc)
                    temp_gc = came_from[temp_gc]
                path_gc_list.reverse()

                # Convert grid path to world coordinates path
                path_wc_list = [start_wc.copy()] # Start with precise original world start
                for i in range(1, len(path_gc_list) -1): # Exclude grid start/end
                    node_gx, node_gy = path_gc_list[i]
                    path_wc_list.append({
                        'x': node_gx * self.grid_resolution + self.grid_resolution / 2.0,
                        'y': node_gy * self.grid_resolution + self.grid_resolution / 2.0
                    })
                path_wc_list.append(end_wc.copy()) # End with precise original world end
                
                return self._smooth_path_douglas_peucker(path_wc_list) # Apply smoothing

            for dx_gc, dy_gc, cost_multiplier in neighbors_deltas:
                neighbor_gc = (current_gc[0] + dx_gc, current_gc[1] + dy_gc)

                if not (0 <= neighbor_gc[0] < grid_cols and 0 <= neighbor_gc[1] < grid_rows):
                    continue # Out of bounds
                if grid[neighbor_gc[1], neighbor_gc[0]] == 1:
                    continue # Obstacle

                move_cost = cost_multiplier * self.grid_resolution # Actual cost in meters
                tentative_g_score = g_score[current_gc] + move_cost

                if neighbor_gc not in g_score or tentative_g_score < g_score[neighbor_gc]:
                    came_from[neighbor_gc] = current_gc
                    g_score[neighbor_gc] = tentative_g_score
                    f_score[neighbor_gc] = tentative_g_score + heuristic(neighbor_gc, end_node_gc)
                    tie_breaker_count += 1
                    open_set.put((f_score[neighbor_gc], tie_breaker_count, neighbor_gc))
        
        logger.warning(f"[{path_type.upper()}] A* search completed. No path found from {start_node_gc} to {end_node_gc}.")
        return None

    def _find_nearest_valid_point(self, point_gc: Tuple[int, int], grid: np.ndarray,
                                  max_search_radius_cells: int = 15) -> Optional[Tuple[int, int]]:
        """
        Finds the nearest non-obstacle grid cell using BFS-like expansion.
        """
        grid_rows, grid_cols = grid.shape
        if not (0 <= point_gc[0] < grid_cols and 0 <= point_gc[1] < grid_rows):
            logger.warning(f"Initial point {point_gc} for _find_nearest_valid_point is out of bounds.")
            return None
        if grid[point_gc[1], point_gc[0]] == 0:
            return point_gc # Already valid

        queue = PriorityQueue() # (distance_sq, (gx, gy)) - A* like search for nearest valid
        queue.put((0, point_gc))
        visited = {point_gc}
        
        # Check in increasing radius (spiral or box expansion)
        for r_offset in range(1, max_search_radius_cells + 1):
            for dr_idx in range(-r_offset, r_offset + 1):
                for dc_idx in range(-r_offset, r_offset + 1):
                    # Only check points on the perimeter of the current search box
                    if abs(dr_idx) != r_offset and abs(dc_idx) != r_offset:
                        continue

                    next_gc_candidate = (point_gc[0] + dc_idx, point_gc[1] + dr_idx)
                    if (0 <= next_gc_candidate[0] < grid_cols and
                        0 <= next_gc_candidate[1] < grid_rows and
                        next_gc_candidate not in visited):
                        
                        visited.add(next_gc_candidate)
                        if grid[next_gc_candidate[1], next_gc_candidate[0]] == 0:
                            return next_gc_candidate # Found nearest valid

        logger.debug(f"No valid point found within {max_search_radius_cells} cells of {point_gc}.")
        return None

    def _smooth_path_douglas_peucker(self, path_wc: List[Dict[str, float]], epsilon: float = 0.5) -> List[Dict[str, float]]:
        """
        Smooths a path using the Ramer-Douglas-Peucker algorithm.
        Epsilon is the max distance between original path and simplified one.
        """
        if not path_wc or len(path_wc) <= 2:
            return path_wc

        # Convert list of dicts to list of tuples for easier processing by helper
        points_tuples = [(p['x'], p['y']) for p in path_wc]

        def douglas_peucker_recursive_tuples(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            if not points or len(points) < 2: return points
            dmax_sq = 0.0
            index = 0
            end_idx = len(points) - 1

            for i in range(1, end_idx):
                d_sq = self._point_line_segment_distance_sq_tuples(points[i], points[0], points[end_idx])
                if d_sq > dmax_sq:
                    index = i
                    dmax_sq = d_sq
            
            results_tuples = []
            if dmax_sq > epsilon**2:
                rec_results1 = douglas_peucker_recursive_tuples(points[0 : index + 1])
                rec_results2 = douglas_peucker_recursive_tuples(points[index : end_idx + 1])
                results_tuples = rec_results1[:-1] + rec_results2
            else:
                results_tuples = [points[0], points[end_idx]]
            return results_tuples

        smoothed_tuples = douglas_peucker_recursive_tuples(points_tuples)
        return [{'x': pt[0], 'y': pt[1]} for pt in smoothed_tuples]


    def _point_line_segment_distance_sq_tuples(self, p: Tuple[float,float], a: Tuple[float,float], b: Tuple[float,float]) -> float:
        """Squared perpendicular distance from point p to line segment a-b (tuples)."""
        px, py = p; ax, ay = a; bx, by = b
        ab_x, ab_y = bx - ax, by - ay
        ap_x, ap_y = px - ax, py - ay
        len_sq_ab = ab_x**2 + ab_y**2
        if len_sq_ab == 0: return ap_x**2 + ap_y**2
        t = (ap_x * ab_x + ap_y * ab_y) / len_sq_ab
        if t < 0: closest_x, closest_y = ax, ay
        elif t > 1: closest_x, closest_y = bx, by
        else: closest_x, closest_y = ax + t * ab_x, ay + t * ab_y
        return (px - closest_x)**2 + (py - closest_y)**2

    def get_total_path_length(self, route_type: str = 'vehicle') -> float:
        """Calculates the total length of all paths of a given type."""
        total_length = 0.0
        routes_to_sum = self.vehicle_routes if route_type == 'vehicle' else self.pedestrian_routes
        for route_info in routes_to_sum:
            path_wc = route_info.get('path', [])
            if path_wc and len(path_wc) > 1:
                for i in range(len(path_wc) - 1):
                    p1, p2 = path_wc[i], path_wc[i+1]
                    total_length += math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
        return total_length

    def _segments_intersect(self, p1_wc: Dict[str,float], q1_wc: Dict[str,float],
                           p2_wc: Dict[str,float], q2_wc: Dict[str,float]) -> bool:
        """Checks if line segment p1q1 intersects p2q2."""
        # Using helper with tuple inputs
        p1 = (p1_wc['x'], p1_wc['y']); q1 = (q1_wc['x'], q1_wc['y'])
        p2 = (p2_wc['x'], p2_wc['y']); q2 = (q2_wc['x'], q2_wc['y'])
        return self._segments_intersect_tuples(p1, q1, p2, q2)

    def _segments_intersect_tuples(self, p1: Tuple, q1: Tuple, p2: Tuple, q2: Tuple) -> bool:
        """Helper for segment intersection with tuple inputs."""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-9: return 0 # Collinear (with tolerance)
            return 1 if val > 0 else 2 # Clockwise or Counterclockwise
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) + 1e-9 and q[0] >= min(p[0], r[0]) - 1e-9 and
                    q[1] <= max(p[1], r[1]) + 1e-9 and q[1] >= min(p[1], r[1]) - 1e-9)

        o1 = orientation(p1, q1, p2); o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1); o4 = orientation(p2, q2, q1)

        if o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0: # General case: different orientations
            if o1 != o2 and o3 != o4: return True
        # Collinear cases
        if o1 == 0 and on_segment(p1, p2, q1): return True
        if o2 == 0 and on_segment(p1, q2, q1): return True
        if o3 == 0 and on_segment(p2, p1, q2): return True
        if o4 == 0 and on_segment(p2, q1, q2): return True
        return False


    def calculate_route_interference(self, proximity_threshold: float = 1.5,
                                     parallel_angle_deg_threshold: float = 15.0) -> float:
        """
        Calculates a score representing interference between vehicle and pedestrian routes.
        Considers intersections and close parallel segments.
        """
        interference_score = 0.0
        if not self.vehicle_routes or not self.pedestrian_routes:
            return 0.0

        # Pre-calculate dot product threshold from angle
        cos_threshold = math.cos(math.radians(parallel_angle_deg_threshold))

        for v_route in self.vehicle_routes:
            v_path = v_route.get('path', [])
            if len(v_path) < 2: continue
            for i in range(len(v_path) - 1):
                v_p1, v_p2 = v_path[i], v_path[i+1]
                for p_route in self.pedestrian_routes:
                    p_path = p_route.get('path', [])
                    if len(p_path) < 2: continue
                    for j in range(len(p_path) - 1):
                        p_p1, p_p2 = p_path[j], p_path[j+1]
                        if self._segments_intersect(v_p1, v_p2, p_p1, p_p2):
                            interference_score += 10.0 # High penalty for direct intersection
                        elif self._are_segments_close_and_parallel(v_p1, v_p2, p_p1, p_p2,
                                                                  proximity_threshold, cos_threshold):
                            interference_score += 3.0 # Penalty for close parallel

        return interference_score

    def _point_to_segment_distance(self, p_wc: Dict[str,float], a_wc: Dict[str,float], b_wc: Dict[str,float]) -> float:
        """Shortest distance from point p_wc to line segment a_wc-b_wc."""
        return math.sqrt(self._point_line_segment_distance_sq_tuples(
            (p_wc['x'], p_wc['y']), (a_wc['x'], a_wc['y']), (b_wc['x'], b_wc['y'])
        ))

    def _are_segments_close_and_parallel(self, v_p1: Dict, v_p2: Dict, p_p1: Dict, p_p2: Dict,
                                         threshold_dist: float, parallel_cos_threshold: float) -> bool:
        """Checks if two path segments are nearly parallel and close."""
        v_dx, v_dy = v_p2['x'] - v_p1['x'], v_p2['y'] - v_p1['y']
        v_len = math.hypot(v_dx, v_dy)
        if v_len < 1e-6: return False
        v_nx, v_ny = v_dx / v_len, v_dy / v_len

        p_dx, p_dy = p_p2['x'] - p_p1['x'], p_p2['y'] - p_p1['y']
        p_len = math.hypot(p_dx, p_dy)
        if p_len < 1e-6: return False
        p_nx, p_ny = p_dx / p_len, p_dy / p_len

        dot_product_abs = abs(v_nx * p_nx + v_ny * p_ny)
        if dot_product_abs < parallel_cos_threshold: # Use pre-calculated cos_threshold
            return False # Not parallel enough

        # Check proximity: average distance of endpoints of one segment to the other segment
        # This is a simplified check for proximity. More robust checks exist (e.g., min distance between segments).
        dist_v1_to_p_seg = self._point_to_segment_distance(v_p1, p_p1, p_p2)
        dist_v2_to_p_seg = self._point_to_segment_distance(v_p2, p_p1, p_p2)
        dist_p1_to_v_seg = self._point_to_segment_distance(p_p1, v_p1, v_p2)
        dist_p2_to_v_seg = self._point_to_segment_distance(p_p2, v_p1, v_p2)
        
        # Check if the average of these distances is below threshold, or if any endpoint is very close
        avg_dist = (dist_v1_to_p_seg + dist_v2_to_p_seg + dist_p1_to_v_seg + dist_p2_to_v_seg) / 4.0
        min_endpoint_dist = min(dist_v1_to_p_seg, dist_v2_to_p_seg, dist_p1_to_v_seg, dist_p2_to_v_seg)

        return avg_dist < threshold_dist or min_endpoint_dist < threshold_dist / 2.0

class OptimizationEngine:
    """
    Manages the construction site elements, their layout, and the optimization process
    to find an optimal arrangement based on constraints and objectives.
    """

    def __init__(self, plot_width, plot_height, elements=None, constraints=None,
                space_requirements=None, building_ratio=0.45):
        """
        Initializes the OptimizationEngine.

        Args:
            plot_width (float): The width of the construction site plot in meters.
            plot_height (float): The height of the construction site plot in meters.
            elements (list, optional): A list of pre-existing SiteElement objects. Defaults to None.
            constraints (dict, optional): A dictionary of constraint parameters. Defaults to None.
            space_requirements (dict, optional): A dictionary of area requirements for elements. Defaults to None.
            building_ratio (float, optional): The proportion of the plot area dedicated to the main building. Defaults to 0.45.
        """
        logger.debug(f"OptimizationEngine __init__ called with plot: {plot_width}x{plot_height}, building_ratio: {building_ratio}")

        self.plot_width = float(plot_width)
        self.plot_height = float(plot_height)
        self.building_ratio = float(building_ratio)

        # Use provided lists/dicts or defaults if None
        self.elements: List[SiteElement] = elements if elements is not None else []
        self.constraints: Dict = constraints if constraints is not None else DEFAULT_CONSTRAINTS.copy()
        self.space_requirements: Dict = space_requirements if space_requirements is not None else DEFAULT_SPACE_REQUIREMENTS.copy()

        # PathPlanner is responsible for route generation between elements
        self.path_planner = PathPlanner(self.plot_width, self.plot_height,
                                      self.elements, self.constraints)
        self.vehicle_routes: List[Dict] = []
        self.pedestrian_routes: List[Dict] = []

        # Weights for different criteria in the evaluation function, set by UI/control widget
        self.optimization_weights: Dict[str, float] = {}

        self.building: Optional[SiteElement] = None # Placeholder for the main building SiteElement
        self.initialize_building() # Create/update the building element and add to self.elements

    def initialize_building(self):
        """
        Initializes or updates the main building element's size and position.
        The building is typically fixed and located in the "northern" part of the site.
        """
        logger.debug("Initializing building...")
        building_total_area = self.plot_width * self.plot_height * self.building_ratio

        # Define building dimensions, e.g., 80% of plot width for building's width
        candidate_width = self.plot_width * 0.80
        candidate_height = building_total_area / candidate_width if candidate_width > 0 else 0

        # Cap height to prevent it from occupying too much of the plot vertically (e.g., max 90%)
        if candidate_height > self.plot_height * 0.90:
            candidate_height = self.plot_height * 0.90
            candidate_width = building_total_area / candidate_height if candidate_height > 0 else 0

        # Ensure width does not exceed plot boundaries (e.g., max 95% to leave side margins)
        if candidate_width > self.plot_width * 0.95:
            candidate_width = self.plot_width * 0.95
            candidate_height = building_total_area / candidate_width if candidate_width > 0 else 0

        # Apply minimum dimensions to ensure the building is always somewhat substantial
        final_width = max(5.0, candidate_width)
        final_height = max(5.0, candidate_height)

        # Position in the "northern" part (top of canvas, assuming top-left origin)
        y_position = 5.0  # Margin from the top plot edge
        x_position = (self.plot_width - final_width) / 2.0 # Centered horizontally

        # Check if building element already exists in the list
        existing_building_idx = -1
        for i, el in enumerate(self.elements):
            if el.name == 'Building':
                existing_building_idx = i
                break

        if existing_building_idx != -1: # Update existing building
            logger.debug("Updating existing Building element.")
            self.building = self.elements[existing_building_idx]
            self.building.x, self.building.y = x_position, y_position
            self.building.width, self.building.height = final_width, final_height
            self.building.rotation = 0 # Buildings usually not rotated
            self.building.movable = False # Building is a fixed entity
        else: # Create new building element
            logger.debug("Creating new Building element.")
            self.building = SiteElement('Building', x_position, y_position,
                                       final_width, final_height, rotation=0)
            self.building.movable = False
            self.elements.append(self.building)
        logger.debug(f"Building initialized: Pos({self.building.x:.1f},{self.building.y:.1f}), Size({self.building.width:.1f}x{self.building.height:.1f})")


    def create_initial_layout(self):
        """
        Generates an initial layout by placing all required site elements.
        The building is placed first, then other elements according to priority and simple heuristics.
        """
        logger.info("Creating initial layout...")
        self.initialize_building() # Ensure building is correctly sized and positioned

        # Retain only the building, remove other elements for a fresh start
        self.elements = [el for el in self.elements if el.name == 'Building']
        if not any(el.name == 'Building' for el in self.elements) and self.building:
            # Ensure building is present if it was somehow removed but self.building exists
            self.elements.insert(0, self.building)
            logger.debug("Building re-added to elements list for initial layout.")


        element_names_to_place = [name for name in SITE_ELEMENTS.keys() if name != 'Building']
        logger.debug(f"Elements to place initially (excluding Building): {element_names_to_place}")

        elements_for_initial_placement = []
        for name in element_names_to_place:
            area = self.space_requirements.get(name, DEFAULT_SPACE_REQUIREMENTS.get(name, 100.0))
            element_config = SITE_ELEMENTS.get(name, {})

            aspect_ratio = element_config.get('aspect_ratio', 1.0)
            w_dim, h_dim = 10.0, 10.0 # Default dimensions

            if element_config.get('resizable', True) is False: # Fixed size element
                w_dim = element_config.get('fixed_width', w_dim)
                h_dim = element_config.get('fixed_height', h_dim)
                # Ensure area matches if fixed dimensions are provided
                if area != (w_dim * h_dim):
                    logger.warning(f"For fixed size element {name}, space requirement {area}m^2 does not match "
                                   f"fixed dimensions {w_dim}x{h_dim}={w_dim*h_dim}m^2. Using fixed dimensions.")
                    area = w_dim * h_dim # This area will be used for evaluation but dimensions are fixed.
                    self.space_requirements[name] = area # Update the official requirement
            elif area > 0:
                # Calculate width and height from area and aspect ratio
                w_dim = math.sqrt(area * aspect_ratio) if area * aspect_ratio > 0 else 1.0
                h_dim = area / w_dim if w_dim > 0 else 1.0
            else: # area <= 0 for a resizable element
                logger.warning(f"Element {name} has non-positive area requirement ({area}). Using default 1x1 dimensions.")
                w_dim, h_dim = 1.0, 1.0


            element_obj = SiteElement(name, 0, 0, max(0.1, w_dim), max(0.1, h_dim)) # Ensure min 0.1
            elements_for_initial_placement.append(element_obj)

        # Sort elements by their defined placement priority
        elements_for_initial_placement.sort(key=lambda e: SITE_ELEMENTS.get(e.name, {}).get('priority', 99))
        logger.debug(f"Sorted elements for placement: {[e.name for e in elements_for_initial_placement]}")

        # Place each element using specific or general placement logic
        for el_to_add in elements_for_initial_placement:
            logger.debug(f"Attempting to place {el_to_add.name}...")
            self._place_element_optimally(el_to_add) # Sets x, y on el_to_add
            self.elements.append(el_to_add) # Add to the main list after position is determined
            logger.debug(f"Placed {el_to_add.name} at ({el_to_add.x:.1f}, {el_to_add.y:.1f})")

        self.update_routes() # Generate paths for this initial layout
        logger.info(f"Initial layout created with {len(self.elements)} elements.")
        return self.elements

    def _place_element_optimally(self, element_to_place: SiteElement,
                                    is_fallback_for_crane: bool = False,
                                    is_fallback_for_security: bool = False): # ADDED PARAMETER
            """
            Determines and sets an initial optimal position for a given SiteElement.
            Uses specialized logic for certain elements (crane, security) or a grid search for others.
            Args:
                element_to_place: The element to place.
                is_fallback_for_crane: If True, this call is a fallback from specialized crane placement,
                                    so skip calling _place_crane_optimally again.
                is_fallback_for_security: If True, this call is a fallback from specialized security post placement.
            """
            best_found_position = None
            highest_score = float('-inf')

            # Use specialized placement routines for critical elements
            if element_to_place.name == 'Tower Crane' and not is_fallback_for_crane:
                self._place_crane_optimally(element_to_place)
                return

            # MODIFIED condition for security post
            if element_to_place.name == 'Security Post' and not is_fallback_for_security:
                self._place_security_post_optimally(element_to_place)
                return

            # General grid search for other elements (and for crane/security when it's a fallback)
            logger.debug(f"Placing {element_to_place.name} using general grid search "
                        f"(Fallback for crane: {is_fallback_for_crane}, Fallback for security: {is_fallback_for_security}).")
            coarse_step = max(2.0, min(element_to_place.width, element_to_place.height) / 2.0)
            fine_step = max(0.5, coarse_step / 4.0)
            search_steps = [coarse_step, fine_step]

            valid_spot_found_in_pass = False
            for step_val in search_steps:
                logger.debug(f"Placing {element_to_place.name}: Grid search with step {step_val}m.")
                for x_val in np.arange(0, self.plot_width - element_to_place.width + 1e-9, step_val):
                    for y_val in np.arange(0, self.plot_height - element_to_place.height + 1e-9, step_val):
                        element_to_place.x = x_val
                        element_to_place.y = y_val

                        if self._check_overlap(element_to_place, strict=True):
                            continue

                        current_score = self._evaluate_element_position_for_initial_placement(element_to_place)
                        if current_score > highest_score:
                            highest_score = current_score
                            best_found_position = (x_val, y_val)

                if best_found_position:
                    valid_spot_found_in_pass = True
                    logger.debug(f"Best position for {element_to_place.name} found in step {step_val}m pass: {best_found_position} (Score: {highest_score})")
                    break

            if not valid_spot_found_in_pass:
                logger.debug(f"No ideal spot for {element_to_place.name} with scoring. Trying any non-overlapping with fine step {fine_step}m.")
                for x_val in np.arange(0, self.plot_width - element_to_place.width + 1e-9, fine_step):
                    for y_val in np.arange(0, self.plot_height - element_to_place.height + 1e-9, fine_step):
                        element_to_place.x = x_val
                        element_to_place.y = y_val
                        if not self._check_overlap(element_to_place, strict=False):
                            best_found_position = (x_val, y_val)
                            valid_spot_found_in_pass = True
                            logger.debug(f"Fallback non-overlapping spot for {element_to_place.name} found: {best_found_position}")
                            break
                    if valid_spot_found_in_pass: break

            if best_found_position:
                element_to_place.x, element_to_place.y = best_found_position
            else:
                element_to_place.x = 1.0
                element_to_place.y = self.plot_height - element_to_place.height - 1.0
                logger.warning(f"Absolute fallback placement for {element_to_place.name} at ({element_to_place.x:.1f}, {element_to_place.y:.1f}) due to limited space or conflicts.")

    def _place_crane_optimally(self, crane: SiteElement):
            """Specialized initial placement for the Tower Crane, focusing on building coverage."""
            logger.debug(f"Placing Tower Crane (specialized): {crane.name} ({crane.width}x{crane.height})")
            building_ref = next((el for el in self.elements if el.name == 'Building'), None)
            if not building_ref:
                logger.warning("No Building found for crane placement. Placing crane centrally.")
                crane.x = self.plot_width / 2.0 - crane.width / 2.0
                crane.y = self.plot_height / 2.0 - crane.height / 2.0
                # Check overlap for this default central position
                if self._check_overlap(crane, strict=True):
                    logger.warning("Central crane position overlaps. Attempting fallback placement.")
                    # MODIFIED CALL:
                    self._place_element_optimally(crane, is_fallback_for_crane=True)
                return

            best_pos = None
            highest_coverage_score = -1.0
            buffer_from_building_edge = self.constraints.get('safety_buffer', 2.0) + 1.0 # Crane needs some clearance

            candidate_positions = []
            # Around building: bottom, left, right, (top if applicable)
            # Bottom edge
            for x_ratio in np.linspace(0.05, 0.95, 7): # Sample 7 points
                x_coord = building_ref.x + building_ref.width * x_ratio - crane.width / 2.0
                y_coord = building_ref.y + building_ref.height + buffer_from_building_edge
                candidate_positions.append((x_coord, y_coord))
            # Left edge
            for y_ratio in np.linspace(0.05, 0.95, 7):
                x_coord = building_ref.x - crane.width - buffer_from_building_edge
                y_coord = building_ref.y + building_ref.height * y_ratio - crane.height / 2.0
                candidate_positions.append((x_coord, y_coord))
            # Right edge
            for y_ratio in np.linspace(0.05, 0.95, 7):
                x_coord = building_ref.x + building_ref.width + buffer_from_building_edge
                y_coord = building_ref.y + building_ref.height * y_ratio - crane.height / 2.0
                candidate_positions.append((x_coord, y_coord))
            # Top edge (if building is not at the very top of the plot)
            if building_ref.y > crane.height + buffer_from_building_edge + 1.0 :
                for x_ratio in np.linspace(0.05, 0.95, 7):
                    x_coord = building_ref.x + building_ref.width * x_ratio - crane.width / 2.0
                    y_coord = building_ref.y - crane.height - buffer_from_building_edge
                    candidate_positions.append((x_coord, y_coord))

            logger.debug(f"Generated {len(candidate_positions)} candidate positions for crane.")

            original_elements_list = list(self.elements) # Make a copy
            self.elements = [el for el in self.elements if el.name != crane.name]

            for x_try, y_try in candidate_positions:
                crane.x, crane.y = x_try, y_try

                if self._check_overlap(crane, strict=True):
                    continue

                current_score = self._evaluate_crane_coverage_for_initial_placement(crane, building_ref)
                if current_score > highest_coverage_score:
                    highest_coverage_score = current_score
                    best_pos = (x_try, y_try)

            self.elements = original_elements_list

            if best_pos:
                crane.x, crane.y = best_pos
                logger.info(f"Optimal initial crane position: ({crane.x:.1f}, {crane.y:.1f}) with coverage score {highest_coverage_score:.1f}")
            else:
                logger.warning("No valid position found for Tower Crane with good coverage. Using general placement AS FALLBACK.")
                # MODIFIED CALL:
                self._place_element_optimally(crane, is_fallback_for_crane=True)


    def _place_security_post_optimally(self, security_post: SiteElement):
            """Specialized initial placement for the Security Post, near the site entrance."""
            logger.debug(f"Placing Security Post (specialized): {security_post.name} ({security_post.width}x{security_post.height})") # Clarify
            entrance_x_world = self.plot_width / 2.0
            y_pos_world = self.plot_height - security_post.height - 1.0

            best_pos = None
            highest_score = float('-inf')

            buffer_from_path_center = 3.0
            candidate_x_coords = [
                entrance_x_world - security_post.width - buffer_from_path_center,
                entrance_x_world + buffer_from_path_center,
                1.0,
                self.plot_width - security_post.width - 1.0
            ]
            logger.debug(f"Candidate X for Security Post: {candidate_x_coords}, Y: {y_pos_world}")

            original_elements_list = list(self.elements)
            self.elements = [el for el in self.elements if el.name != security_post.name]

            for x_try in candidate_x_coords:
                security_post.x, security_post.y = x_try, y_pos_world

                if self._check_overlap(security_post, strict=True):
                    continue

                dist_to_left_ideal = abs(x_try - (entrance_x_world - security_post.width - buffer_from_path_center))
                dist_to_right_ideal = abs(x_try - (entrance_x_world + buffer_from_path_center))
                current_score = -min(dist_to_left_ideal, dist_to_right_ideal)

                if current_score > highest_score:
                    highest_score = current_score
                    best_pos = (x_try, y_pos_world)

            self.elements = original_elements_list

            if best_pos:
                security_post.x, security_post.y = best_pos
                logger.info(f"Optimal Security Post position: ({security_post.x:.1f}, {security_post.y:.1f})")
            else:
                logger.warning("No valid position found for Security Post. Using general placement AS FALLBACK.")
                # MODIFIED CALL:
                self._place_element_optimally(security_post, is_fallback_for_security=True)

    def _check_overlap(self, element_to_check: SiteElement, strict: bool = True) -> bool:
        """
        Checks if the given element overlaps with any existing elements in the layout
        or extends beyond the plot boundaries. Includes a hard check against Building.
        """
        buffer_val = self.constraints.get('safety_buffer', 1.0) if strict else 0.0

        # 1. Check against plot boundaries
        # Use element's AABB for a quick check, then precise corners if needed or if rotated.
        # For unrotated:
        if element_to_check.rotation == 0:
            if not (element_to_check.x >= -1e-3 and \
                    element_to_check.y >= -1e-3 and \
                    element_to_check.x + element_to_check.width <= self.plot_width + 1e-3 and \
                    element_to_check.y + element_to_check.height <= self.plot_height + 1e-3):
                # Check precise corners if AABB seems out but might be due to floating point
                corners_of_element = element_to_check.get_corners()
                for cx, cy in corners_of_element:
                    if not (0.0 <= cx <= self.plot_width and 0.0 <= cy <= self.plot_height):
                        # logger.debug(f"Overlap: {element_to_check.name} out of bounds at corner ({cx:.1f},{cy:.1f}). Plot: {self.plot_width}x{self.plot_height}")
                        return True # An entire corner is out of bounds
        else: # Rotated element, must check all corners
            corners_of_element = element_to_check.get_corners()
            for cx, cy in corners_of_element:
                if not (-1e-3 <= cx <= self.plot_width + 1e-3 and -1e-3 <= cy <= self.plot_height + 1e-3):
                    # logger.debug(f"Overlap: Rotated {element_to_check.name} out of bounds at corner ({cx:.1f},{cy:.1f}).")
                    return True


        # 2. Check for overlaps with other placed elements
        building_element_ref = next((el for el in self.elements if el.name == 'Building'), None)

        for existing_element_in_layout in self.elements:
            if existing_element_in_layout == element_to_check:
                continue

            # If the element being checked can be placed inside a building and the
            # existing_element_in_layout is NOT the building, skip overlap check
            # (as its placement logic is different). This rule is complex.
            # For now, assume internal placement is handled by a different mechanism
            # and general overlap checks apply to all external items.

            effective_buffer = buffer_val
            # Special handling for Building: movable elements should not overlap Building AT ALL.
            if building_element_ref and \
               element_to_check.movable and \
               existing_element_in_layout == building_element_ref and \
               not element_to_check.placeable_inside_building: # Movable, external item vs Building
                if element_to_check.overlaps(building_element_ref, 0.0): # Strict no-buffer overlap
                    # logger.debug(f"Overlap: Movable {element_to_check.name} overlaps Building.")
                    return True
            # For placeable_inside_building elements, they should not overlap building EXTERNALLY.
            # Their internal placement is a separate concern.
            elif building_element_ref and \
                 element_to_check.placeable_inside_building and \
                 existing_element_in_layout == building_element_ref:
                 if element_to_check.overlaps(building_element_ref, 0.0): # Strict no-buffer overlap
                    # logger.debug(f"Overlap: Placeable_inside {element_to_check.name} overlaps Building externally.")
                    return True
            # Standard overlap check for all other element pairs
            elif element_to_check.overlaps(existing_element_in_layout, effective_buffer):
                # logger.debug(f"Overlap: {element_to_check.name} overlaps {existing_element_in_layout.name} with buffer {effective_buffer:.1f}.")
                return True

        return False # No overlaps detected


    def _evaluate_element_position_for_initial_placement(self, element: SiteElement) -> float:
        """
        Provides a simplified scoring for an element's position during initial layout generation.
        Helps guide elements to generally sensible starting locations. Higher score is better.
        """
        score = 0.0
        building = next((el for el in self.elements if el.name == 'Building'), None)

        # General preference: avoid being too close to plot edges (unless element is e.g. Security Post)
        if element.name not in ['Security Post', 'Fuel Tank']: # These might prefer edges/corners
            min_dist_to_any_edge = min(
                element.x, element.y,
                self.plot_width - (element.x + element.width),
                self.plot_height - (element.y + element.height)
            )
            preferred_edge_margin = 3.0 # Prefer to be at least this far from edge
            if min_dist_to_any_edge < preferred_edge_margin:
                score -= (preferred_edge_margin - min_dist_to_any_edge) * 10.0

        # Proximity to building (can be positive or negative depending on element)
        if building and element.name not in ['Fuel Tank', 'Security Post', 'Workers Dormitory', 'Welding Workshop', 'Machinery Parking']:
            dist_to_building = element.distance_to(building)
            # Offices, Material Storage, Sanitary Facilities prefer closer to building
            if element.name in ['Material Storage', 'Sanitary Facilities'] or 'Office' in element.name:
                # Score higher for closer, up to a point. Very close might be bad.
                # Target distance e.g. 5-15m. Penalty if too far or too close.
                target_dist_building = 10.0
                if dist_to_building < 3.0: # Too close
                    score -= (3.0 - dist_to_building) * 5.0
                elif dist_to_building < target_dist_building:
                    score += (target_dist_building - dist_to_building) * 1.5
                else: # Further than target
                    score -= (dist_to_building - target_dist_building) * 0.5


        # Specific relationships for key elements (if their counterparts are already placed)
        if element.name == 'Welding Workshop':
            storage = next((el for el in self.elements if el.name == 'Material Storage'), None)
            if storage: # If storage is already placed
                dist_ws = element.distance_to(storage)
                max_efficient_dist = self.constraints.get('max_distance_storage_welding', 20.0)
                if dist_ws <= max_efficient_dist: score += (max_efficient_dist - dist_ws) * 2.0
                else: score -= (dist_ws - max_efficient_dist) * 3.0

        if element.name == 'Fuel Tank': # Prefer to be far from sensitive areas and often at edges
            # Bonus for being near horizontal/vertical edges/corners
            corner_proximity_bonus = 50.0
            edge_dist_x = min(element.x, self.plot_width - (element.x + element.width))
            edge_dist_y = min(element.y, self.plot_height - (element.y + element.height))
            if edge_dist_x < 2.0 and edge_dist_y < 2.0 : score += corner_proximity_bonus # Corner
            elif edge_dist_x < 2.0 or edge_dist_y < 2.0: score += corner_proximity_bonus / 2.0 # Edge

        if 'Office' in element.name or element.name == 'Workers Dormitory':
            for hazard_name_check in ['Welding Workshop', 'Machinery Parking', 'Fuel Tank']:
                hazard_ref = next((el for el in self.elements if el.name == hazard_name_check), None)
                if hazard_ref:
                    dist_to_hazard_el = element.distance_to(hazard_ref)
                    min_safe_dist_initial = self.constraints.get(f'min_distance_{element.name.lower().replace(" ","")}_{hazard_name_check.lower().replace(" ","")}', 15.0)
                    # Fallback for specific keys like min_distance_offices_welding etc.
                    if "Office" in element.name and hazard_name_check == "Welding Workshop":
                        min_safe_dist_initial = self.constraints.get('min_distance_offices_welding', 18.0)
                    elif "Office" in element.name and hazard_name_check == "Machinery Parking":
                        min_safe_dist_initial = self.constraints.get('min_distance_offices_machinery', 12.0)
                    elif element.name == "Workers Dormitory" and hazard_name_check == "Fuel Tank":
                         min_safe_dist_initial = self.constraints.get('min_distance_fuel_dormitory', 25.0)


                    if dist_to_hazard_el < min_safe_dist_initial:
                        score -= (min_safe_dist_initial - dist_to_hazard_el) * 5.0
                    else:
                        score += (dist_to_hazard_el - min_safe_dist_initial) * 0.5

        # Accessibility from "entrance" (assumed bottom edge, center) for some elements
        if element.name in ['Material Storage', 'Machinery Parking']: # Security handled separately
            # Higher score if element's y is larger (closer to bottom) and x is central
            center_y_factor = (element.y + element.height / 2.0) / self.plot_height # 0 (top) to 1 (bottom)
            center_x_factor = 1.0 - abs((element.x + element.width / 2.0) - self.plot_width / 2.0) / (self.plot_width / 2.0) # 0 (edge) to 1 (center)
            score += (center_y_factor * 0.7 + center_x_factor * 0.3) * 30.0

        return score

    def _evaluate_crane_coverage_for_initial_placement(self, crane: SiteElement, building_to_cover: SiteElement) -> float:
        """Scores crane position based on its coverage of the main building. Used for initial placement."""
        if not building_to_cover: return 0.0

        crane_cx, crane_cy = crane.get_center()
        crane_reach_dist = self.constraints.get('crane_reach', 40.0)

        building_corners_coords = building_to_cover.get_corners()
        corners_covered_count = 0
        for b_corner_x, b_corner_y in building_corners_coords:
            if math.hypot(crane_cx - b_corner_x, crane_cy - b_corner_y) <= crane_reach_dist:
                corners_covered_count += 1

        building_center_x, building_center_y = building_to_cover.get_center()
        is_building_center_covered = math.hypot(crane_cx - building_center_x, crane_cy - building_center_y) <= crane_reach_dist

        # Scoring based on coverage
        coverage_score = corners_covered_count * 25.0
        if is_building_center_covered: coverage_score += 35.0
        if corners_covered_count == 4 and is_building_center_covered: # Bonus for full coverage
            coverage_score += 70.0 # Strong bonus
        elif corners_covered_count >= 2 and is_building_center_covered: # Good partial coverage
            coverage_score += 20.0
        return coverage_score

    def optimize_layout(self, iterations=1000, initial_temperature=100.0, cooling_rate=0.95,
                        min_temperature=0.1, reheat_factor=1.5, reheat_threshold_temp=5.0,
                        progress_callback=None):
        """
        Optimizes the current layout using a simulated annealing algorithm.

        Args:
            iterations (int): Number of iterations for the algorithm.
            initial_temperature (float): Initial temperature for simulated annealing.
            cooling_rate (float): Rate at which temperature decreases.
            min_temperature (float): The minimum temperature the system can reach.
            reheat_factor (float): Factor by which to multiply temperature if reheating.
            reheat_threshold_temp (float): If temperature drops below this and stuck, consider reheating.
            progress_callback (callable, optional): Function to call for progress updates.

        Returns:
            list: The list of SiteElement objects representing the optimized layout.
        """
        logger.info(f"Starting SA Optimization: Iter={iterations}, T0={initial_temperature:.1f}, Cool={cooling_rate:.3f}, "
                    f"MinT={min_temperature:.3f}, ReheatF={reheat_factor:.1f}, ReheatThr={reheat_threshold_temp:.1f}")

        if not self.elements: # No elements to optimize
            logger.warning("Optimization called with no elements.")
            if progress_callback: progress_callback(100)
            return []

        if not self.optimization_weights:
            logger.warning("optimize_layout called without optimization_weights. Using a default set.")
            self.optimization_weights = { # Fallback, should be set by UI
                'overlap': 200.0, 'out_of_bounds': 150.0,
                'movable_on_building_penalty': 2500.0,
                'vehicle_path_length_penalty': 0.1, 'pedestrian_path_length_penalty': 0.05,
                'route_interference_penalty': 2.0
            }

        # Current state is self.elements
        current_score = self.evaluate_layout() # Evaluates self.elements
        logger.debug(f"Initial score for SA: {current_score:.2f}")

        best_score_so_far = current_score
        best_layout_config = self.clone_elements() # Deep copy of current self.elements

        temp = float(initial_temperature)
        no_improvement_streak = 0
        # Adjust MAX_STREAK_FOR_REHEAT based on total iterations
        MAX_STREAK_FOR_REHEAT = max(50, iterations // 20) # e.g., if no improvement for 5% of iterations, or min 50

        for i in range(iterations):
            if progress_callback and i % max(1, iterations // 50) == 0: # Update progress more frequently
                progress_callback(int((i / iterations) * 100))

            # Snapshot current state of self.elements before attempting a change
            elements_before_change_snapshot = self.clone_elements()
            score_before_change = current_score # Store score corresponding to snapshot

            # make_random_change modifies self.elements directly
            original_state_of_THE_changed_element = self.make_random_change()

            if original_state_of_THE_changed_element: # If a valid change was made to self.elements
                new_score = self.evaluate_layout() # Evaluate the new self.elements state
                score_delta = new_score - current_score # current_score is from previous accepted state
                accepted_change = False

                if score_delta > 0: # New layout is better, always accept
                    accepted_change = True
                    no_improvement_streak = 0
                elif temp > min_temperature: # If new layout is worse, accept with a probability
                    acceptance_probability = math.exp(score_delta / temp)
                    if random.random() < acceptance_probability:
                        accepted_change = True
                        no_improvement_streak += 1
                    else: # Rejected
                        no_improvement_streak += 1
                else: # Temperature is very low, effectively only accept improvements (or if score_delta == 0)
                    if score_delta >= 0: # Accept if equal or better at min_temp
                        accepted_change = True
                    no_improvement_streak += 1

                if accepted_change:
                    current_score = new_score # Update current_score to the new accepted score
                    if new_score > best_score_so_far:
                        best_score_so_far = new_score
                        best_layout_config = self.clone_elements() # Clone current self.elements
                        no_improvement_streak = 0 # Reset streak on new best
                        logger.debug(f"Iter {i}, New Best Score: {best_score_so_far:.2f}, Temp: {temp:.2f}")
                else: # Reject and revert self.elements to state before the change
                    self.elements = elements_before_change_snapshot
                    current_score = score_before_change # Restore score as well
            else: # No valid change could be made by make_random_change
                no_improvement_streak += 1

            temp *= cooling_rate
            if temp < min_temperature:
                temp = min_temperature

            # Reheating logic
            if temp <= reheat_threshold_temp and temp > min_temperature * 1.01 and \
               no_improvement_streak > MAX_STREAK_FOR_REHEAT and \
               i < iterations * 0.90: # Avoid reheating very late in the process
                new_temp_candidate = temp * reheat_factor
                temp = min(initial_temperature * 0.5, new_temp_candidate) # Reheat, but cap, e.g. to 50% of T0
                no_improvement_streak = 0 # Reset streak after reheat
                logger.info(f"SA Reheating: Iteration {i}, No improvement streak: {no_improvement_streak}. Old Temp: {temp/reheat_factor:.2f}, New Temp: {temp:.2f}")

        self.elements = best_layout_config # Set the engine's elements to the best configuration found
        self.update_routes() # Final update of routes for the best layout

        final_score = self.evaluate_layout() # Evaluate the final best layout
        logger.info(f"SA Optimization Finished. Best score: {final_score:.2f} (was {best_score_so_far:.2f} internally). Temp ended at: {temp:.3f}")

        if progress_callback: progress_callback(100) # Signal completion
        return self.elements


    def _revert_element_change(self, original_element_state: SiteElement):
        """Reverts a single element in self.elements to its provided original state."""
        if not original_element_state: return

        for idx, current_el in enumerate(self.elements):
            if current_el.name == original_element_state.name: # Assuming name is unique for movable
                # To be fully robust, especially if multiple elements of same name can exist
                # and be movable, an ID system or direct object comparison would be better.
                # For now, this relies on make_random_change returning a snapshot of the *specific*
                # element it modified.
                self.elements[idx] = original_element_state # Replace with the original state copy
                logger.debug(f"Reverted change for element {original_element_state.name}")
                return
        logger.warning(f"Could not find element {original_element_state.name} to revert in self.elements.")


    def make_random_change(self) -> Optional[SiteElement]:
        """
        Applies a random modification (position, size, or rotation) to one of the movable elements
        IN PLACE within self.elements.
        If the change results in an invalid state (e.g., overlap, out of bounds), it's reverted.

        Returns:
            SiteElement: A deep copy (snapshot) of the element *before* a successful, valid change was applied.
                         Returns None if no movable elements exist or if the attempted change was invalid and reverted.
        """
        movable_elements_refs = [el for el in self.elements if el.movable and not el.is_placed_inside_building]
        if not movable_elements_refs:
            # logger.debug("No movable elements to change.")
            return None

        element_to_change = random.choice(movable_elements_refs)

        # Create a deep copy (snapshot) of the element's state BEFORE any modification
        original_element_snapshot = self.clone_elements(source_elements_list=[element_to_change])[0]


        # Determine the type of change
        element_config = SITE_ELEMENTS.get(element_to_change.name, {})
        can_resize = element_config.get('resizable', True)
        can_rotate = element_config.get('resizable', True) # Often tied to resizable, or could be separate flag

        change_type_weights = {'position': 0.6}
        if can_resize: change_type_weights['size'] = 0.25
        if can_rotate and element_to_change.name != 'Building': # Buildings usually don't rotate
             change_type_weights['rotation'] = 0.15

        # Normalize weights if some options are disabled
        total_weight = sum(change_type_weights.values())
        if total_weight == 0: return None # No possible changes
        normalized_weights = [w / total_weight for w in change_type_weights.values()]

        type_of_change = random.choices(
            list(change_type_weights.keys()),
            weights=normalized_weights
        )[0]

        is_change_valid_and_applied = False
        # Store original geometric properties of the element being changed
        orig_x, orig_y = element_to_change.x, element_to_change.y
        orig_w, orig_h = element_to_change.width, element_to_change.height
        orig_rot = element_to_change.rotation

        if type_of_change == 'position':
            max_move_factor = 0.10 # Smaller, more controlled moves
            dx = random.uniform(-self.plot_width * max_move_factor, self.plot_width * max_move_factor)
            dy = random.uniform(-self.plot_height * max_move_factor, self.plot_height * max_move_factor)
            element_to_change.x += dx
            element_to_change.y += dy
            # Clamping handled by _check_overlap's boundary check or explicitly after

        elif type_of_change == 'size': # Assumes can_resize was true
            original_center_x, original_center_y = orig_x + orig_w/2, orig_y + orig_h/2

            area_change_factor = random.uniform(0.85, 1.15) # +/- 15% area change
            current_area = orig_w * orig_h
            new_target_area = max(1.0, current_area * area_change_factor) # Min area 1 m^2

            aspect_ratio = element_config.get('aspect_ratio', orig_w / orig_h if orig_h > 0 else 1.0)
            # Slightly perturb aspect ratio if desired
            aspect_ratio_perturb_factor = random.uniform(0.9, 1.1)
            effective_aspect_ratio = aspect_ratio * aspect_ratio_perturb_factor

            prospective_width = math.sqrt(new_target_area * effective_aspect_ratio)
            prospective_height = new_target_area / prospective_width if prospective_width > 1e-3 else 1.0

            min_dim_allowed = SITE_ELEMENTS.get(element_to_change.name, {}).get('min_dim', 1.0) # Default min 1m
            element_to_change.width = max(min_dim_allowed, prospective_width)
            element_to_change.height = max(min_dim_allowed, prospective_height)

            # Re-center
            element_to_change.x = original_center_x - element_to_change.width / 2.0
            element_to_change.y = original_center_y - element_to_change.height / 2.0

        elif type_of_change == 'rotation': # Assumes can_rotate was true
            original_center_x, original_center_y = orig_x + orig_w/2, orig_y + orig_h/2
            
            # Allow small random rotation or snapping to 90 degrees
            if random.random() < 0.7: # 70% chance of 90-degree step
                allowed_rotations_degrees = [0, 90, 180, 270]
                possible_new_rots = [r for r in allowed_rotations_degrees if abs(r - orig_rot) > 1e-3]
                if not possible_new_rots: possible_new_rots = allowed_rotations_degrees
                element_to_change.rotation = random.choice(possible_new_rots)
            else: # Small random angle change
                angle_change = random.uniform(-20, 20) # degrees
                element_to_change.rotation = (orig_rot + angle_change + 360.0) % 360.0

            # Dimension swap if orientation flips (e.g. 0 to 90) and not square
            old_orientation_major_axis = (int(orig_rot % 180.0) == 0) # True if landscape-ish
            new_orientation_major_axis = (int(element_to_change.rotation % 180.0) == 0)

            if old_orientation_major_axis != new_orientation_major_axis and abs(orig_w - orig_h) > 1e-3:
                element_to_change.width, element_to_change.height = orig_h, orig_w # Swap dimensions

            # Re-center after potential dimension swap
            element_to_change.x = original_center_x - element_to_change.width / 2.0
            element_to_change.y = original_center_y - element_to_change.height / 2.0

        # Validate the change (bounds and overlap)
        if not self._check_overlap(element_to_change, strict=True): # strict=True also checks bounds
            is_change_valid_and_applied = True
        else: # Revert the change on element_to_change
            element_to_change.x, element_to_change.y = orig_x, orig_y
            element_to_change.width, element_to_change.height = orig_w, orig_h
            element_to_change.rotation = orig_rot
            # logger.debug(f"Random change for {element_to_change.name} (type: {type_of_change}) resulted in overlap/OOB. Reverted.")

        if is_change_valid_and_applied:
            # logger.debug(f"Successful random change for {element_to_change.name} (type: {type_of_change}).")
            return original_element_snapshot # Return the state *before* this successful change
        return None # No valid change was made, or an attempted change was reverted


    def evaluate_layout(self) -> float:
        """
        Calculates a comprehensive score for the current layout (self.elements),
        considering overlaps, boundary adherence, constraints, path lengths, and interference.
        Lower scores are generally worse (due to penalties). Higher scores are better.
        """
        # CRITICAL: Ensure paths are updated based on the current element positions before evaluation
        # This call to update_routes will use self.elements
        self.update_routes()

        eval_weights = self.optimization_weights
        if not eval_weights:
            logger.critical("evaluate_layout: Optimization weights not set! This is a bug or setup error. Using emergency defaults.")
            eval_weights = { # A minimal set of emergency default weights
                'overlap': 500.0, 'out_of_bounds': 300.0, # Higher penalties for critical issues
                'movable_on_building_penalty': 5000.0,
                # ... (include all keys used below with some default values) ...
                'storage_welding_dist_penalty': 3.0, 'storage_welding_dist_bonus': 2.0,
                'fuel_welding_dist_penalty': 12.0, 'fuel_welding_safety_bonus': 1.5,
                'fuel_dorm_dist_penalty': 18.0, 'fuel_dorm_safety_bonus': 2.0,
                'crane_coverage_penalty': 30.0, 'crane_full_coverage_bonus': 60.0,
                'comfort_safety_penalty': 4.0, 'comfort_safety_bonus': 0.8,
                'spacing_bonus': 1.2, 'spacing_penalty': 2.5,
                'office_proximity_penalty': 0.6,
                'security_entrance_bonus': 1.2, 'security_entrance_penalty': 0.6,
                'accessibility_bonus_factor': 1.0,
                'vehicle_path_length_penalty': 0.25,
                'pedestrian_path_length_penalty': 0.15,
                'route_interference_penalty': 6.0
            }

        layout_score = 1000.0  # Start with a base score; penalties will reduce it.

        # --- Hard Constraints: Overlaps and Boundary Violations ---
        # These are checked first and can lead to large penalties.
        num_overlaps = 0
        num_out_of_bounds = 0

        for i, elem1 in enumerate(self.elements):
            # Check if any corner of elem1 is outside plot boundaries
            elem1_corners = elem1.get_corners()
            is_oob = False
            for corner_x_coord, corner_y_coord in elem1_corners:
                if not (0.0 <= corner_x_coord <= self.plot_width and \
                        0.0 <= corner_y_coord <= self.plot_height):
                    is_oob = True
                    break
            if is_oob:
                num_out_of_bounds += 1
                # Calculate how far out of bounds for scaled penalty (optional)
                # For now, flat penalty per OOB element.
                layout_score -= eval_weights.get('out_of_bounds', 150.0)


            # Check for overlaps with other elements
            for j in range(i + 1, len(self.elements)):
                elem2 = self.elements[j]
                # Skip overlap check for elements placed inside the building against each other
                # or against the building itself, as their "overlap" is managed differently.
                if getattr(elem1, 'is_placed_inside_building', False) and \
                   getattr(elem2, 'is_placed_inside_building', False):
                    continue
                if getattr(elem1, 'is_placed_inside_building', False) and elem2.name == 'Building':
                    continue
                if getattr(elem2, 'is_placed_inside_building', False) and elem1.name == 'Building':
                    continue


                # CRITICAL: Heavy penalty if any movable element overlaps the main Building footprint
                # This applies to elements NOT intended to be inside the building.
                is_elem1_building = elem1.name == 'Building'
                is_elem2_building = elem2.name == 'Building'

                if (is_elem1_building and elem2.movable and not getattr(elem2, 'placeable_inside_building', False)) or \
                   (is_elem2_building and elem1.movable and not getattr(elem1, 'placeable_inside_building', False)):
                    if elem1.overlaps(elem2, 0.0): # Strict, no-buffer check
                        layout_score -= eval_weights.get('movable_on_building_penalty', 2500.0)
                        num_overlaps +=1
                # Standard overlap check for other pairs
                elif elem1.overlaps(elem2, 0.0): # Using 0.0 buffer for direct overlap, safety_buffer handled separately
                    layout_score -= eval_weights.get('overlap', 200.0)
                    num_overlaps +=1

        # --- Retrieve specific elements for constraint checks ---
        # (Ensure these handle None if element is not present in self.elements)
        elements_by_name = {el.name: el for el in self.elements}
        storage = elements_by_name.get('Material Storage')
        welding = elements_by_name.get('Welding Workshop')
        fuel_tank = elements_by_name.get('Fuel Tank')
        dormitory = elements_by_name.get('Workers Dormitory')
        tower_crane = elements_by_name.get('Tower Crane')
        building_reference = elements_by_name.get('Building') # Should always exist from initialize_building
        offices_list = [el for el in self.elements if 'Office' in el.name]
        security_post = elements_by_name.get('Security Post')

        # --- Constraint-based Scoring (Soft Constraints) ---
        # 1. Material Storage to Welding Workshop Distance
        if storage and welding:
            dist_val = storage.distance_to(welding)
            max_dist_constraint = self.constraints.get('max_distance_storage_welding', 25.0)
            if dist_val <= max_dist_constraint:
                layout_score += (max_dist_constraint - dist_val) * eval_weights.get('storage_welding_dist_bonus', 2.0)
            else:
                layout_score -= (dist_val - max_dist_constraint) * eval_weights.get('storage_welding_dist_penalty', 3.0)

        # 2. Fuel Tank Safety Distances
        if fuel_tank:
            if welding:
                dist_val = fuel_tank.distance_to(welding)
                min_dist_constraint = self.constraints.get('min_distance_fuel_welding', 15.0)
                if dist_val < min_dist_constraint:
                    layout_score -= (min_dist_constraint - dist_val) * eval_weights.get('fuel_welding_dist_penalty', 12.0)
                else:
                    layout_score += (dist_val - min_dist_constraint) * eval_weights.get('fuel_welding_safety_bonus', 1.5)
            if dormitory:
                dist_val = fuel_tank.distance_to(dormitory)
                min_dist_constraint = self.constraints.get('min_distance_fuel_dormitory', 20.0)
                if dist_val < min_dist_constraint:
                    layout_score -= (min_dist_constraint - dist_val) * eval_weights.get('fuel_dorm_dist_penalty', 18.0)
                else:
                    layout_score += (dist_val - min_dist_constraint) * eval_weights.get('fuel_dorm_safety_bonus', 2.0)

        # 3. Tower Crane Coverage of the Main Building
        if tower_crane and building_reference:
            crane_cx, crane_cy = tower_crane.get_center()
            crane_r = self.constraints.get('crane_reach', 40.0)
            bldg_corners = building_reference.get_corners()
            corners_covered = sum(1 for bcx, bcy in bldg_corners if math.hypot(crane_cx - bcx, crane_cy - bcy) <= crane_r)
            bldg_cx, bldg_cy = building_reference.get_center()
            center_is_covered = math.hypot(crane_cx - bldg_cx, crane_cy - bldg_cy) <= crane_r

            if corners_covered < 4 or not center_is_covered:
                missed_points_total = (4 - corners_covered) + (0 if center_is_covered else 1)
                layout_score -= missed_points_total * eval_weights.get('crane_coverage_penalty', 30.0)
            if corners_covered == 4 and center_is_covered:
                layout_score += eval_weights.get('crane_full_coverage_bonus', 60.0)

        # 4. Comfort and Safety for Offices and Dormitories (distance from hazards)
        comfort_elements = offices_list + ([dormitory] if dormitory else [])
        hazard_types = {
            'Welding Workshop': ('min_distance_offices_welding', 'min_distance_dormitory_welding'), # Example, define specific keys
            'Machinery Parking': ('min_distance_offices_machinery', 'min_distance_dormitory_machinery'),
            'Fuel Tank': ('min_distance_offices_fuel', 'min_distance_fuel_dormitory') # Note: fuel_dorm is already used
        }
        for comf_el in comfort_elements:
            for hazard_name, (office_key, dorm_key) in hazard_types.items():
                hazard_el = elements_by_name.get(hazard_name)
                if hazard_el:
                    constraint_key = ""
                    if "Office" in comf_el.name: constraint_key = office_key
                    elif comf_el.name == "Workers Dormitory": constraint_key = dorm_key
                    
                    # Fallback if specific key for dorm/hazard isn't defined, use office key
                    if not constraint_key or constraint_key not in self.constraints:
                         constraint_key = office_key # Default to office key if dorm specific not found

                    min_dist_val = self.constraints.get(constraint_key, 15.0)

                    actual_dist = comf_el.distance_to(hazard_el)
                    if actual_dist < min_dist_val:
                        layout_score -= (min_dist_val - actual_dist) * eval_weights.get('comfort_safety_penalty', 4.0)
                    else:
                        layout_score += (actual_dist - min_dist_val) * eval_weights.get('comfort_safety_bonus', 0.8)

        # 5. Spacing Between All Elements (Safety Buffer Adherence)
        safety_buffer_dist = self.constraints.get('safety_buffer', 1.0) # Use the constraint
        building_safety_buffer_dist = self.constraints.get('building_safety_buffer', safety_buffer_dist) # Specific for building

        for i, el_one in enumerate(self.elements):
            # Skip spacing checks for elements placed inside the building against other internal elements or the building itself
            if getattr(el_one, 'is_placed_inside_building', False): continue

            for el_two in self.elements[i+1:]:
                if getattr(el_two, 'is_placed_inside_building', False): continue

                dist_between_edges = el_one.distance_to(el_two) # Returns 0 if overlapping (already penalized)
                
                current_buffer_target = safety_buffer_dist
                if el_one.name == 'Building' or el_two.name == 'Building':
                    if self.constraints.get('buffer_building_override', False): # Check if override is active
                        current_buffer_target = building_safety_buffer_dist
                
                if dist_between_edges > 0: # If they are not overlapping
                    if dist_between_edges >= current_buffer_target:
                         layout_score += (dist_between_edges - current_buffer_target) * eval_weights.get('spacing_bonus', 0.5) # Smaller bonus
                    else: # Closer than buffer, but not overlapping - penalty
                         layout_score -= (current_buffer_target - dist_between_edges) * eval_weights.get('spacing_penalty', 2.5)

        # 6. Proximity Among Offices (Offices should ideally be grouped)
        if len(offices_list) > 1:
            total_office_distance = 0
            num_office_pairs_counted = 0
            for i, office1 in enumerate(offices_list):
                for office2 in offices_list[i+1:]:
                    total_office_distance += office1.distance_to(office2)
                    num_office_pairs_counted += 1
            if num_office_pairs_counted > 0:
                average_office_distance = total_office_distance / num_office_pairs_counted
                target_avg_office_dist = 15.0 # meters
                if average_office_distance > target_avg_office_dist:
                     layout_score -= (average_office_distance - target_avg_office_dist) * eval_weights.get('office_proximity_penalty', 0.6)
                else: # Bonus if offices are close
                     layout_score += (target_avg_office_dist - average_office_distance) * eval_weights.get('office_proximity_bonus', 0.3)


        # 7. Security Post Proximity to Site Entrance (assumed bottom-center)
        if security_post:
            entrance_x_coord, entrance_y_coord = self.path_planner.site_entry_world_coord['x'], self.path_planner.site_entry_world_coord['y']
            sec_post_cx, sec_post_cy = security_post.get_center()
            dist_to_entrance_pt = math.hypot(sec_post_cx - entrance_x_coord, sec_post_cy - entrance_y_coord)

            ideal_prox_to_entrance = security_post.width / 2.0 + 3.0 # e.g. radius + 3m
            max_good_dist_from_entrance = 15.0
            if dist_to_entrance_pt < max_good_dist_from_entrance:
                layout_score += (max_good_dist_from_entrance - dist_to_entrance_pt) * eval_weights.get('security_entrance_bonus', 1.2)
            else:
                layout_score -= (dist_to_entrance_pt - max_good_dist_from_entrance) * eval_weights.get('security_entrance_penalty', 0.6)

        # 8. Accessibility for Material Storage and Machinery Parking
        for el_name_for_access in ['Material Storage', 'Machinery Parking']:
            access_element = elements_by_name.get(el_name_for_access)
            if access_element:
                # Bonus for being closer to the bottom edge (larger y-value)
                # And closer to site entry X
                entry_x = self.path_planner.site_entry_world_coord['x']
                el_center_x, el_center_y = access_element.get_center()

                y_proximity_score = (el_center_y / self.plot_height) # Normalized 0 to 1 (bottom is better)
                x_proximity_score = 1.0 - (abs(el_center_x - entry_x) / (self.plot_width / 2.0)) # Normalized 0 (edge) to 1 (entry_x)
                
                # Weighted average of y and x proximity scores
                accessibility_score = (y_proximity_score * 0.6 + x_proximity_score * 0.4) * 20.0 # Max 20 points
                layout_score += accessibility_score * eval_weights.get('accessibility_bonus_factor', 1.0)


        # --- Path-related Scoring (Path Lengths and Interference) ---
        if self.path_planner: # Path planner should have routes calculated by self.update_routes()
            vehicle_path_total_len = self.path_planner.get_total_path_length('vehicle')
            pedestrian_path_total_len = self.path_planner.get_total_path_length('pedestrian')
            routes_interference_val = self.path_planner.calculate_route_interference()

            layout_score -= vehicle_path_total_len * eval_weights.get('vehicle_path_length_penalty', 0.25)
            layout_score -= pedestrian_path_total_len * eval_weights.get('pedestrian_path_length_penalty', 0.15)
            layout_score -= routes_interference_val * eval_weights.get('route_interference_penalty', 6.0)

        return layout_score

    def update_routes(self):
        """
        Recalculates all vehicle and pedestrian routes using the PathPlanner.
        This should be called whenever element positions change.
        """
        if not self.elements:
            self.vehicle_routes, self.pedestrian_routes = [], []
            logger.debug("update_routes: No elements, routes cleared.")
            return [], []

        # Ensure PathPlanner instance is up-to-date with current site state
        if self.path_planner:
            self.path_planner.update_elements(self.elements) # Use dedicated update method
            self.path_planner.constraints = self.constraints.copy() # Pass a copy
            self.path_planner.plot_width = self.plot_width
            self.path_planner.plot_height = self.plot_height
        else:
             logger.error("PathPlanner not initialized in OptimizationEngine! Cannot update routes.")
             self.path_planner = PathPlanner(self.plot_width, self.plot_height,
                                      self.elements, self.constraints) # Failsafe

        # Generate new routes, force rebuild of obstruction grid as elements may have moved
        self.vehicle_routes, self.pedestrian_routes = self.path_planner.create_routes(force_rebuild_grid=True)
        # logger.debug(f"Routes updated. Vehicle: {len(self.vehicle_routes)}, Pedestrian: {len(self.pedestrian_routes)}")
        return self.vehicle_routes, self.pedestrian_routes

    def clone_elements(self, source_elements_list: Optional[List[SiteElement]] = None) -> List[SiteElement]:
        """
        Creates and returns a deep copy of a list of SiteElement objects.
        If no list is provided, clones the engine's current `self.elements`.
        """
        elements_to_clone = source_elements_list if source_elements_list is not None else self.elements
        if elements_to_clone is None: # Should not happen if self.elements is always a list
            return []

        cloned_list = []
        for original_element in elements_to_clone:
            # Create a new SiteElement instance with copied attributes
            cloned_element = SiteElement(
                original_element.name, original_element.x, original_element.y,
                original_element.width, original_element.height, original_element.rotation
            )
            # Deep copy mutable attributes like QColor
            cloned_element.color = QColor(original_element.color)
            cloned_element.movable = original_element.movable
            cloned_element.selected = original_element.selected # Copy selection state too for full snapshot
            cloned_element.priority = original_element.priority
            cloned_element.placeable_inside_building = original_element.placeable_inside_building

            # Copy internal placement attributes if they exist
            cloned_element.is_placed_inside_building = getattr(original_element, 'is_placed_inside_building', False)
            # For internal_placement_info, if it's a dict, a shallow copy is often enough,
            # but deepcopy if it contains mutable structures.
            original_internal_info = getattr(original_element, 'internal_placement_info', None)
            cloned_element.internal_placement_info = original_internal_info.copy() if isinstance(original_internal_info, dict) else original_internal_info

            cloned_element.original_external_x = getattr(original_element, 'original_external_x', None)
            cloned_element.original_external_y = getattr(original_element, 'original_external_y', None)
            cloned_element.original_external_width = getattr(original_element, 'original_external_width', None)
            cloned_element.original_external_height = getattr(original_element, 'original_external_height', None)
            cloned_element.original_external_rotation = getattr(original_element, 'original_external_rotation', None)


            # If SiteElement has other mutable attributes (e.g., list of connected_elements),
            # ensure they are also deep copied:
            # cloned_element.connected_elements = list(original_element.connected_elements)

            cloned_list.append(cloned_element)

        return cloned_list

# --- Optimization Worker Thread ---
class OptimizationWorker(QThread):
    """
    Worker thread for running the optimization process in the background,
    preventing the GUI from freezing during potentially long computations.
    It communicates progress and completion back to the main thread via signals.
    """
    progress_updated = pyqtSignal(int)  # Emits current progress percentage (0-100)
    optimization_complete = pyqtSignal(list)  # Emits the list of optimized SiteElement objects
    optimization_failed = pyqtSignal(str) # Emits an error message string if optimization fails critically

    def __init__(self,
                 optimizer_instance: 'OptimizationEngine', # Type hint for clarity
                 iterations: int,
                 initial_temperature_sa: float,
                 cooling_rate_sa: float,
                 min_temperature_sa: float,
                 reheat_factor_sa: float,
                 reheat_threshold_temp_sa: float):
        """
        Initializes the OptimizationWorker.

        Args:
            optimizer_instance: The instance of the OptimizationEngine to use.
                                This instance will be modified by the optimization process.
            iterations: Number of iterations for the simulated annealing algorithm.
            initial_temperature_sa: Starting temperature for SA.
            cooling_rate_sa: Cooling rate for SA (e.g., 0.99).
            min_temperature_sa: Minimum temperature for SA.
            reheat_factor_sa: Factor by which to multiply temperature upon reheating.
            reheat_threshold_temp_sa: Temperature threshold below which reheating might occur if stuck.
        """
        super().__init__()
        self.optimizer = optimizer_instance
        self.iterations = iterations
        self.initial_temperature_sa = initial_temperature_sa
        self.cooling_rate_sa = cooling_rate_sa
        self.min_temperature_sa = min_temperature_sa
        self.reheat_factor_sa = reheat_factor_sa
        self.reheat_threshold_temp_sa = reheat_threshold_temp_sa

        logger.info(f"OptimizationWorker initialized with {self.iterations} iterations.")

    def run(self):
        """
        The main execution method for the QThread. This method is called when
        `thread.start()` is invoked. It runs the optimization algorithm.
        """
        logger.info("OptimizationWorker: run() method started.")
        try:
            # The optimizer's optimize_layout method will modify its own self.elements list.
            # It's crucial that this method is designed to be long-running and that
            # the optimizer_instance is not accessed directly by the main GUI thread
            # while this worker is running (except through signals/slots or thread-safe mechanisms).
            logger.debug(f"Calling optimizer.optimize_layout with T0={self.initial_temperature_sa}, "
                         f"CoolRate={self.cooling_rate_sa}, MinT={self.min_temperature_sa}, "
                         f"ReheatF={self.reheat_factor_sa}, ReheatT={self.reheat_threshold_temp_sa}")

            # The optimize_layout method in OptimizationEngine now directly modifies
            # self.optimizer.elements to the best found configuration.
            self.optimizer.optimize_layout(
                iterations=self.iterations,
                initial_temperature=self.initial_temperature_sa,
                cooling_rate=self.cooling_rate_sa,
                min_temperature=self.min_temperature_sa,
                reheat_factor=self.reheat_factor_sa,
                reheat_threshold_temp=self.reheat_threshold_temp_sa,
                progress_callback=self.progress_updated.emit # Pass the signal emitter as callback
            )

            # After optimization, self.optimizer.elements contains the best layout.
            # Emit a deep copy of this list to the main thread.
            if hasattr(self.optimizer, 'clone_elements'):
                final_elements_copy = self.optimizer.clone_elements()
                logger.info(f"Optimization complete. Emitting {len(final_elements_copy)} elements.")
                self.optimization_complete.emit(final_elements_copy)
            else:
                # Fallback if clone_elements is missing (should not happen with correct engine)
                error_msg = "OptimizationEngine is missing 'clone_elements' method. Cannot emit results."
                logger.error(error_msg)
                self.optimization_failed.emit(error_msg)
                self.progress_updated.emit(100) # Still mark as complete for UI unlock

        except Exception as e:
            error_message = f"Critical error during optimization in OptimizationWorker: {str(e)}"
            logger.error(error_message, exc_info=True) # Log with full traceback
            self.optimization_failed.emit(error_message)
            # Ensure progress is marked as complete (or errored) to avoid UI hang.
            # The main thread might use this to know the process ended, even if with an error.
            self.progress_updated.emit(100) # Or a special value like -1 for error state.
        finally:
            logger.info("OptimizationWorker: run() method finished.")
            # The QThread.finished signal will be emitted automatically after this method returns.
# --- SiteCanvas (Visualization) ---
class SiteCanvas(QWidget):
    """
    Widget for visualizing and interacting with the construction site layout.
    Handles zooming, panning, element selection, and drawing of site elements,
    routes, grid, crane radius, and other visual aids.
    Distinguishes display for internally placed elements.
    """
    element_selected = pyqtSignal(object) # Emits the selected SiteElement or None
    element_moved = pyqtSignal(object)    # Emits the moved SiteElement (after drag or key move)
    view_changed = pyqtSignal()           # Emitted when zoom/pan changes view significantly (e.g., for scale bar update)
    context_menu_requested_at_world = pyqtSignal(QPointF, object) # Emits world_pos, hovered_element (or None)

    MIN_SCALE_FACTOR_PX_PER_M = 0.02 # Minimum zoom out (e.g., 2% of nominal, effectively 0.2 px/m if nominal is 10)
    MAX_SCALE_FACTOR_PX_PER_M = 200.0 # Maximum zoom in (e.g., 2000% of nominal, effectively 2000 px/m)
    NOMINAL_PX_PER_METER = 10.0 # A reference scale for calculating zoom percentage and default view

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(600, 450) # Ensure a decent default size
        self.setAutoFillBackground(True)
        self.setFocusPolicy(Qt.StrongFocus) # For keyboard events (arrows, zoom, etc.)
        self.setMouseTracking(True) # For hover effects, dynamic cursors, and mouse coordinate display

        # Plot (World) Dimensions - these will be set by the main application
        self.plot_width_m: float = 80.0  # Default, will be updated
        self.plot_height_m: float = 50.0 # Default, will be updated

        # View Transformation State (Zoom/Pan)
        self.scale_factor_px_per_m: float = self.NOMINAL_PX_PER_METER # Current pixels per meter
        self.view_offset_x_px: float = 20.0  # Screen X coordinate for world origin (0,0)
        self.view_offset_y_px: float = 20.0  # Screen Y coordinate for world origin (0,0)

        # Cached rectangles for efficient painting and culling
        self.plot_rect_on_screen: QRectF = QRectF() # The plot area as drawn on screen
        self.widget_viewport_rect_f: QRectF = QRectF(self.rect()) # The current widget's QRect as QRectF

        # Data for Display - managed by the main application
        self.elements_list: List[SiteElement] = []
        self.vehicle_routes_list: List[Dict] = []
        self.pedestrian_routes_list: List[Dict] = []
        self.current_constraints: Dict = {} # For things like crane reach display
        self.building_element: Optional[SiteElement] = None # Cache main building for quick access

        # Interaction State
        self.selected_element_obj: Optional[SiteElement] = None
        self.is_panning: bool = False
        self.is_dragging_element: bool = False
        self.last_mouse_pos_screen: QPointF = QPointF(0, 0) # Last known mouse position in screen coordinates
        self.drag_element_start_pos_world: QPointF = QPointF(0, 0) # Element's top-left in world at drag start
        self.drag_mouse_start_pos_screen: QPointF = QPointF(0,0) # Mouse screen pos at drag start

        # Display Options - controlled by DisplayOptionsWidget via main app
        self.show_grid_lines: bool = True
        self.show_element_dimensions: bool = True
        self.show_element_labels: bool = True
        self.show_crane_operational_radius: bool = True
        self.show_scale_bar_info: bool = True
        self.show_internal_element_markers: bool = True
        self.show_out_of_bounds_elements: bool = False # Option to draw elements even if partially OOB (for debugging)
        self.zoom_sensitivity_wheel: float = 0.12  # Controls how much mouse wheel zooms
        self.zoom_sensitivity_keys: float = 1.15 # Factor for +/- key zoom

        self._update_cached_rects() # Initialize screen rects based on current state
        logger.debug("SiteCanvas initialized.")

    def set_plot_dimensions(self, width_m: float, height_m: float):
        """Sets the world dimensions of the plot being visualized."""
        old_width, old_height = self.plot_width_m, self.plot_height_m
        self.plot_width_m = max(1.0, float(width_m)) # Ensure positive dimensions
        self.plot_height_m = max(1.0, float(height_m))
        logger.debug(f"SiteCanvas: Plot dimensions updated to {self.plot_width_m:.1f}m x {self.plot_height_m:.1f}m.")

        # If dimensions changed significantly, fit plot to view. Otherwise, maintain current view.
        if abs(old_width - self.plot_width_m) > 1e-3 or abs(old_height - self.plot_height_m) > 1e-3:
            self.fit_plot_to_view()
        else:
            self._update_cached_rects() # Just update cached rects if dimensions are same (e.g. during init)
            self.update() # Trigger repaint

    def set_elements(self, elements: List[SiteElement]):
        """Sets the list of SiteElement objects to be displayed."""
        self.elements_list = list(elements) # Store a copy to avoid external modification issues
        self.building_element = next((el for el in self.elements_list if el.name == 'Building'), None)
        logger.debug(f"SiteCanvas: Elements list updated ({len(self.elements_list)} total).")
        self.update() # Trigger repaint

    def set_routes(self, vehicle_routes: List[Dict], pedestrian_routes: List[Dict]):
        """Sets the lists of vehicle and pedestrian routes to be displayed."""
        self.vehicle_routes_list = list(vehicle_routes)
        self.pedestrian_routes_list = list(pedestrian_routes)
        logger.debug(f"SiteCanvas: Routes updated (Vehicle: {len(vehicle_routes)}, Pedestrian: {len(pedestrian_routes)}).")
        self.update() # Trigger repaint

    def set_constraints(self, constraints_dict: Dict):
        """Sets the current project constraints (used for things like crane radius display)."""
        self.current_constraints = constraints_dict.copy()
        logger.debug(f"SiteCanvas: Constraints updated.")
        self.update() # Trigger repaint if constraints affect visuals

    def fit_plot_to_view(self, margin_px: float = 25.0):
        """Adjusts scale and offset to fit the entire plot within the widget view with a margin."""
        if self.plot_width_m <= 1e-3 or self.plot_height_m <= 1e-3:
            logger.warning("SiteCanvas: Cannot fit plot with zero or near-zero dimensions.")
            return

        drawable_width_px = self.width() - 2 * margin_px
        drawable_height_px = self.height() - 2 * margin_px

        if drawable_width_px <= 10 or drawable_height_px <= 10: # Widget too small for meaningful fit
            logger.warning("SiteCanvas: Widget area too small to fit plot meaningfully. Using minimum scale.")
            self.scale_factor_px_per_m = self.MIN_SCALE_FACTOR_PX_PER_M
            # Attempt to center the origin of the (tiny) plot
            self.view_offset_x_px = self.width() / 2.0
            self.view_offset_y_px = self.height() / 2.0
        else:
            scale_x = drawable_width_px / self.plot_width_m
            scale_y = drawable_height_px / self.plot_height_m
            # Choose the smaller scale to ensure entire plot fits, respecting min/max limits
            self.scale_factor_px_per_m = max(self.MIN_SCALE_FACTOR_PX_PER_M, min(scale_x, scale_y))
            self.scale_factor_px_per_m = min(self.MAX_SCALE_FACTOR_PX_PER_M, self.scale_factor_px_per_m)

            # Recalculate plot dimensions on screen with the new scale
            plot_screen_width_scaled = self.plot_width_m * self.scale_factor_px_per_m
            plot_screen_height_scaled = self.plot_height_m * self.scale_factor_px_per_m

            # Center the scaled plot within the drawable area
            self.view_offset_x_px = margin_px + (drawable_width_px - plot_screen_width_scaled) / 2.0
            self.view_offset_y_px = margin_px + (drawable_height_px - plot_screen_height_scaled) / 2.0

        self._update_cached_rects() # Update screen representation of plot
        self.view_changed.emit()    # Notify that view parameters changed
        self.update()               # Trigger repaint
        logger.debug(f"SiteCanvas: Fit plot to view. New scale: {self.scale_factor_px_per_m:.2f} px/m. "
                     f"Offset: ({self.view_offset_x_px:.1f}, {self.view_offset_y_px:.1f})px.")


    def _update_cached_rects(self):
        """Calculates and caches QRectFs for the plot area as drawn on screen and the widget viewport."""
        plot_origin_s = self.world_to_screen_coords(0, 0) # Top-left of plot in screen coords
        plot_screen_width_scaled = self.plot_width_m * self.scale_factor_px_per_m
        plot_screen_height_scaled = self.plot_height_m * self.scale_factor_px_per_m
        self.plot_rect_on_screen = QRectF(plot_origin_s, QSizeF(plot_screen_width_scaled, plot_screen_height_scaled))
        self.widget_viewport_rect_f = QRectF(self.rect()) # Current QWidget.rect() as QRectF


    # --- Coordinate Transformations ---
    def world_to_screen_coords(self, world_x_m: float, world_y_m: float) -> QPointF:
        """Converts world coordinates (meters) to screen coordinates (pixels)."""
        screen_x = self.view_offset_x_px + world_x_m * self.scale_factor_px_per_m
        screen_y = self.view_offset_y_px + world_y_m * self.scale_factor_px_per_m
        return QPointF(screen_x, screen_y)

    def screen_to_world_coords(self, screen_x_px: float, screen_y_px: float) -> QPointF:
        """Converts screen coordinates (pixels) to world coordinates (meters)."""
        if abs(self.scale_factor_px_per_m) < 1e-9: # Avoid division by zero or very small scale
            logger.warning("Screen to world conversion attempted with near-zero scale factor. Returning plot center.")
            return QPointF(self.plot_width_m / 2.0, self.plot_height_m / 2.0)
        world_x = (screen_x_px - self.view_offset_x_px) / self.scale_factor_px_per_m
        world_y = (screen_y_px - self.view_offset_y_px) / self.scale_factor_px_per_m
        return QPointF(world_x, world_y)

    # --- Painting ---
    def paintEvent(self, event: QPaintEvent):
        """Handles all custom painting for the canvas."""
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)

        self._update_cached_rects() # Ensure cached screen rectangles are current for this paint cycle

        # Fill background
        painter.fillRect(self.rect(), self.palette().color(QPalette.Window)) # Use theme background

        # Draw plot boundary if it's visible in the widget viewport
        if self.plot_rect_on_screen.intersects(self.widget_viewport_rect_f):
            painter.setPen(QPen(QColor(90, 90, 90), 2.0, Qt.SolidLine)) # Slightly lighter border for dark bg
            painter.drawRect(self.plot_rect_on_screen)

        # Draw grid if enabled and visible
        if self.show_grid_lines:
            self._draw_grid(painter)

        # Save painter state and set clipping region for drawing elements, routes, etc.
        # This prevents drawing outside the visible intersection of plot and widget.
        painter.save()
        effective_clip_rect = self.plot_rect_on_screen.intersected(self.widget_viewport_rect_f)
        # Adjust clip rect slightly inward to avoid drawing on the very edge of the plot boundary line
        painter.setClipRect(effective_clip_rect.adjusted(1, 1, -1, -1), Qt.IntersectClip)


        if self.show_crane_operational_radius:
            self._draw_crane_radius(painter)

        # Draw routes (vehicle then pedestrian, so pedestrian can be on top if desired)
        self._draw_routes(painter, self.vehicle_routes_list, QColor(60, 60, 180, 190), 0.30, Qt.SolidLine) # Vehicle: blueish
        self._draw_routes(painter, self.pedestrian_routes_list, QColor(200, 120, 40, 210), 0.15, Qt.DashLine) # Pedestrian: orangish

        # Separate elements into external and internal for potentially different drawing logic or order
        external_elements = sorted(
            [el for el in self.elements_list if not getattr(el, 'is_placed_inside_building', False)],
            key=lambda el: SITE_ELEMENTS.get(el.name, {}).get('priority', 99) # Draw lower priority (often larger/fixed) first
        )
        internal_elements = [el for el in self.elements_list if getattr(el, 'is_placed_inside_building', False)]

        # Draw external elements
        for element in external_elements:
            self._draw_element_external(painter, element)

        # Draw markers for internal elements on the building
        if self.show_internal_element_markers and self.building_element and internal_elements:
            # Optionally, redraw building slightly differently if it has internal items (e.g. highlight)
            # self._draw_element_external(painter, self.building_element, highlight_internal_host=True)
            for i, internal_el in enumerate(internal_elements):
                self._draw_element_internal_marker(painter, internal_el, self.building_element, i, len(internal_elements))

        painter.restore() # Restore painter state (clipping, transforms)

        # Draw overlays like scale bar and view info last, on top of everything
        if self.show_scale_bar_info:
            self._draw_scale_bar(painter)
            self._draw_view_info(painter) # Includes mouse coordinates

    def _draw_grid(self, painter: QPainter):
        """Draws major and minor grid lines on the canvas."""
        # Determine appropriate grid spacing based on current scale
        # Aim for major lines every ~50-100px, minor every ~10-25px on screen
        target_major_px = 75.0
        target_minor_px = 20.0

        # Calculate world spacing from target pixel spacing
        major_spacing_m_ideal = target_major_px / self.scale_factor_px_per_m
        minor_spacing_m_ideal = target_minor_px / self.scale_factor_px_per_m

        # Snap to "nice" numbers (1, 2, 5, 10, 20, 50, 100...)
        nice_numbers = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        major_spacing_m = min(nice_numbers, key=lambda x: abs(x - major_spacing_m_ideal) + (0 if x >= major_spacing_m_ideal*0.5 else float('inf')))
        minor_spacing_m = min(nice_numbers, key=lambda x: abs(x - minor_spacing_m_ideal) + (0 if x >= minor_spacing_m_ideal*0.5 else float('inf')))

        # Conditions for showing minor grid lines
        show_minor = True
        if minor_spacing_m * self.scale_factor_px_per_m < 8: # Too dense in pixels
            show_minor = False
        if major_spacing_m <= minor_spacing_m : # Minor spacing is not smaller than major
             minor_spacing_m = major_spacing_m / 5.0 # Try to make it a subdivision
             if minor_spacing_m * self.scale_factor_px_per_m < 8: show_minor = False


        # Drawing area: intersection of plot_rect_on_screen and widget_viewport_rect_f
        draw_area = self.plot_rect_on_screen.intersected(self.widget_viewport_rect_f)
        if draw_area.isEmpty(): return # Nothing to draw if no intersection

        world_tl_visible = self.screen_to_world_coords(draw_area.left(), draw_area.top())
        world_br_visible = self.screen_to_world_coords(draw_area.right(), draw_area.bottom())

        # Major grid lines
        pen_major = QPen(QColor(80, 80, 80, 120), 1.0, Qt.SolidLine) # Darker for dark theme
        painter.setPen(pen_major)

        painter.setPen(pen_major)
        start_x_m_major = math.floor(world_tl_visible.x() / major_spacing_m) * major_spacing_m
        for wx_m in np.arange(start_x_m_major, world_br_visible.x() + 1e-6, major_spacing_m):
            if wx_m < -1e-3 or wx_m > self.plot_width_m + 1e-3: continue # Only draw within plot world bounds (with tolerance)
            p1_s = self.world_to_screen_coords(wx_m, world_tl_visible.y())
            p2_s = self.world_to_screen_coords(wx_m, world_br_visible.y())
            painter.drawLine(QLineF(p1_s, p2_s))

        start_y_m_major = math.floor(world_tl_visible.y() / major_spacing_m) * major_spacing_m
        for wy_m in np.arange(start_y_m_major, world_br_visible.y() + 1e-6, major_spacing_m):
            if wy_m < -1e-3 or wy_m > self.plot_height_m + 1e-3: continue
            p1_s = self.world_to_screen_coords(world_tl_visible.x(), wy_m)
            p2_s = self.world_to_screen_coords(world_br_visible.x(), wy_m)
            painter.drawLine(QLineF(p1_s, p2_s))

        # Minor grid lines (if enabled and conditions met)
        if show_minor and minor_spacing_m > 0.01 : # Ensure minor_spacing is sensible
            pen_minor = QPen(QColor(65, 65, 65, 90), 0.75, Qt.DotLine)
            painter.setPen(pen_minor)
            start_x_m_minor = math.floor(world_tl_visible.x() / minor_spacing_m) * minor_spacing_m
            for wx_m in np.arange(start_x_m_minor, world_br_visible.x() + 1e-6, minor_spacing_m):
                if wx_m < -1e-3 or wx_m > self.plot_width_m + 1e-3: continue
                if abs(wx_m % major_spacing_m) < 1e-6 : continue # Don't redraw major lines
                p1_s = self.world_to_screen_coords(wx_m, world_tl_visible.y())
                p2_s = self.world_to_screen_coords(wx_m, world_br_visible.y())
                painter.drawLine(QLineF(p1_s, p2_s))

            start_y_m_minor = math.floor(world_tl_visible.y() / minor_spacing_m) * minor_spacing_m
            for wy_m in np.arange(start_y_m_minor, world_br_visible.y() + 1e-6, minor_spacing_m):
                if wy_m < -1e-3 or wy_m > self.plot_height_m + 1e-3: continue
                if abs(wy_m % major_spacing_m) < 1e-6: continue
                p1_s = self.world_to_screen_coords(world_tl_visible.x(), wy_m)
                p2_s = self.world_to_screen_coords(world_br_visible.x(), wy_m)
                painter.drawLine(QLineF(p1_s, p2_s))

    def _draw_routes(self, painter: QPainter, routes_list: List[Dict], color: QColor,
                     base_line_width_m: float, line_style: Qt.PenStyle):
        """Draws a list of routes (polylines) on the canvas."""
        if not routes_list: return

        # Calculate line width in pixels, ensuring it's reasonable
        line_width_px = max(0.8, min(8.0, base_line_width_m * self.scale_factor_px_per_m))
        if line_width_px < 0.5: return # Too thin to be visible

        pen = QPen(color, line_width_px)
        pen.setStyle(line_style)
        pen.setCapStyle(Qt.RoundCap)   # Smoother line ends
        pen.setJoinStyle(Qt.RoundJoin) # Smoother line joins
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush) # Paths are lines, not filled shapes

        for route_info in routes_list:
            path_wc = route_info.get('path', []) # List of {'x': ..., 'y': ...} dicts
            if len(path_wc) < 2: continue # Need at least two points for a line

            # Convert world coordinate path to screen coordinate polygon
            screen_polygon = QPolygonF()
            for point_wc in path_wc:
                if 'x' in point_wc and 'y' in point_wc: # Basic validation
                    screen_polygon.append(self.world_to_screen_coords(point_wc['x'], point_wc['y']))

            if not screen_polygon.isEmpty() and \
               screen_polygon.boundingRect().intersects(self.widget_viewport_rect_f): # Culling
                painter.drawPolyline(screen_polygon)


    def _draw_crane_radius(self, painter: QPainter):
        """Draws the operational radius circle for Tower Crane elements."""
        # Find active (not internal) tower cranes
        active_cranes = [el for el in self.elements_list if el.name == 'Tower Crane' and
                         not getattr(el, 'is_placed_inside_building', False)]
        if not active_cranes: return

        reach_m = self.current_constraints.get('crane_reach', 40.0)
        if reach_m <= 0: return

        radius_s = reach_m * self.scale_factor_px_per_m
        if radius_s < 2.0: return # Too small to draw meaningfully

        pen_color = QColor(180, 0, 0, 100) # Semi-transparent dark red for pen
        brush_color = QColor(220, 0, 0, 25) # Very transparent red for fill
        line_width_px = max(1.0, min(2.5, 0.10 * self.scale_factor_px_per_m)) # Dynamic line width

        painter.setPen(QPen(pen_color, line_width_px, Qt.DashDotLine))
        painter.setBrush(QBrush(brush_color))

        for crane in active_cranes:
            cx_w, cy_w = crane.get_center()
            center_s = self.world_to_screen_coords(cx_w, cy_w)

            # Define the circle's bounding rectangle in screen coordinates
            circle_rect_s = QRectF(0, 0, 2 * radius_s, 2 * radius_s)
            circle_rect_s.moveCenter(center_s)

            # Culling: only draw if the circle intersects the visible widget area
            if circle_rect_s.intersects(self.widget_viewport_rect_f):
                painter.drawEllipse(center_s, radius_s, radius_s)


    def _draw_element_external(self, painter: QPainter, element: SiteElement, highlight_internal_host: bool = False):
        """Draws a single externally placed SiteElement."""
        el_origin_s = self.world_to_screen_coords(element.x, element.y)
        el_w_s = element.width * self.scale_factor_px_per_m
        el_h_s = element.height * self.scale_factor_px_per_m

        # Culling: if element is too small on screen or completely outside viewport
        if (el_w_s < 0.5 and el_h_s < 0.5): # Threshold for being too small
            return

        # For rotated elements, get actual screen corners for more accurate AABB culling
        if element.rotation != 0:
            screen_corners = [self.world_to_screen_coords(wc[0], wc[1]) for wc in element.get_corners()]
            if not screen_corners: return # Should not happen
            min_sx = min(p.x() for p in screen_corners)
            max_sx = max(p.x() for p in screen_corners)
            min_sy = min(p.y() for p in screen_corners)
            max_sy = max(p.y() for p in screen_corners)
            aabb_screen = QRectF(QPointF(min_sx, min_sy), QPointF(max_sx, max_sy))
        else: # Unrotated, AABB is simpler
            aabb_screen = QRectF(el_origin_s, QSizeF(el_w_s, el_h_s))

        if not aabb_screen.intersects(self.widget_viewport_rect_f) and not self.show_out_of_bounds_elements:
            return

        painter.save() # Save current painter state (transform, pen, brush)

        # Apply rotation if any
        if element.rotation != 0:
            # Rotation pivot is the center of the element in screen coordinates
            center_s_x = el_origin_s.x() + el_w_s / 2.0
            center_s_y = el_origin_s.y() + el_h_s / 2.0
            painter.translate(center_s_x, center_s_y)
            painter.rotate(element.rotation)
            painter.translate(-center_s_x, -center_s_y) # Translate back

        # Define the rectangle to be drawn (now in element's local rotated frame if applicable)
        element_rect_s = QRectF(el_origin_s, QSizeF(el_w_s, el_h_s))

        # Setup brush (fill)
        color = element.color
        # Simple gradient for a bit of depth
        grad = QLinearGradient(element_rect_s.topLeft(), element_rect_s.bottomRight())
        grad.setColorAt(0, color.lighter(125))
        grad.setColorAt(0.5, color)
        grad.setColorAt(1, color.darker(125))
        painter.setBrush(QBrush(grad))

        # Setup pen (border)
        border_width_px = max(0.75, min(3.5, 0.15 * self.scale_factor_px_per_m)) # Dynamic border width
        border_color = color.darker(150)
        pen_style = Qt.SolidLine

        if element.selected:
            border_width_px = max(1.5, min(4.5, 0.25 * self.scale_factor_px_per_m))
            border_color = QColor(0, 100, 220, 220) # Bright blue for selection
            pen_style = Qt.DashLine # Different style for selection border
        elif highlight_internal_host: # For the Building element if it contains internal items
            border_width_px = max(1.2, min(4.0, 0.20 * self.scale_factor_px_per_m))
            border_color = QColor(255, 150, 0, 180) # Orangeish highlight
            pen_style = Qt.SolidLine

        painter.setPen(QPen(border_color, border_width_px, pen_style))
        painter.drawRect(element_rect_s) # Draw the rectangle

        # Draw text (label and dimensions) if conditions met
        min_dim_for_text_s = 15.0 # Minimum screen dimension (px) to attempt drawing text inside
        if min(el_w_s, el_h_s) > min_dim_for_text_s and self.scale_factor_px_per_m > 0.5: # Zoomed in enough
            # Determine text color based on background brightness
            brightness = (color.redF() * 0.299 + color.greenF() * 0.587 + color.blueF() * 0.114)
            text_color = Qt.black if brightness > 0.6 else Qt.white # Adjusted brightness threshold
            
            # Calculate dynamic font size
            font_pixel_size = max(7.0, min(16.0, min(el_w_s, el_h_s) / 5.0, # Relative to element size
                                          12.0 * self.scale_factor_px_per_m / self.NOMINAL_PX_PER_METER)) # Relative to zoom
            font = painter.font()
            font.setPixelSize(int(font_pixel_size))
            painter.setFont(font)
            painter.setPen(text_color)

            text_margin_px = border_width_px + 2.0 # Margin from element border
            text_rect_s = element_rect_s.adjusted(text_margin_px, text_margin_px, -text_margin_px, -text_margin_px)
            
            lines_to_draw = []
            if self.show_element_labels and element.name:
                lines_to_draw.append(element.name)
            if self.show_element_dimensions and font_pixel_size > 7.5: # Only show dims if font is not too tiny
                lines_to_draw.append(f"{element.width:.1f}m × {element.height:.1f}m")
            
            full_text_content = "\n".join(lines_to_draw)
            if full_text_content and text_rect_s.width() > 5 and text_rect_s.height() > 5:
                 # Use QFontMetricsF for accurate text bounding box
                 fm = QFontMetricsF(font)
                 # Check if text will fit; if not, might draw only name or use elided text
                 required_text_rect = fm.boundingRect(text_rect_s, Qt.AlignCenter | Qt.TextWordWrap, full_text_content)
                 if required_text_rect.height() > text_rect_s.height() and len(lines_to_draw) > 1:
                     # If multi-line text too tall, try drawing only the first line (name)
                     first_line_content = lines_to_draw[0]
                     required_first_line_rect = fm.boundingRect(text_rect_s, Qt.AlignCenter | Qt.TextWordWrap, first_line_content)
                     if required_first_line_rect.height() <= text_rect_s.height():
                         painter.drawText(text_rect_s, Qt.AlignCenter | Qt.TextWordWrap, first_line_content)
                 else: # Either single line or multi-line fits
                     painter.drawText(text_rect_s, Qt.AlignCenter | Qt.TextWordWrap, full_text_content)
        painter.restore() # Restore painter state


    def _draw_element_internal_marker(self, painter: QPainter, element: SiteElement,
                                      building: SiteElement, index: int, total_internal_elements: int):
        """Draws a marker for an element placed inside the 'building' SiteElement."""
        if not building: return # Should not happen if called correctly

        # Position markers in a grid-like fashion on the building surface for visualization.
        # This is a simplified representation.
        # Approximate number of columns for markers based on building aspect ratio and total items.
        building_aspect_ratio = building.width / max(1.0, building.height)
        cols = max(1, int(math.sqrt(total_internal_elements * building_aspect_ratio)))
        rows = max(1, math.ceil(total_internal_elements / cols))

        # Calculate cell dimensions for markers within the building
        cell_width_w = building.width / cols if cols > 0 else building.width
        cell_height_w = building.height / rows if rows > 0 else building.height

        # Determine grid position for this marker
        col_idx = index % cols
        row_idx = index // cols

        # Calculate marker center in world coordinates (relative to building's top-left)
        marker_center_x_w = building.x + (col_idx + 0.5) * cell_width_w
        marker_center_y_w = building.y + (row_idx + 0.5) * cell_height_w
        marker_center_s = self.world_to_screen_coords(marker_center_x_w, marker_center_y_w)

        # Determine marker size dynamically based on cell size on screen, with caps
        marker_radius_s = min(cell_width_w * self.scale_factor_px_per_m * 0.35, # 35% of cell width
                              cell_height_w * self.scale_factor_px_per_m * 0.35, # 35% of cell height
                              10.0 * self.scale_factor_px_per_m / self.NOMINAL_PX_PER_METER, # Max size relative to nominal zoom
                              15.0) # Absolute max pixel radius
        marker_radius_s = max(3.0, marker_radius_s) # Absolute min pixel radius for visibility

        # Culling: Check if marker's bounding box is visible
        marker_bounding_rect_s = QRectF(0, 0, marker_radius_s * 2, marker_radius_s * 2)
        marker_bounding_rect_s.moveCenter(marker_center_s)
        if not marker_bounding_rect_s.intersects(self.widget_viewport_rect_f):
            return # Marker is off-screen

        painter.save()
        marker_color = element.color
        painter.setBrush(QBrush(marker_color.lighter(115)))
        pen_border_width_px = max(0.5, marker_radius_s / 10.0)
        painter.setPen(QPen(marker_color.darker(150), pen_border_width_px))
        painter.drawEllipse(marker_center_s, marker_radius_s, marker_radius_s) # Draw circular marker

        # Draw initial letter if marker is large enough and labels are enabled
        if marker_radius_s > 6.0 and self.show_element_labels and element.name:
            font_pixel_size = int(marker_radius_s * 0.9) # Font size relative to marker radius
            font = painter.font(); font.setPixelSize(font_pixel_size); font.setBold(True)
            painter.setFont(font)
            # Text color based on marker background brightness
            brightness = (marker_color.redF()*0.299 + marker_color.greenF()*0.587 + marker_color.blueF()*0.114)
            text_color = Qt.black if brightness > 0.55 else Qt.white
            painter.setPen(text_color)

            initial_letter = element.name[0].upper()
            # Use QFontMetricsF for centering text within the ellipse marker
            fm = QFontMetricsF(font)
            text_width = fm.horizontalAdvance(initial_letter)
            text_height = fm.height() # Approximate height
            # Adjust text position slightly for better centering in circle
            text_x_s = marker_center_s.x() - text_width / 2.0
            text_y_s = marker_center_s.y() - fm.descent() + text_height / 2.0 - (text_height * 0.1) # Heuristic adjustment
            painter.drawText(QPointF(text_x_s, text_y_s), initial_letter)
        painter.restore()


    def _draw_scale_bar(self, painter: QPainter):
        """Draws a dynamic scale bar at the bottom-left of the canvas."""
        margin_px = 20
        bar_y_pos_px = self.height() - 25 # Position from bottom
        bar_height_px = 8 # Thickness of scale bar ticks

        # Target an on-screen width for the scale bar (e.g., 1/8th of canvas width)
        target_bar_width_px = max(50.0, self.width() / 8.0)
        if target_bar_width_px < 30 : return # Too small to be useful

        # Calculate corresponding length in world units (meters)
        target_length_m = target_bar_width_px / self.scale_factor_px_per_m

        # Find a "nice" round number for the scale bar length in meters
        # (e.g., 1, 2, 5, 10, 20, 50, 100... meters)
        nice_lengths_m = [l * (10**e) for e in range(-2, 4) for l in [1, 2, 2.5, 5]] # 0.01m to 5000m
        # Choose the nice length that is closest to (and preferably not much larger than) target_length_m
        best_length_m = min(nice_lengths_m, key=lambda l_m: abs(l_m - target_length_m) + (0 if l_m <= target_length_m * 1.8 else float('inf')))

        actual_bar_width_px = best_length_m * self.scale_factor_px_per_m
        if actual_bar_width_px < 20: return # Still too small on screen after choosing nice number

        start_x_px = margin_px
        painter.setPen(QPen(self.palette().color(QPalette.Text), 1.5))
        painter.setBrush(Qt.NoBrush)

        # Draw main horizontal line
        painter.drawLine(QLineF(start_x_px, bar_y_pos_px, start_x_px + actual_bar_width_px, bar_y_pos_px))
        # Draw ticks at ends
        painter.drawLine(QLineF(start_x_px, bar_y_pos_px - bar_height_px/2, start_x_px, bar_y_pos_px + bar_height_px/2))
        painter.drawLine(QLineF(start_x_px + actual_bar_width_px, bar_y_pos_px - bar_height_px/2,
                                start_x_px + actual_bar_width_px, bar_y_pos_px + bar_height_px/2))

        # Draw label for the scale bar
        font = painter.font(); font.setPixelSize(10); painter.setFont(font); painter.setPen(self.palette().color(QPalette.Text))
        label_text = f"{best_length_m:g} m" # 'g' format: concise float/int (e.g., "50 m", "2.5 m")
        fm = QFontMetricsF(font)
        label_width_px = fm.horizontalAdvance(label_text)
        # Center label text above the scale bar
        painter.drawText(QPointF(start_x_px + (actual_bar_width_px - label_width_px)/2.0,
                                  bar_y_pos_px - bar_height_px/2.0 - 4), label_text)


    def _draw_view_info(self, painter: QPainter):
        """Draws view information like zoom level and mouse coordinates at the bottom-right."""
        font = painter.font(); font.setPixelSize(9); painter.setFont(font)
        painter.setPen(self.palette().color(QPalette.Midlight)) # Use palette color

        zoom_percentage = (self.scale_factor_px_per_m / self.NOMINAL_PX_PER_METER) * 100
        info_parts = [f"Zoom: {zoom_percentage:.0f}%"]

        # Get mouse coordinates relative to this widget
        mouse_local_pos_qpoint = self.mapFromGlobal(QCursor.pos())
        if self.rect().contains(mouse_local_pos_qpoint): # If mouse is over the canvas
            mouse_world_pos_qpointf = self.screen_to_world_coords(mouse_local_pos_qpoint.x(), mouse_local_pos_qpoint.y())
            info_parts.append(f"Cursor (Site): {mouse_world_pos_qpointf.x():.1f}m, {mouse_world_pos_qpointf.y():.1f}m")

        text_to_draw = "  |  ".join(info_parts)
        fm = QFontMetricsF(font)
        text_rect = fm.boundingRect(text_to_draw)
        # Position at bottom-right with margin
        painter.drawText(QPointF(self.width() - text_rect.width() - 10, self.height() - 8), text_to_draw)


    # --- Mouse and Keyboard Events ---
    def mousePressEvent(self, event: QMouseEvent):
        self.last_mouse_pos_screen = QPointF(event.pos()) # Store as QPointF for precision

        if event.button() == Qt.LeftButton:
            # Normal Left Click (select/drag) - not Shift+Click (which is for panning)
            if not (event.modifiers() & Qt.ShiftModifier):
                world_pos_click = self.screen_to_world_coords(self.last_mouse_pos_screen.x(), self.last_mouse_pos_screen.y())
                newly_selected_element: Optional[SiteElement] = None

                # Check external elements first, in reverse draw order (topmost visually)
                # Sorting by priority (lower number = higher priority = drawn last = checked first for click)
                clickable_elements = sorted(
                    [el for el in self.elements_list if not getattr(el, 'is_placed_inside_building', False)],
                    key=lambda el: SITE_ELEMENTS.get(el.name, {}).get('priority', 99), reverse=True
                )

                for el_candidate in clickable_elements:
                    if el_candidate.contains_point(world_pos_click.x(), world_pos_click.y()):
                        # Only truly select if it's not the unmovable Building, or if it's movable
                        if el_candidate.movable or el_candidate.name == "Building": # Allow selecting Building
                             newly_selected_element = el_candidate
                        break # Found topmost clickable element

                # Update selection state
                if self.selected_element_obj != newly_selected_element:
                    if self.selected_element_obj:
                        self.selected_element_obj.selected = False # Deselect old
                    self.selected_element_obj = newly_selected_element
                    if self.selected_element_obj:
                        self.selected_element_obj.selected = True # Select new
                    self.element_selected.emit(self.selected_element_obj) # Notify main app

                # If a movable element is selected, prepare for dragging
                if self.selected_element_obj and \
                (self.selected_element_obj.movable or self.selected_element_obj.name == "Building") and \
                not getattr(self.selected_element_obj, 'is_placed_inside_building', False):
                    self.is_dragging_element = True
                    self.drag_mouse_start_pos_screen = self.last_mouse_pos_screen
                    self.drag_element_start_pos_world = QPointF(self.selected_element_obj.x, self.selected_element_obj.y)
                    self.setCursor(Qt.ClosedHandCursor) # Change cursor to indicate dragging
                elif self.selected_element_obj: # Selected a non-movable element (e.g. Building)
                    self.setCursor(Qt.ArrowCursor) # Standard cursor
                self.update() # Repaint to show selection change

        # Panning: Middle mouse button OR Left button + Shift
        if event.button() == Qt.MiddleButton or \
           (event.button() == Qt.LeftButton and (event.modifiers() & Qt.ShiftModifier)):
            self.is_panning = True
            self.setCursor(Qt.SizeAllCursor) # Cursor indicating panning capability


    def mouseMoveEvent(self, event: QMouseEvent):
        current_pos_screen = QPointF(event.pos())
        delta_screen = current_pos_screen - self.last_mouse_pos_screen # Movement since last event

        if self.is_panning and (event.buttons() & Qt.MiddleButton or \
                               (event.buttons() & Qt.LeftButton and (QApplication.keyboardModifiers() & Qt.ShiftModifier))):
            self.view_offset_x_px += delta_screen.x()
            self.view_offset_y_px += delta_screen.y()
            self._update_cached_rects()
            self.view_changed.emit() # Notify main app that view changed
            self.update() # Repaint
        elif self.is_dragging_element and self.selected_element_obj and (event.buttons() & Qt.LeftButton):
            if abs(self.scale_factor_px_per_m) < 1e-9: return # Avoid division by zero

            # Calculate total mouse delta from drag start in screen coordinates
            total_mouse_delta_screen_x = current_pos_screen.x() - self.drag_mouse_start_pos_screen.x()
            total_mouse_delta_screen_y = current_pos_screen.y() - self.drag_mouse_start_pos_screen.y()

            # Convert total screen delta to world delta
            delta_world_x = total_mouse_delta_screen_x / self.scale_factor_px_per_m
            delta_world_y = total_mouse_delta_screen_y / self.scale_factor_px_per_m

            new_world_x = self.drag_element_start_pos_world.x() + delta_world_x
            new_world_y = self.drag_element_start_pos_world.y() + delta_world_y

            self.selected_element_obj.x = new_world_x
            self.selected_element_obj.y = new_world_y
            self._clamp_element_to_plot_boundaries(self.selected_element_obj) # Ensure it stays within plot
            self.update() # Repaint to show dragged element
            self.element_moved.emit(self.selected_element_obj) # Emit intermediate move for live updates elsewhere if needed
        else: # Not panning or dragging, handle hover effects
            self.update_cursor_for_hover(current_pos_screen)

        self.last_mouse_pos_screen = current_pos_screen
        if self.show_scale_bar_info : self.update() # For live mouse coordinate display update

    def update_cursor_for_hover(self, screen_pos: QPointF):
        """Updates the mouse cursor based on what's under it (e.g., movable element)."""
        if self.is_dragging_element or self.is_panning: return # Cursor already set for these actions

        world_hover_pos = self.screen_to_world_coords(screen_pos.x(), screen_pos.y())
        hovered_element: Optional[SiteElement] = None
        # Check in reverse draw order (topmost visually)
        clickable_elements = sorted(
            [el for el in self.elements_list if not getattr(el, 'is_placed_inside_building', False)],
            key=lambda el: SITE_ELEMENTS.get(el.name, {}).get('priority', 99), reverse=True
        )
        for el_candidate in clickable_elements:
            if el_candidate.contains_point(world_hover_pos.x(), world_hover_pos.y()):
                hovered_element = el_candidate
                break

        if hovered_element and hovered_element.movable and not getattr(hovered_element, 'is_placed_inside_building', False):
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)


    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.is_dragging_element:
            self.is_dragging_element = False
            if self.selected_element_obj:
                # Final clamping and emit definitive move event
                self._clamp_element_to_plot_boundaries(self.selected_element_obj)
                self.element_moved.emit(self.selected_element_obj) # Final emit after drag completion
            self.update_cursor_for_hover(QPointF(event.pos())) # Reset cursor based on final position
            self.update()

        elif self.is_panning and (event.button() == Qt.MiddleButton or \
            (event.button() == Qt.LeftButton and (QApplication.keyboardModifiers() & Qt.ShiftModifier))): # Check modifier for left button release too
            self.is_panning = False
            self.update_cursor_for_hover(QPointF(event.pos())) # Reset cursor
            self.update()


    def wheelEvent(self, event: QWheelEvent):
        """Handles mouse wheel events for zooming."""
        mouse_pos_screen = QPointF(event.position() if hasattr(event, 'position') else event.pos()) # Qt5/Qt6 compatibility
        # angleDelta().y() is typically +/- 120 for one step of a standard mouse wheel
        delta_degrees = event.angleDelta().y() / 8.0 # Standard way to get degrees
        num_steps = delta_degrees / 15.0 # Convert degrees to number of "steps"

        zoom_factor = 1.0 + (num_steps * self.zoom_sensitivity_wheel)
        self._zoom_view(zoom_factor, mouse_pos_screen) # Zoom centered on mouse cursor
        self.update() # Repaint


    def keyPressEvent(self, event: QKeyEvent):
        """Handles keyboard events for panning, zooming, and element manipulation."""
        key_action_taken = False
        pan_step_px = 60.0 # Pan step in screen pixels for arrow keys
        # Element move step in world meters (coarse and fine)
        element_move_step_m_coarse = max(0.5, self.plot_width_m / 100.0) # e.g. 0.5m or 1% of plot width
        element_move_step_m_fine = max(0.1, element_move_step_m_coarse / 5.0)

        current_element_move_step_m = element_move_step_m_fine if (event.modifiers() & Qt.ShiftModifier) else element_move_step_m_coarse

        # View Manipulation (Panning and Zooming with keys)
        # No modifiers for these, or specific ones if needed to avoid conflict.
        if event.modifiers() == Qt.NoModifier:
            if event.key() == Qt.Key_Left:  self.view_offset_x_px += pan_step_px; key_action_taken = True
            elif event.key() == Qt.Key_Right: self.view_offset_x_px -= pan_step_px; key_action_taken = True
            elif event.key() == Qt.Key_Up:    self.view_offset_y_px += pan_step_px; key_action_taken = True
            elif event.key() == Qt.Key_Down:  self.view_offset_y_px -= pan_step_px; key_action_taken = True
            elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal: # '+' or '=' for zoom in
                self._zoom_view(self.zoom_sensitivity_keys, QPointF(self.width()/2, self.height()/2)); key_action_taken = True # Zoom to center
            elif event.key() == Qt.Key_Minus: # '-' for zoom out
                self._zoom_view(1.0 / self.zoom_sensitivity_keys, QPointF(self.width()/2, self.height()/2)); key_action_taken = True
            elif event.key() == Qt.Key_Home or event.key() == Qt.Key_F: # Home or 'F' to fit plot to view
                self.fit_plot_to_view(); key_action_taken = True # fit_plot_to_view calls update

        # Selected Element Manipulation (using Ctrl + Arrow keys for movement, 'R' for rotation)
        if self.selected_element_obj and self.selected_element_obj.movable and \
           not getattr(self.selected_element_obj, 'is_placed_inside_building', False):
            sel_el = self.selected_element_obj
            orig_x_w, orig_y_w = sel_el.x, sel_el.y
            orig_rot_deg = sel_el.rotation
            orig_w_w, orig_h_w = sel_el.width, sel_el.height # For rotation dimension swap

            element_changed_by_key = False
            if event.modifiers() == Qt.ControlModifier: # Ctrl + Arrow for position
                if event.key() == Qt.Key_Left:   sel_el.x -= current_element_move_step_m; element_changed_by_key = True
                elif event.key() == Qt.Key_Right: sel_el.x += current_element_move_step_m; element_changed_by_key = True
                elif event.key() == Qt.Key_Up:     sel_el.y -= current_element_move_step_m; element_changed_by_key = True
                elif event.key() == Qt.Key_Down:   sel_el.y += current_element_move_step_m; element_changed_by_key = True
            elif event.key() == Qt.Key_R and event.modifiers() == Qt.NoModifier: # 'R' to rotate (no modifiers)
                rotation_step_deg = 15.0 if (event.modifiers() & Qt.ShiftModifier) else 90.0 # Fine or coarse rotation
                sel_el.rotation = (orig_rot_deg + rotation_step_deg) % 360.0
                # Snap to exact 90s if close after 90-degree step
                if rotation_step_deg == 90.0:
                     sel_el.rotation = round(sel_el.rotation / 90.0) * 90.0

                # Handle dimension swap if orientation changes significantly (e.g. landscape to portrait)
                is_major_axis_flip = (int(sel_el.rotation % 180.0) // 90) != (int(orig_rot_deg % 180.0) // 90)
                if is_major_axis_flip and abs(orig_w_w - orig_h_w) > 1e-3: # If not a square
                    # Preserve center point during dimension swap
                    center_x_w, center_y_w = orig_x_w + orig_w_w / 2.0, orig_y_w + orig_h_w / 2.0
                    sel_el.width, sel_el.height = orig_h_w, orig_w_w # Swap dimensions
                    sel_el.x = center_x_w - sel_el.width / 2.0 # Recalculate top-left
                    sel_el.y = center_y_w - sel_el.height / 2.0
                element_changed_by_key = True

            if element_changed_by_key:
                self._clamp_element_to_plot_boundaries(sel_el)
                self.element_moved.emit(sel_el) # Emit signal after any property change
                key_action_taken = True # Ensure view update

        # General Keys
        if event.key() == Qt.Key_Escape:
            if self.selected_element_obj:
                self.clear_selection()
                key_action_taken = True
            # Could add other Escape behaviors, e.g., cancel current drag/pan (though mouse release handles this)

        if key_action_taken:
            self._update_cached_rects() # If view offset changed
            self.view_changed.emit()   # If view parameters (zoom/pan) changed
            self.update()              # Repaint canvas
        else:
            super().keyPressEvent(event) # Pass unhandled key events to parent


    def _zoom_view(self, factor: float, center_point_screen: QPointF):
        """Zooms the view by a given factor, centered on a specific screen point."""
        # Current world coordinates at the center_point_screen (mouse cursor or widget center)
        world_pos_at_center_s = self.screen_to_world_coords(center_point_screen.x(), center_point_screen.y())

        # Calculate new scale factor, clamping to min/max limits
        new_scale_px_per_m = self.scale_factor_px_per_m * factor
        new_scale_px_per_m = max(self.MIN_SCALE_FACTOR_PX_PER_M,
                                 min(self.MAX_SCALE_FACTOR_PX_PER_M, new_scale_px_per_m))

        if abs(new_scale_px_per_m - self.scale_factor_px_per_m) < 1e-6 : return # No significant change in scale

        self.scale_factor_px_per_m = new_scale_px_per_m

        # Adjust view offset so that world_pos_at_center_s remains at center_point_screen
        self.view_offset_x_px = center_point_screen.x() - world_pos_at_center_s.x() * self.scale_factor_px_per_m
        self.view_offset_y_px = center_point_screen.y() - world_pos_at_center_s.y() * self.scale_factor_px_per_m

        self._update_cached_rects() # Update screen representation based on new scale/offset
        self.view_changed.emit()    # Notify that view parameters changed
        # self.update() will be called by the event handler that called _zoom_view


    def _clamp_element_to_plot_boundaries(self, element: SiteElement):
        """Ensures an element (after move or resize) stays within the plot boundaries."""
        if getattr(element, 'is_placed_inside_building', False): return # Internal elements not clamped this way

        # For rotated elements, clamping is more complex. We need to find the AABB of the rotated element
        # and then shift the element if this AABB is out of bounds.
        corners_w = element.get_corners()
        if not corners_w:
            logger.warning(f"Cannot clamp element {element.name}, failed to get corners.")
            return

        min_x_w = min(c[0] for c in corners_w)
        max_x_w = max(c[0] for c in corners_w)
        min_y_w = min(c[1] for c in corners_w)
        max_y_w = max(c[1] for c in corners_w)

        delta_x_w, delta_y_w = 0.0, 0.0
        if min_x_w < 0.0: delta_x_w = -min_x_w # Shift right
        elif max_x_w > self.plot_width_m: delta_x_w = self.plot_width_m - max_x_w # Shift left

        if min_y_w < 0.0: delta_y_w = -min_y_w # Shift down
        elif max_y_w > self.plot_height_m: delta_y_w = self.plot_height_m - max_y_w # Shift up

        if abs(delta_x_w) > 1e-3 or abs(delta_y_w) > 1e-3: # If any shift is needed
            element.x += delta_x_w
            element.y += delta_y_w
            # logger.debug(f"Clamped element {element.name} by ({delta_x_w:.1f}, {delta_y_w:.1f})")


    def resizeEvent(self, event: QResizeEvent):
        """Handles widget resize events."""
        super().resizeEvent(event)
        self._update_cached_rects() # Update widget_viewport_rect_f
        # Option: maintain current center world point and scale, or call fit_plot_to_view,
        # or just let the existing view parameters dictate how content is shown in new size.
        # For now, just update rects and assume user will pan/zoom if needed, or app calls fit_plot_to_view.
        self.view_changed.emit() # Viewport size changed, might affect scale bar or other overlays
        self.update() # Trigger repaint with new size


    def clear_selection(self):
        """Clears the currently selected element."""
        if self.selected_element_obj:
            self.selected_element_obj.selected = False
            old_selection = self.selected_element_obj
            self.selected_element_obj = None
            self.element_selected.emit(None) # Notify main app that selection is cleared
            logger.debug(f"SiteCanvas: Selection cleared for element '{old_selection.name}'.")
            self.update() # Repaint to remove selection highlight


    def set_display_options(self, show_grid: bool, show_dims: bool, show_labels: bool,
                            show_crane_radius: bool, show_internal_markers: bool,
                            show_oob_elements: Optional[bool] = None): # Added oob option if used
        """Updates the canvas display options based on external settings."""
        self.show_grid_lines = show_grid
        self.show_element_dimensions = show_dims
        self.show_element_labels = show_labels
        self.show_crane_operational_radius = show_crane_radius
        self.show_internal_element_markers = show_internal_markers
        if show_oob_elements is not None: # Only update if explicitly passed
            self.show_out_of_bounds_elements = show_oob_elements

        logger.debug(f"SiteCanvas: Display options updated - Grid:{self.show_grid_lines}, "
                     f"Dims:{self.show_element_dimensions}, Labels:{self.show_element_labels}, "
                     f"CraneR:{self.show_crane_operational_radius}, InternalM:{self.show_internal_element_markers}, "
                     f"ShowOOB:{self.show_out_of_bounds_elements}")
        self.update() # Trigger repaint with new display options

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Handles right-click context menu requests."""
        screen_pos = event.pos()
        world_pos = self.screen_to_world_coords(screen_pos.x(), screen_pos.y())

        hovered_element: Optional[SiteElement] = None
        clickable_elements = sorted(
            [el for el in self.elements_list if not getattr(el, 'is_placed_inside_building', False)],
            key=lambda el: SITE_ELEMENTS.get(el.name, {}).get('priority', 99), reverse=True
        )
        for el_candidate in clickable_elements:
            if el_candidate.contains_point(world_pos.x(), world_pos.y()):
                hovered_element = el_candidate
                break
        
        # Emit a signal that the main application can connect to for creating the context menu
        self.context_menu_requested_at_world.emit(world_pos, hovered_element)
        # The main app will then create and show the QMenu at event.globalPos()


class ElementPropertiesWidget(QWidget):
    """
    Widget for displaying and editing properties of a selected site element.
    Handles different states for external, internal, and fixed (Building) elements.
    Provides feedback on area and other derived properties.
    """
    properties_changed = pyqtSignal(object) # Emits the SiteElement object that changed
    request_element_delete = pyqtSignal(object) # New: Signal to request deletion of the element
    request_element_duplicate = pyqtSignal(object) # New: Signal to request duplication

    def __init__(self, parent=None):
        super().__init__(parent)
        self.element: Optional[SiteElement] = None
        self._is_updating_ui = False # Flag to prevent signal loops during UI updates
        self.initUI()
        self.setMinimumWidth(300) # Ensure it has some decent width
        logger.debug("ElementPropertiesWidget initialized.")

    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10) # Consistent spacing
        main_layout.setContentsMargins(8,8,8,8) # Add some padding

        self.title_label = QLabel("No Element Selected")
        font = self.title_label.font()
        font.setPointSize(14)
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)

        # --- General Properties Group ---
        general_group = QGroupBox("General Information")
        general_layout = QGridLayout(general_group)
        general_layout.setColumnStretch(1, 1)
        general_layout.setVerticalSpacing(6) # Spacing between rows in grid

        row = 0
        general_layout.addWidget(QLabel("Name:"), row, 0)
        self.name_value_label = QLabel("-")
        self.name_value_label.setStyleSheet("font-weight: bold;")
        self.name_value_label.setTextInteractionFlags(Qt.TextSelectableByMouse) # Allow copying name
        general_layout.addWidget(self.name_value_label, row, 1)
        row +=1

        general_layout.addWidget(QLabel("Type:"), row, 0)
        self.type_value_label = QLabel("-") # e.g., SITE_ELEMENTS key
        self.type_value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        general_layout.addWidget(self.type_value_label, row, 1)
        row +=1

        general_layout.addWidget(QLabel("Placement:"), row, 0)
        self.placement_status_label = QLabel("-") # "External" or "Internal (Building)"
        general_layout.addWidget(self.placement_status_label, row, 1)
        row +=1

        self.placeable_info_label = QLabel("") # For "Eligible for internal placement"
        self.placeable_info_label.setStyleSheet(f"font-style: italic; color: {DARK_THEME_INFO_TEXT_COLOR.name()};")
        general_layout.addWidget(self.placeable_info_label, row, 0, 1, 2) # Span both columns
        row +=1

        main_layout.addWidget(general_group)

        # --- Transform Properties Group ---
        self.transform_group_box = QGroupBox("Transform & Dimensions") # Keep ref to group box
        self.transform_layout = QGridLayout(self.transform_group_box)
        self.transform_layout.setColumnStretch(1, 1)
        self.transform_layout.setVerticalSpacing(6)

        row = 0
        self.transform_layout.addWidget(QLabel("Position X (m):"), row, 0)
        self.pos_x_input = QDoubleSpinBox()
        self.pos_x_input.setRange(-10000, 10000); self.pos_x_input.setDecimals(2); self.pos_x_input.setSingleStep(0.5)
        self.pos_x_input.setToolTip("Horizontal position of the element's top-left corner (world coordinates).")
        self.pos_x_input.valueChanged.connect(self._on_property_changed)
        self.transform_layout.addWidget(self.pos_x_input, row, 1)
        row +=1

        self.transform_layout.addWidget(QLabel("Position Y (m):"), row, 0)
        self.pos_y_input = QDoubleSpinBox()
        self.pos_y_input.setRange(-10000, 10000); self.pos_y_input.setDecimals(2); self.pos_y_input.setSingleStep(0.5)
        self.pos_y_input.setToolTip("Vertical position of the element's top-left corner (world coordinates).")
        self.pos_y_input.valueChanged.connect(self._on_property_changed)
        self.transform_layout.addWidget(self.pos_y_input, row, 1)
        row +=1

        self.transform_layout.addWidget(QLabel("Width (m):"), row, 0)
        self.width_input = QDoubleSpinBox()
        self.width_input.setRange(0.1, 10000); self.width_input.setDecimals(2); self.width_input.setSingleStep(0.5)
        self.width_input.setToolTip("Width of the element.")
        self.width_input.valueChanged.connect(self._on_property_changed)
        self.transform_layout.addWidget(self.width_input, row, 1)
        row +=1

        self.transform_layout.addWidget(QLabel("Height (m):"), row, 0)
        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(0.1, 10000); self.height_input.setDecimals(2); self.height_input.setSingleStep(0.5)
        self.height_input.setToolTip("Height of the element.")
        self.height_input.valueChanged.connect(self._on_property_changed)
        self.transform_layout.addWidget(self.height_input, row, 1)
        row +=1

        self.transform_layout.addWidget(QLabel("Rotation (°):"), row, 0)
        self.rotation_input = QSpinBox()
        self.rotation_input.setRange(0, 359); self.rotation_input.setSingleStep(1); self.rotation_input.setWrapping(True)
        self.rotation_input.setSuffix("°")
        self.rotation_input.setToolTip("Rotation angle of the element around its center (0-359 degrees).")
        self.rotation_input.valueChanged.connect(self._on_property_changed)
        self.transform_layout.addWidget(self.rotation_input, row, 1)
        row +=1

        self.transform_layout.addWidget(QLabel("Area (m²):"), row, 0)
        self.area_label = QLabel("0.00")
        self.area_label.setStyleSheet(f"font-style: italic; color: {DARK_THEME_INFO_TEXT_COLOR.name()};")
        self.area_label.setToolTip("Calculated area (Width × Height).")
        self.transform_layout.addWidget(self.area_label, row, 1)
        row +=1

        main_layout.addWidget(self.transform_group_box)

        # --- Attributes Group ---
        attributes_group = QGroupBox("Attributes & Appearance")
        attributes_layout = QGridLayout(attributes_group) # Use QGridLayout for better alignment
        attributes_layout.setColumnStretch(1,1)

        self.movable_checkbox = QCheckBox("Movable by Optimizer")
        self.movable_checkbox.setToolTip("If checked, the optimization algorithm can move this element.")
        self.movable_checkbox.toggled.connect(self._on_property_changed)
        attributes_layout.addWidget(self.movable_checkbox, 0, 0)

        self.color_button = QPushButton("Change Color...")
        self.color_button.setToolTip("Select a new color for this element.")
        self.color_button.clicked.connect(self._change_element_color)
        attributes_layout.addWidget(self.color_button, 0, 1) # Place next to checkbox

        main_layout.addWidget(attributes_group)

        # --- Actions Group ---
        actions_group = QGroupBox("Element Actions")
        actions_layout = QHBoxLayout(actions_group)

        self.duplicate_button = QPushButton(QIcon.fromTheme("edit-copy", QIcon(":/icons/duplicate.png")), "Duplicate") # Example with theme icon fallback
        self.duplicate_button.setToolTip("Create a copy of this element (if allowed).")
        self.duplicate_button.clicked.connect(self._on_duplicate_requested)
        actions_layout.addWidget(self.duplicate_button)

        self.delete_button = QPushButton(QIcon.fromTheme("edit-delete", QIcon(":/icons/delete.png")), "Delete")
        self.delete_button.setToolTip("Remove this element from the layout (if allowed).")
        self.delete_button.setStyleSheet(f"QPushButton {{ color: #FF6B6B; font-weight: bold; background-color: {QColor(80,40,40).name()}; border: 1px solid {QColor(120,60,60).name()}; }} "
                                         f"QPushButton:hover {{ background-color: {QColor(100,50,50).name()}; }} "
                                         f"QPushButton:pressed {{ background-color: {QColor(70,30,30).name()}; }}")
        self.delete_button.clicked.connect(self._on_delete_requested)
        actions_layout.addWidget(self.delete_button)

        main_layout.addWidget(actions_group)

        main_layout.addStretch(1) # Push content to top
        self.setLayout(main_layout)

        self._set_all_fields_enabled_status(False) # Disable all inputs initially


    def set_element(self, element_obj: Optional['SiteElement']):
        self._is_updating_ui = True
        self.element = element_obj

        if self.element:
            element_config = SITE_ELEMENTS.get(self.element.name, {})
            is_internal = getattr(self.element, 'is_placed_inside_building', False)
            is_building = self.element.name == 'Building'

            self.title_label.setText(f"{self.element.name}") # Simplified title

            # General Info
            self.name_value_label.setText(self.element.name)
            self.type_value_label.setText(self.element.name) # Using name as type for now
            self.placement_status_label.setText("Internal (in Building)" if is_internal else "External")

            if element_config.get('placeable_inside_building', False) and not is_internal and not is_building:
                self.placeable_info_label.setText("(Can be placed inside Building)")
            else:
                self.placeable_info_label.setText("")

            # Transform & Dimensions
            self.pos_x_input.setValue(self.element.x)
            self.pos_y_input.setValue(self.element.y)
            self.width_input.setValue(self.element.width)
            self.height_input.setValue(self.element.height)
            self.rotation_input.setValue(int(round(self.element.rotation))) # QSpinBox expects int
            self.area_label.setText(f"{self.element.area():.2f} m²")

            # Attributes
            self.movable_checkbox.setChecked(self.element.movable)
            self._update_color_button_style()

            # --- Determine Field Enabled States ---
            # Transform fields (X, Y, Width, Height, Rotation)
            # Building: X,Y might be adjustable if we decide to allow manual building repositioning here.
            # For now, assume Building's transform is fixed by its initialization logic.
            # Internal elements: Transform is relative/managed differently, not directly editable here.
            can_edit_transform_xy = not is_internal and not is_building # Movable elements
            can_edit_transform_wh_rot = not is_internal and not is_building and element_config.get('resizable', True)

            self.pos_x_input.setEnabled(can_edit_transform_xy)
            self.pos_y_input.setEnabled(can_edit_transform_xy)
            self.width_input.setEnabled(can_edit_transform_wh_rot)
            self.height_input.setEnabled(can_edit_transform_wh_rot)
            self.rotation_input.setEnabled(can_edit_transform_wh_rot) # Rotation often tied to resizable

            # If element is fixed size (not resizable), W/H inputs should be read-only
            if not element_config.get('resizable', True) and not is_building: # Building is special case for resizable flag
                self.width_input.setReadOnly(True)
                self.height_input.setReadOnly(True)
            else:
                self.width_input.setReadOnly(False)
                self.height_input.setReadOnly(False)


            # Movable checkbox
            # Building has 'movable: False' in SITE_ELEMENTS. Internal elements not movable by this flag.
            can_toggle_movable_flag = not is_building and not is_internal
            self.movable_checkbox.setEnabled(can_toggle_movable_flag)
            # If SITE_ELEMENTS defines movable=False (e.g. Crane), this checkbox should be disabled.
            if not element_config.get('movable', True) and not is_building and not is_internal:
                 self.movable_checkbox.setEnabled(False)


            # Color button
            self.color_button.setEnabled(True) # Always allow color change

            # Action buttons
            can_delete = not is_building # Building cannot be deleted
            self.delete_button.setEnabled(can_delete)

            # Duplication: Not for Building, not for internal items.
            # Also, consider if some elements are unique by nature (e.g., only one "Building").
            # Fixed, non-resizable elements might also not be suitable for simple duplication
            # if their properties are tightly controlled.
            can_duplicate = not is_building and not is_internal and self.element.movable # Simple rule for now
            self.duplicate_button.setEnabled(can_duplicate)

            self.transform_group_box.setEnabled(True) # Enable the group if any transform is editable
            attributes_group = self.findChild(QGroupBox, "Attributes & Appearance") # Find by object name if needed
            if attributes_group: attributes_group.setEnabled(True)
            actions_group = self.findChild(QGroupBox, "Element Actions")
            if actions_group: actions_group.setEnabled(True)
            general_group = self.findChild(QGroupBox, "General Information")
            if general_group: general_group.setEnabled(True)


        else: # No element selected
            self.title_label.setText("No Element Selected")
            self.name_value_label.setText("-")
            self.type_value_label.setText("-")
            self.placement_status_label.setText("-")
            self.placeable_info_label.setText("")

            self.pos_x_input.setValue(0)
            self.pos_y_input.setValue(0)
            self.width_input.setValue(0)
            self.height_input.setValue(0)
            self.rotation_input.setValue(0)
            self.area_label.setText("N/A")

            self.movable_checkbox.setChecked(False)
            self.color_button.setStyleSheet("") # Reset color button style

            self._set_all_fields_enabled_status(False)

        self._is_updating_ui = False

    def _set_all_fields_enabled_status(self, enabled: bool):
        """Enable/disable all input fields and groups based on whether an element is selected."""
        # Iterate over all relevant input widgets and groups
        self.transform_group_box.setEnabled(enabled)
        # Need to find other groups by object name or iterate through layout if not stored as attributes
        attributes_group = self.findChild(QGroupBox, "Attributes & Appearance") # Example, if you set objectName
        if attributes_group: attributes_group.setEnabled(enabled)
        actions_group = self.findChild(QGroupBox, "Element Actions")
        if actions_group: actions_group.setEnabled(enabled)
        general_group = self.findChild(QGroupBox, "General Information")
        if general_group: general_group.setEnabled(enabled)


        # If disabling all, specifically handle individual input widgets within enabled groups
        if not enabled:
            self.pos_x_input.setEnabled(False)
            self.pos_y_input.setEnabled(False)
            self.width_input.setEnabled(False)
            self.height_input.setEnabled(False)
            self.rotation_input.setEnabled(False)
            self.movable_checkbox.setEnabled(False)
            self.color_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            self.duplicate_button.setEnabled(False)


    def _on_property_changed(self, value=None):
        if self._is_updating_ui or not self.element:
            return

        changed_property_sender = self.sender()
        original_center_x, original_center_y = 0,0
        is_transform_change = changed_property_sender in [self.pos_x_input, self.pos_y_input, self.width_input, self.height_input, self.rotation_input]

        if is_transform_change:
            # Store original center before any dimension or rotation changes for potential re-centering
            original_center_x = self.element.x + self.element.width / 2.0
            original_center_y = self.element.y + self.element.height / 2.0

        # Update element properties from UI fields if they are enabled
        if self.pos_x_input.isEnabled() and changed_property_sender == self.pos_x_input:
            self.element.x = self.pos_x_input.value()
        if self.pos_y_input.isEnabled() and changed_property_sender == self.pos_y_input:
            self.element.y = self.pos_y_input.value()
        if self.width_input.isEnabled() and changed_property_sender == self.width_input:
            self.element.width = self.width_input.value()
        if self.height_input.isEnabled() and changed_property_sender == self.height_input:
            self.element.height = self.height_input.value()
        if self.rotation_input.isEnabled() and changed_property_sender == self.rotation_input:
            self.element.rotation = float(self.rotation_input.value())
        if self.movable_checkbox.isEnabled() and changed_property_sender == self.movable_checkbox:
            self.element.movable = self.movable_checkbox.isChecked()

        # Post-change adjustments, like re-centering after W/H/Rotation change
        if is_transform_change:
            element_config = SITE_ELEMENTS.get(self.element.name, {})
            # If dimensions changed, or rotation changed for non-square resizable item, re-center.
            # The logic for swapping W/H on 90-deg rotation is complex if Aspect Ratio is fixed.
            # For now, just re-center if W, H, or Rotation changed.
            # A fixed aspect ratio should be handled by SpaceRequirementsWidget influencing W/H.
            # This panel directly edits W/H.

            # If resizable, and W or H changed, OR rotation changed, adjust X,Y to keep center
            if element_config.get('resizable', True):
                 if changed_property_sender in [self.width_input, self.height_input, self.rotation_input]:
                    new_center_x = self.element.x + self.element.width / 2.0
                    new_center_y = self.element.y + self.element.height / 2.0

                    # If element was not rotated before, or rotation is 0, simple re-centering
                    if self.element.rotation == 0 or (changed_property_sender != self.rotation_input):
                        self.element.x = original_center_x - self.element.width / 2.0
                        self.element.y = original_center_y - self.element.height / 2.0
                    # If rotation changed, the original_center_x/y was for the unrotated state.
                    # The new x,y will be relative to the rotated state. This is complex.
                    # The canvas handles rotation drawing around the element's current x,y + w/2, h/2.
                    # For direct W/H input, keeping top-left fixed and letting user adjust rotation/position is simpler.
                    # Let's simplify: if W/H changed, ensure new x,y maintain the original center.
                    # If rotation changed, the current x,y is the pivot for rotation (top-left).
                    # No, pivot is center. So if rotation is the only thing changed, x,y (top-left) must adjust.

                    if changed_property_sender == self.width_input or changed_property_sender == self.height_input:
                         self.element.x = original_center_x - self.element.width / 2.0
                         self.element.y = original_center_y - self.element.height / 2.0

                    # Update UI for potentially re-centered X, Y without re-triggering signals
                    if abs(self.element.x - self.pos_x_input.value()) > 1e-3 or \
                       abs(self.element.y - self.pos_y_input.value()) > 1e-3:
                        self._is_updating_ui = True
                        self.pos_x_input.setValue(self.element.x)
                        self.pos_y_input.setValue(self.element.y)
                        self._is_updating_ui = False

        # Update area label
        self.area_label.setText(f"{self.element.area():.2f} m²")
        self.properties_changed.emit(self.element)
        logger.debug(f"Property '{changed_property_sender.objectName if hasattr(changed_property_sender, 'objectName') else 'unknown'}' changed for {self.element.name}")


    def _change_element_color(self):
        if not self.element: return
        initial_color = self.element.color if hasattr(self.element, 'color') else QColor(200,200,200)
        new_color = QColorDialog.getColor(initial_color, self, f"Select Color for {self.element.name}")
        if new_color.isValid():
            self.element.color = new_color
            self._update_color_button_style()
            if not self._is_updating_ui:
                self.properties_changed.emit(self.element)
                logger.debug(f"Color changed for {self.element.name} to {new_color.name()}")


    def _update_color_button_style(self):
            if self.element and hasattr(self.element, 'color'):
                c = self.element.color
                brightness = (c.redF() * 0.299 + c.greenF() * 0.587 + c.blueF() * 0.114)
                text_color = "black" if brightness > 0.65 else "white" # Adjusted threshold for better contrast
                self.color_button.setStyleSheet(
                    f"QPushButton {{ background-color: {c.name()}; color: {text_color}; border: 1px solid {DARK_THEME_BORDER_COLOR.name()}; padding: 4px; }}"
                    f"QPushButton:hover {{ background-color: {c.lighter(115).name()}; }}"
                    f"QPushButton:pressed {{ background-color: {c.darker(115).name()}; }}"
                )
            else:
                self.color_button.setStyleSheet("") # Reset to default (will pick up global app style)


    def _on_delete_requested(self):
        if self.element: # Button should be disabled if no element or if it's the Building
            self.request_element_delete.emit(self.element)


    def _on_duplicate_requested(self):
        if self.element: # Button should be disabled if not appropriate
            self.request_element_duplicate.emit(self.element)

# --- SpaceRequirementsWidget ---
class SpaceRequirementsWidget(QWidget):
    """
    Widget for specifying space requirements for different site elements.
    Allows setting total area directly or calculating it based on personnel counts
    for applicable elements (e.g., offices, dormitories).
    Provides suggested dimensions based on area and typical aspect ratios.
    """
    # Emits the full dictionary of {element_name: area_in_sq_m}
    requirements_changed = pyqtSignal(dict)
    # Emits the personnel configuration data {element_name: {'count': N, 'area_per_person': X}}
    personnel_config_changed = pyqtSignal(dict)


    def __init__(self, parent: Optional[QWidget] = None,
                 site_elements_config: Optional[Dict] = None,
                 default_personnel_config: Optional[Dict] = None,
                 default_space_requirements: Optional[Dict] = None):
        super().__init__(parent)

        # Use provided configurations or fall back to global defaults
        self.SITE_ELEMENTS_CONFIG = site_elements_config if site_elements_config is not None else SITE_ELEMENTS
        self.DEFAULT_PERSONNEL_CONFIG = default_personnel_config if default_personnel_config is not None else DEFAULT_PERSONNEL_CONFIG
        self.DEFAULT_SPACE_REQUIREMENTS = default_space_requirements if default_space_requirements is not None else DEFAULT_SPACE_REQUIREMENTS

        # Initialize internal state from these effective defaults
        # These will be updated by user interaction or when data is loaded.
        self.current_requirements: Dict[str, float] = self.DEFAULT_SPACE_REQUIREMENTS.copy()
        self.current_personnel_config_data: Dict[str, Dict] = {
            name: data.copy() for name, data in self.DEFAULT_PERSONNEL_CONFIG.items()
        }

        # Identify elements for which personnel-based calculation is applicable
        self.personnel_controlled_elements_names: List[str] = list(self.current_personnel_config_data.keys())

        # Map to store references to the dynamically created input widgets for each element
        # Structure: {element_name: {'p_count_spin': QSpinBox, 'app_spin': QDoubleSpinBox, ...}}
        self.inputs_widgets_map: Dict[str, Dict[str, QWidget]] = {}

        self._is_programmatic_update: bool = False # Flag to prevent signal loops during UI updates

        self.initUI()
        self._populate_ui_from_current_state() # Populate UI based on initial internal state
        logger.debug("SpaceRequirementsWidget initialized.")


    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5, 5, 5, 5) # Reduced margins

        title_label = QLabel("Space Requirements Configuration")
        font = title_label.font(); font.setPointSize(13); font.setBold(True)
        title_label.setFont(font)
        main_layout.addWidget(title_label)

        info_label = QLabel(
            "Define the required area for each type of site facility. For facilities like offices and "
            "dormitories, you can specify personnel counts and area per person, or choose to manually "
            "set the total area. The main 'Building' footprint is configured separately under 'Site & Building Configuration'."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"font-size: 9pt; color: {DARK_THEME_INFO_TEXT_COLOR.name()};")
        main_layout.addWidget(info_label)

        # --- Scrollable Area for Element Inputs ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.StyledPanel)
        scroll_area.setFrameShadow(QFrame.Sunken) # Give it a slight inset look
        scroll_content_widget = QWidget()
        self.grid_layout = QGridLayout(scroll_content_widget) # Layout for the scrollable content
        self.grid_layout.setVerticalSpacing(10)
        self.grid_layout.setHorizontalSpacing(8)

        # Column stretches for better layout management within the grid
        self.grid_layout.setColumnStretch(0, 2)  # Element Name (more space)
        self.grid_layout.setColumnStretch(1, 3)  # Personnel Input GroupBox (most space)
        self.grid_layout.setColumnStretch(2, 1)  # Total Area SpinBox
        self.grid_layout.setColumnStretch(3, 1)  # Suggested Dimensions Label

        # Headers for the grid
        headers = ["Facility Type", "Personnel-Based Calculation", "Total Area (m²)", "Suggested W×H (m)"]
        header_tooltips = [
            "Name of the temporary site facility.",
            "For offices or dormitories: define area based on personnel numbers, or check 'Set Manually' to directly input total area.",
            "Total required floor area for the facility in square meters.",
            "Approximate Width × Height based on the total area and a typical aspect ratio for this facility type. Actual dimensions are determined during layout."
        ]
        for col, (txt, tooltip_text) in enumerate(zip(headers, header_tooltips)):
            lbl = QLabel(f"<b>{txt}</b>") # Bold header text
            lbl.setAlignment(Qt.AlignCenter if col > 0 else Qt.AlignLeft | Qt.AlignVCenter)
            lbl.setToolTip(tooltip_text)
            lbl.setStyleSheet(f"padding-bottom: 3px; border-bottom: 1px solid {DARK_THEME_BORDER_COLOR.name()}; color: {DARK_THEME_TEXT_COLOR.name()};")
            self.grid_layout.addWidget(lbl, 0, col)

        # --- Populate rows for each site element type (excluding 'Building') ---
        current_row_idx = 1 # Start after headers
        # Sort elements, e.g., by priority or name, for consistent display
        sorted_element_names = sorted(
            [name for name in self.SITE_ELEMENTS_CONFIG.keys() if name != 'Building'],
            key=lambda name: self.SITE_ELEMENTS_CONFIG[name].get('priority', 99)
        )

        for el_name in sorted_element_names:
            self.inputs_widgets_map[el_name] = {} # Initialize dict for this element's widgets

            # Element Name Label
            el_name_label = QLabel(el_name)
            el_name_label.setStyleSheet("font-weight: bold;")
            self.grid_layout.addWidget(el_name_label, current_row_idx, 0, Qt.AlignTop | Qt.AlignLeft)

            # Personnel Input Section (conditionally shown)
            if el_name in self.personnel_controlled_elements_names:
                personnel_group_box = QGroupBox() # No title, just for grouping inputs
                personnel_group_box.setFlat(True) # Less obtrusive border
                personnel_group_box.setStyleSheet(
                    f"QGroupBox {{ border: 1px solid {DARK_THEME_BORDER_COLOR.name()}; border-radius: 3px; margin-top: 0px; padding: 4px; }}"
                )
                personnel_grid = QGridLayout(personnel_group_box)
                personnel_grid.setSpacing(6)
                personnel_grid.setContentsMargins(3,3,3,3)


                override_cb = QCheckBox("Set Manually")
                override_cb.setToolTip("Check to manually define the total area below, ignoring personnel-based calculations for this facility.")
                personnel_grid.addWidget(override_cb, 0, 0, 1, 2) # Span both columns
                self.inputs_widgets_map[el_name]['override_cb'] = override_cb

                personnel_grid.addWidget(QLabel("# People:"), 1, 0)
                p_count_spin = QSpinBox()
                p_count_spin.setRange(0, 2000); p_count_spin.setSingleStep(1)
                p_count_spin.setToolTip("Number of personnel to be accommodated.")
                personnel_grid.addWidget(p_count_spin, 1, 1)
                self.inputs_widgets_map[el_name]['p_count_spin'] = p_count_spin

                personnel_grid.addWidget(QLabel("Area/Person (m²):"), 2, 0)
                app_spin = QDoubleSpinBox()
                app_spin.setRange(0.5, 100.0); app_spin.setDecimals(1)
                app_spin.setSingleStep(0.1); app_spin.setSuffix(" m²")
                app_spin.setToolTip("Standard area allocation per person.")
                personnel_grid.addWidget(app_spin, 2, 1)
                self.inputs_widgets_map[el_name]['app_spin'] = app_spin

                self.grid_layout.addWidget(personnel_group_box, current_row_idx, 1, Qt.AlignTop)
            else:
                # Placeholder for non-personnel elements to maintain grid alignment
                placeholder_label = QLabel("- Not Applicable -")
                placeholder_label.setAlignment(Qt.AlignCenter)
                placeholder_label.setStyleSheet(f"color: {DARK_THEME_INFO_TEXT_COLOR.name()}; font-style: italic;")
                self.grid_layout.addWidget(placeholder_label, current_row_idx, 1, Qt.AlignTop | Qt.AlignCenter)

            # Total Area SpinBox (always present)
            total_area_spin = QDoubleSpinBox()
            total_area_spin.setRange(0.1, 50000.0); total_area_spin.setDecimals(1); total_area_spin.setSingleStep(1.0)
            total_area_spin.setSuffix(" m²")
            total_area_spin.setButtonSymbols(QAbstractSpinBox.PlusMinus) # Cleaner look for buttons
            total_area_spin.setToolTip(f"Total required area for {el_name}. "
                                      f"Can be auto-calculated or set manually if applicable.")
            self.grid_layout.addWidget(total_area_spin, current_row_idx, 2, Qt.AlignTop | Qt.AlignCenter)
            self.inputs_widgets_map[el_name]['total_area_spin'] = total_area_spin

            # Suggested Dimensions Label (always present)
            suggestion_lbl = QLabel("N/A")
            suggestion_lbl.setToolTip("Approximate Width × Height based on total area and a typical aspect ratio. "
                                      "Fixed-size elements will show their defined dimensions.")
            suggestion_lbl.setStyleSheet(f"font-style: italic; color: {DARK_THEME_INFO_TEXT_COLOR.name()};")
            suggestion_lbl.setMinimumWidth(100) # Ensure space for text like "12.3m × 45.6m"
            suggestion_lbl.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(suggestion_lbl, current_row_idx, 3, Qt.AlignTop | Qt.AlignCenter)
            self.inputs_widgets_map[el_name]['suggestion_lbl'] = suggestion_lbl

            current_row_idx += 1

        # Add vertical stretch to push grid items to the top if content is short
        self.grid_layout.setRowStretch(current_row_idx, 1)

        scroll_area.setWidget(scroll_content_widget)
        main_layout.addWidget(scroll_area)

        # --- Reset Button ---
        reset_button_layout = QHBoxLayout()
        reset_button_layout.addStretch(1) # Push button to the right
        reset_btn = QPushButton(QIcon.fromTheme("edit-clear", QIcon()), "Reset All to Defaults")
        reset_btn.setToolTip("Resets all space requirements and personnel configurations on this page to their initial default values.")
        reset_btn.clicked.connect(self._confirm_and_reset_to_defaults)
        reset_button_layout.addWidget(reset_btn)
        main_layout.addLayout(reset_button_layout)

        # --- Connect Signals after UI is built ---
        self._connect_element_signals()

    def _get_header_tooltip(self, col_index: int) -> str: # Retained for reference, but tooltips now inline
        tooltips = [
            "Name of the site facility.",
            "For elements like offices or dormitories, define based on personnel numbers or set total area manually.",
            "Total required floor area for the facility in square meters.",
            "A suggested Width × Height based on the total area and a typical aspect ratio for this facility type."
        ]
        return tooltips[col_index] if 0 <= col_index < len(tooltips) else ""

    def _connect_element_signals(self):
        """Connects signals for the dynamically created input widgets for each element."""
        for el_name, widgets_dict in self.inputs_widgets_map.items():
            if 'override_cb' in widgets_dict: # For personnel-controlled elements
                widgets_dict['override_cb'].toggled.connect(
                    lambda checked, name=el_name: self._handle_override_toggle(name, checked)
                )
            if 'p_count_spin' in widgets_dict:
                widgets_dict['p_count_spin'].valueChanged.connect(
                    lambda value, name=el_name: self._personnel_inputs_changed(name)
                )
            if 'app_spin' in widgets_dict:
                widgets_dict['app_spin'].valueChanged.connect(
                    lambda value, name=el_name: self._personnel_inputs_changed(name)
                )

            # Connect total_area_spin valueChanged for all elements
            widgets_dict['total_area_spin'].valueChanged.connect(
                lambda value, name=el_name: self._total_area_manually_changed(name)
            )

    def _populate_ui_from_current_state(self):
        """
        Populates all UI fields based on the current internal state:
        `self.current_requirements` and `self.current_personnel_config_data`.
        This method is called during initialization and after loading data.
        """
        self._is_programmatic_update = True # Prevent signals during this population
        logger.debug("Populating SpaceRequirementsWidget UI from current state.")

        for el_name, widgets_dict in self.inputs_widgets_map.items():
            current_total_area_for_element = self.current_requirements.get(el_name, 0.0)
            element_master_config = self.SITE_ELEMENTS_CONFIG.get(el_name, {})
            is_fixed_size = not element_master_config.get('resizable', True)

            if el_name in self.personnel_controlled_elements_names:
                personnel_cfg = self.current_personnel_config_data.get(el_name,
                                                                      {'count': 0, 'area_per_person': 0.0, 'override': False})
                widgets_dict['p_count_spin'].setValue(personnel_cfg.get('count', 0))
                widgets_dict['app_spin'].setValue(personnel_cfg.get('area_per_person', 0.0))

                # Determine if "Set Manually" (override) should be checked
                # Override is true if 'override' flag is set in personnel_cfg,
                # OR if the current_total_area significantly differs from personnel calculation (and not fixed size).
                is_overridden_by_flag = personnel_cfg.get('override', False)
                calculated_from_personnel = personnel_cfg.get('count', 0) * personnel_cfg.get('area_per_person', 0.0)

                should_be_overridden = is_overridden_by_flag
                if not is_fixed_size and not is_overridden_by_flag: # Only check area diff if not fixed and not flagged
                    if calculated_from_personnel <= 1e-3 and current_total_area_for_element > 1e-3:
                        should_be_overridden = True # Manual area set when personnel calc is zero
                    elif abs(current_total_area_for_element - calculated_from_personnel) > 1e-2 and current_total_area_for_element > 1e-3:
                        should_be_overridden = True # Significant difference

                widgets_dict['override_cb'].setChecked(should_be_overridden)
                self._enable_personnel_inputs(el_name, not should_be_overridden) # Update enabled state

                if not should_be_overridden and calculated_from_personnel > 1e-3:
                    # If not overridden and personnel calc is valid, it dictates the area
                    current_total_area_for_element = calculated_from_personnel
                    # This ensures self.current_requirements is consistent if it was loaded differently
                    self.current_requirements[el_name] = current_total_area_for_element

            # Set the total area spinbox value
            widgets_dict['total_area_spin'].setValue(current_total_area_for_element)
            # Fixed-size elements' total_area_spin should be read-only
            if is_fixed_size:
                widgets_dict['total_area_spin'].setReadOnly(True)
                fixed_w = element_master_config.get('fixed_width', 0)
                fixed_h = element_master_config.get('fixed_height', 0)
                fixed_area = fixed_w * fixed_h
                if abs(current_total_area_for_element - fixed_area) > 1e-3:
                    logger.warning(f"For fixed element {el_name}, current area {current_total_area_for_element} "
                                   f"differs from fixed {fixed_area}. Setting to fixed area.")
                    self.current_requirements[el_name] = fixed_area
                    widgets_dict['total_area_spin'].setValue(fixed_area)


            self._update_suggestion_label(el_name, current_total_area_for_element)

        self._is_programmatic_update = False
        # Do not emit requirements_changed here, as this is for initial population or reset.
        # Changes are emitted through user interaction or explicit data setting methods.

    def _confirm_and_reset_to_defaults(self):
        reply = QMessageBox.question(self, "Confirm Reset",
                                     "Are you sure you want to reset all space requirements and personnel settings to their original defaults for this application session?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.reset_to_application_defaults()


    def reset_to_application_defaults(self):
        """Resets internal state and UI to the application's global default configurations."""
        logger.info("SpaceRequirementsWidget: Resetting to application defaults.")
        # Re-initialize from the global constants defined at the top of the file
        self.current_personnel_config_data = {
            name: data.copy() for name, data in self.DEFAULT_PERSONNEL_CONFIG.items()
        }
        self.current_requirements = self.DEFAULT_SPACE_REQUIREMENTS.copy()

        self._populate_ui_from_current_state() # Repopulate UI with these new defaults

        # Emit signals to notify the rest of the application about the reset
        if not self._is_programmatic_update:
            self.requirements_changed.emit(self.current_requirements.copy())
            self.personnel_config_changed.emit(self.current_personnel_config_data.copy())
        logger.info("Space requirements reset to application defaults and signals emitted.")


    def set_requirements_data(self, new_requirements_data: Dict[str, float],
                              new_personnel_config_data: Optional[Dict[str, Dict]] = None):
        """
        Programmatically sets the requirements and personnel config, then updates the UI.
        This is typically called when loading a project.
        """
        logger.info("SpaceRequirementsWidget: Setting data programmatically (e.g., from loaded project).")
        self.current_requirements = new_requirements_data.copy()
        if new_personnel_config_data:
            self.current_personnel_config_data = {
                name: data.copy() for name, data in new_personnel_config_data.items()
            }
        else: # If no personnel config provided, reset to defaults before applying new requirements
             self.current_personnel_config_data = {
                name: data.copy() for name, data in self.DEFAULT_PERSONNEL_CONFIG.items()
            }


        self._populate_ui_from_current_state() # This will sync UI with the new data

        # After programmatic setting, emit the changes so the rest of app is aware
        if not self._is_programmatic_update: # Should already be false, but as a safeguard
            self.requirements_changed.emit(self.current_requirements.copy())
            self.personnel_config_changed.emit(self.current_personnel_config_data.copy())
        logger.info("Space requirements data set programmatically and signals emitted.")


    def _enable_personnel_inputs(self, el_name: str, enabled: bool):
        """Helper to manage enabled state of personnel input fields based on override checkbox."""
        if el_name in self.inputs_widgets_map and el_name in self.personnel_controlled_elements_names:
            widgets = self.inputs_widgets_map[el_name]
            widgets['p_count_spin'].setEnabled(enabled)
            widgets['app_spin'].setEnabled(enabled)
            # Total area spinbox is editable if override is checked (i.e., personnel inputs disabled),
            # UNLESS the element itself is fixed-size.
            element_master_config = self.SITE_ELEMENTS_CONFIG.get(el_name, {})
            is_fixed_size = not element_master_config.get('resizable', True)
            if is_fixed_size:
                widgets['total_area_spin'].setReadOnly(True)
            else:
                widgets['total_area_spin'].setReadOnly(enabled) # ReadOnly if personnel inputs are active

    def _handle_override_toggle(self, el_name: str, is_override_checked: bool):
        """Handles toggling of the 'Set Manually' (override) checkbox."""
        if self._is_programmatic_update: return
        logger.debug(f"Override toggle for {el_name}: {'Checked (Manual)' if is_override_checked else 'Unchecked (Personnel-based)'}")

        self._enable_personnel_inputs(el_name, not is_override_checked)

        # Update the override flag in internal personnel config
        if el_name in self.current_personnel_config_data:
            self.current_personnel_config_data[el_name]['override'] = is_override_checked

        if not is_override_checked: # Switched TO personnel-based calculation
            self._personnel_inputs_changed(el_name) # Recalculate and update total area
        else: # Switched TO manual total area input
            # No immediate change to total area value needed here, user will edit the spinbox.
            # Just ensure the current value in spinbox is reflected internally if it's different.
            self._total_area_manually_changed(el_name)

        # Emit personnel config change as the override flag has changed
        self.personnel_config_changed.emit(self.current_personnel_config_data.copy())


    def _personnel_inputs_changed(self, el_name: str):
        """Handles changes in personnel count or area/person spinboxes."""
        if self._is_programmatic_update: return
        if el_name not in self.personnel_controlled_elements_names: return

        widgets = self.inputs_widgets_map[el_name]
        is_overridden = widgets['override_cb'].isChecked()

        count = widgets['p_count_spin'].value()
        area_pp = widgets['app_spin'].value()

        # Always update internal personnel_config_data for later use, even if overridden
        if el_name in self.current_personnel_config_data:
            self.current_personnel_config_data[el_name]['count'] = count
            self.current_personnel_config_data[el_name]['area_per_person'] = area_pp
            self.personnel_config_changed.emit(self.current_personnel_config_data.copy())


        if is_overridden:
            # If overridden, personnel inputs don't change total area.
            # User must uncheck override or edit total area directly.
            logger.debug(f"Personnel input changed for {el_name}, but it's overridden. No area update.")
            return

        calculated_total_area = count * area_pp
        logger.debug(f"Personnel input changed for {el_name} (not overridden). New calculated area: {calculated_total_area}")


        self._is_programmatic_update = True # Prevent total_area_spin's valueChanged from re-triggering
        widgets['total_area_spin'].setValue(calculated_total_area)
        self._is_programmatic_update = False

        self.current_requirements[el_name] = calculated_total_area
        self._update_suggestion_label(el_name, calculated_total_area)
        self.requirements_changed.emit(self.current_requirements.copy())


    def _total_area_manually_changed(self, el_name: str):
        """Handles changes in the total area spinbox (manual input)."""
        if self._is_programmatic_update: return

        widgets = self.inputs_widgets_map[el_name]
        new_total_area = widgets['total_area_spin'].value()

        # For personnel-controlled elements, this manual change implies override.
        if el_name in self.personnel_controlled_elements_names:
            if not widgets['override_cb'].isChecked():
                # This case should ideally not happen if ReadOnly state is managed correctly.
                # If total_area_spin is editable, override_cb should be checked.
                # However, if it does, respect the manual change.
                logger.warning(f"Total area for {el_name} changed manually while override was not checked. Forcing override.")
                self._is_programmatic_update = True
                widgets['override_cb'].setChecked(True) # Programmatically check override
                self._enable_personnel_inputs(el_name, False) # Disable personnel fields
                if el_name in self.current_personnel_config_data: # Update internal override flag
                    self.current_personnel_config_data[el_name]['override'] = True
                    self.personnel_config_changed.emit(self.current_personnel_config_data.copy())
                self._is_programmatic_update = False


        self.current_requirements[el_name] = new_total_area
        self._update_suggestion_label(el_name, new_total_area)
        self.requirements_changed.emit(self.current_requirements.copy())
        logger.debug(f"Total area for {el_name} manually changed to: {new_total_area}")


    def _update_suggestion_label(self, el_name: str, current_total_area: float):
        """Updates the 'Suggested Dimensions' label for an element based on its area."""
        suggestion_lbl = self.inputs_widgets_map[el_name].get('suggestion_lbl')
        if not suggestion_lbl: return

        if current_total_area <= 1e-6: # Effectively zero area
            suggestion_lbl.setText("N/A (Zero Area)")
            return

        element_master_config = self.SITE_ELEMENTS_CONFIG.get(el_name, {})
        is_resizable = element_master_config.get('resizable', True)
        fixed_w = element_master_config.get('fixed_width')
        fixed_h = element_master_config.get('fixed_height')

        if not is_resizable and fixed_w is not None and fixed_h is not None:
            # For fixed-size elements, display their fixed dimensions
            suggestion_lbl.setText(f"Fixed: {fixed_w:.1f} × {fixed_h:.1f}")
            # Also ensure the area matches the fixed dimensions internally.
            # This might have been handled by _populate_ui_from_current_state but good to re-check.
            fixed_area = fixed_w * fixed_h
            if abs(self.current_requirements.get(el_name, 0.0) - fixed_area) > 1e-3:
                logger.debug(f"Updating area for fixed {el_name} to {fixed_area} from suggestion label update.")
                self.current_requirements[el_name] = fixed_area
                self._is_programmatic_update = True
                self.inputs_widgets_map[el_name]['total_area_spin'].setValue(fixed_area)
                self._is_programmatic_update = False
                # No need to emit requirements_changed here, as this is a corrective measure within an update flow.
        elif is_resizable:
            aspect_ratio = element_master_config.get('aspect_ratio', 1.0) # Default to square if no aspect ratio
            if aspect_ratio <= 0: aspect_ratio = 1.0 # Prevent division by zero or negative

            # Calculate width and height from area and aspect ratio:
            # area = width * height; width / height = aspect_ratio => height = width / aspect_ratio
            # area = width * (width / aspect_ratio) = width^2 / aspect_ratio
            # width^2 = area * aspect_ratio
            width = math.sqrt(current_total_area * aspect_ratio)
            height = current_total_area / width if width > 1e-6 else 0.0
            suggestion_lbl.setText(f"~ {width:.1f} × {height:.1f}")
        else:
            # Should not happen if logic is correct (either fixed or resizable with aspect ratio)
            suggestion_lbl.setText("Config Error")


    def get_current_requirements(self) -> Dict[str, float]:
        """
        Returns a copy of the current space requirements dictionary, ensuring it's
        consistent with the UI state by re-reading from spinboxes if necessary.
        This method primarily returns the internal cache which should be kept in sync.
        """
        # The internal self.current_requirements should be the source of truth,
        # updated by the various handler methods.
        # For robustness, one could re-read all total_area_spin values here,
        # but it's generally better to ensure handlers keep self.current_requirements correct.
        return self.current_requirements.copy()

    def get_current_personnel_config(self) -> Dict[str, Dict]:
        """
        Returns a copy of the current personnel configuration data,
        which includes counts, area/person, and override status.
        """
        # Similar to get_current_requirements, relies on handlers keeping internal state correct.
        return {name: data.copy() for name, data in self.current_personnel_config_data.items()}

# --- ConstraintsWidget ---
class ConstraintsWidget(QWidget):
    """
    Widget for displaying and adjusting various constraint parameters that
    influence the layout optimization process, such as minimum distances,
    crane reach, and safety buffers.
    """
    # Emits the full dictionary of {constraint_key: value} when any constraint changes.
    constraints_changed = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Initialize internal constraints dictionary with a copy of global defaults.
        # This dictionary will be the source of truth for current constraint values.
        self.current_constraints: Dict[str, Union[float, bool]] = DEFAULT_CONSTRAINTS.copy()

        # Map to store references to the dynamically created QDoubleSpinBox or QCheckBox widgets.
        # Structure: {constraint_key: QWidget_instance}
        self.input_widgets_map: Dict[str, QWidget] = {}
        self._is_programmatic_update: bool = False # Flag to prevent signal loops

        self.initUI()
        self._populate_ui_from_current_state() # Populate UI based on initial state
        logger.debug("ConstraintsWidget initialized.")

    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5,5,5,5)

        title_label = QLabel("Layout Constraints Configuration")
        font = title_label.font(); font.setPointSize(13); font.setBold(True)
        title_label.setFont(font)
        main_layout.addWidget(title_label)

        info_label = QLabel(
            "Adjust parameters that define relationships and restrictions for the site layout. "
            "These include safety distances, operational limits (e.g., crane reach), and general spacing."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"font-size: 9pt; color: {DARK_THEME_INFO_TEXT_COLOR.name()};")
        main_layout.addWidget(info_label)

        # Scrollable area for the list of constraints
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.StyledPanel)
        scroll_area.setFrameShadow(QFrame.Sunken)

        constraint_widget_content = QWidget() # Content widget for the scroll area
        grid_layout = QGridLayout(constraint_widget_content)
        grid_layout.setColumnStretch(1, 1) # Allow input widgets to expand
        grid_layout.setColumnStretch(2, 2) # Allow description label to expand
        grid_layout.setVerticalSpacing(8)
        grid_layout.setHorizontalSpacing(10)


        current_row_idx = 0
        # Sort constraints by key for consistent display order
        for key, value in sorted(self.current_constraints.items()):
            # Create a user-friendly label from the constraint key
            label_text = key.replace('_', ' ').replace('min', 'Min.').replace('max', 'Max.')
            label_text = label_text.title() # Capitalize words

            unit_suffix = " (m)" if isinstance(value, (float, int)) else ""
            if "ratio" in key.lower() or "factor" in key.lower() : unit_suffix = "" # No unit for ratios/factors
            if isinstance(value, bool): unit_suffix = ""


            grid_layout.addWidget(QLabel(f"{label_text}{unit_suffix}:"), current_row_idx, 0, Qt.AlignRight | Qt.AlignTop)

            input_widget: QWidget
            if isinstance(value, bool):
                checkbox = QCheckBox()
                checkbox.setChecked(value)
                checkbox.toggled.connect(self._emit_constraint_change)
                input_widget = checkbox
                grid_layout.addWidget(input_widget, current_row_idx, 1, Qt.AlignLeft | Qt.AlignTop)
            elif isinstance(value, (float, int)):
                spinbox = QDoubleSpinBox()
                # Determine appropriate range and step based on key or value characteristics
                if "reach" in key or "distance" in key:
                    spinbox.setRange(0.0, 500.0); spinbox.setSingleStep(1.0); spinbox.setDecimals(1)
                elif "buffer" in key or "width" in key:
                    spinbox.setRange(0.1, 50.0); spinbox.setSingleStep(0.1); spinbox.setDecimals(1)
                elif "ratio" in key or "factor" in key:
                    spinbox.setRange(0.0, 10.0); spinbox.setSingleStep(0.05); spinbox.setDecimals(2)
                else: # Generic float/int
                    spinbox.setRange(-1000.0, 1000.0); spinbox.setSingleStep(0.5); spinbox.setDecimals(1)

                spinbox.setValue(float(value))
                spinbox.valueChanged.connect(self._emit_constraint_change)
                input_widget = spinbox
                grid_layout.addWidget(input_widget, current_row_idx, 1, Qt.AlignLeft | Qt.AlignTop)
            else:
                # Should not happen if DEFAULT_CONSTRAINTS is well-defined
                logger.error(f"Unsupported constraint type for key '{key}': {type(value)}")
                input_widget = QLabel(f"Error: Unsupported type {type(value)}")
                grid_layout.addWidget(input_widget, current_row_idx, 1, Qt.AlignLeft | Qt.AlignTop)


            input_widget.setObjectName(key) # Store the constraint key in the widget for easy retrieval
            self.input_widgets_map[key] = input_widget

            # Add description label for the constraint
            description_text = self._get_constraint_description(key)
            description_label = QLabel(description_text)
            description_label.setWordWrap(True)
            description_label.setStyleSheet(f"color: {DARK_THEME_INFO_TEXT_COLOR.name()}; font-size: 8pt; padding-left: 5px;")
            grid_layout.addWidget(description_label, current_row_idx, 2, Qt.AlignLeft | Qt.AlignTop)

            current_row_idx += 1
        
        # Add vertical stretch to push grid items to the top
        grid_layout.setRowStretch(current_row_idx, 1)

        scroll_area.setWidget(constraint_widget_content)
        main_layout.addWidget(scroll_area)

        # Reset Button
        reset_button_layout = QHBoxLayout()
        reset_button_layout.addStretch(1) # Push button to the right
        reset_btn = QPushButton(QIcon.fromTheme("edit-clear", QIcon()), "Reset All to Defaults")
        reset_btn.setToolTip("Resets all constraint parameters on this page to their initial default values for this application session.")
        reset_btn.clicked.connect(self._confirm_and_reset_to_defaults)
        reset_button_layout.addWidget(reset_btn)
        main_layout.addLayout(reset_button_layout)

    def _get_constraint_description(self, key: str) -> str:
        """Returns a user-friendly description for a given constraint key."""
        descriptions = {
            'min_distance_fuel_dormitory': "Minimum safe distance between Fuel Tanks and Workers' Dormitories.",
            'min_distance_fuel_welding': "Minimum safe distance between Fuel Tanks and Welding Workshops due to fire hazard.",
            'max_distance_storage_welding': "Maximum efficient travel distance between Material Storage areas and Welding Workshops.",
            'crane_reach': "Operational reach (radius) of the primary Tower Crane(s) from its center.",
            'min_distance_offices_welding': "Minimum distance for comfort/safety between Office facilities and noisy/hazardous Welding Workshops.",
            'min_distance_offices_machinery': "Minimum distance for comfort/safety between Office facilities and Machinery Parking areas (noise/fumes).",
            'min_path_width': "Minimum clear width required for primary pedestrian and vehicle circulation paths.",
            'safety_buffer': "General minimum buffer space to be maintained between the edges of any two distinct site elements.",
            'building_safety_buffer': "Specific minimum buffer space around the main Building structure. Overrides general safety_buffer if 'buffer_building_override' is true.",
            'buffer_building_override': "If checked, use 'building_safety_buffer' for the main Building instead of the general 'safety_buffer'."
            # Add more descriptions as new constraints are introduced
        }
        return descriptions.get(key, "General constraint parameter influencing layout optimization.")

    def _populate_ui_from_current_state(self):
        """Populates all UI input fields based on the `self.current_constraints` dictionary."""
        self._is_programmatic_update = True # Prevent signals during this population
        logger.debug("Populating ConstraintsWidget UI from current state.")

        for key, widget in self.input_widgets_map.items():
            value = self.current_constraints.get(key)
            if value is None:
                logger.warning(f"Constraint key '{key}' found in UI map but not in current_constraints data. Skipping.")
                continue

            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            else:
                logger.warning(f"Widget for key '{key}' is of an unhandled type: {type(widget)}")
        self._is_programmatic_update = False

    def _emit_constraint_change(self):
        """
        Called when any input widget's value changes. Updates the internal
        `current_constraints` dictionary and emits the `constraints_changed` signal.
        """
        if self._is_programmatic_update:
            return

        sender_widget = self.sender()
        if not sender_widget:
            logger.warning("Constraint change signal received with no sender. Ignoring.")
            return

        key = sender_widget.objectName() # Retrieve constraint key from widget's objectName
        if key not in self.current_constraints:
            logger.error(f"Constraint key '{key}' from sender widget not found in internal constraints. This is a bug.")
            return

        new_value: Union[float, bool]
        if isinstance(sender_widget, QDoubleSpinBox):
            new_value = sender_widget.value()
        elif isinstance(sender_widget, QCheckBox):
            new_value = sender_widget.isChecked()
        else:
            logger.warning(f"Unhandled sender widget type for constraint change: {type(sender_widget)}")
            return

        if self.current_constraints[key] != new_value:
            self.current_constraints[key] = new_value
            logger.debug(f"Constraint '{key}' changed to: {new_value}")
            self.constraints_changed.emit(self.current_constraints.copy()) # Emit a copy
        else:
            logger.debug(f"Constraint '{key}' value emitted, but no change from current value ({new_value}).")


    def _confirm_and_reset_to_defaults(self):
        reply = QMessageBox.question(self, "Confirm Reset",
                                     "Are you sure you want to reset all layout constraints to their original default values for this application session?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.reset_to_application_defaults()


    def reset_to_application_defaults(self):
        """Resets all constraint values to their global `DEFAULT_CONSTRAINTS`."""
        logger.info("ConstraintsWidget: Resetting to application defaults.")
        self.current_constraints = DEFAULT_CONSTRAINTS.copy() # Re-initialize from global defaults
        self._populate_ui_from_current_state() # Update all UI widgets

        # Emit the change after resetting
        if not self._is_programmatic_update: # Should be false here
            self.constraints_changed.emit(self.current_constraints.copy())
        logger.info("Constraints reset to application defaults and signal emitted.")

    def set_constraints_data(self, new_constraints_data: Dict[str, Union[float, bool]]):
        """
        Programmatically sets the constraint values from an external dictionary
        (e.g., when loading a project) and updates the UI.
        """
        logger.info("ConstraintsWidget: Setting data programmatically (e.g., from loaded project).")
        # Ensure all keys from new_constraints_data are valid and merge with defaults for missing keys
        # This handles cases where a loaded project might be missing some newer constraints.
        temp_constraints = DEFAULT_CONSTRAINTS.copy()
        temp_constraints.update(new_constraints_data) # Override defaults with loaded values
        self.current_constraints = temp_constraints

        self._populate_ui_from_current_state() # Update UI to reflect these new values

        # After programmatic setting, emit the changes so the rest of app is aware
        if not self._is_programmatic_update: # Should be false
            self.constraints_changed.emit(self.current_constraints.copy())
        logger.info("Constraints data set programmatically and signal emitted.")


    def get_current_constraints(self) -> Dict[str, Union[float, bool]]:
        """
        Returns a copy of the current constraints dictionary.
        The internal `current_constraints` is kept in sync by UI interaction handlers.
        """
        return self.current_constraints.copy()

# --- OptimizationControlWidget ---

# Define these constants at the module level or as class attributes if preferred
SA_MIN_TEMPERATURE = 0.1  # Default min temperature for Simulated Annealing
SA_REHEAT_FACTOR = 1.5    # Default reheat factor
SA_REHEAT_THRESHOLD_TEMP = 10.0 # Default reheat threshold

class OptimizationControlWidget(QWidget):
    """Widget for controlling the optimization process and approach"""

    # Signal: iterations, initial_temp, cooling_rate, min_temp, reheat_factor, reheat_threshold_temp
    start_optimization = pyqtSignal(int, float, float, float, float, float)
    # Signal to request the current optimization weights from the main application or engine
    request_weights_update = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        # optimization_weights will be set by update_optimization_weights based on user selection
        self.optimization_weights: Dict[str, float] = {}
        self.initUI()
        # Initialize weights based on the default selected radio button (Balanced)
        default_approach_id = self.approach_buttons.checkedId()
        self.update_optimization_weights(default_approach_id)
        logger.debug("OptimizationControlWidget initialized.")


    def initUI(self):
        layout = QVBoxLayout(self) # Use self for parent layout
        layout.setSpacing(10) # Consistent spacing

        title = QLabel("Optimization Controls")
        font = title.font()
        font.setPointSize(13) # Slightly larger title
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        info_label = QLabel(
            "Configure the Simulated Annealing algorithm and select an optimization "
            "strategy to guide the layout process towards desired objectives."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"font-size: 9pt; color: {DARK_THEME_INFO_TEXT_COLOR.name()};") # Softer color
        layout.addWidget(info_label)

        # Parameters Grid GroupBox
        params_group = QGroupBox("Algorithm Parameters (Simulated Annealing)")
        grid = QGridLayout(params_group) # Set layout for the group box
        grid.setColumnStretch(1, 1) # Allow spinboxes to expand
        grid.setSpacing(8) # Internal spacing for grid

        row_idx = 0
        grid.addWidget(QLabel("Iterations:"), row_idx, 0)
        self.iterations_input = QSpinBox()
        self.iterations_input.setRange(100, 50000) # Wider range
        self.iterations_input.setValue(5000)       # Increased default
        self.iterations_input.setSingleStep(500)
        self.iterations_input.setToolTip("Total number of steps the optimization algorithm will perform. "
                                         "More iterations can lead to better solutions but increase computation time.")
        grid.addWidget(self.iterations_input, row_idx, 1)
        row_idx += 1

        grid.addWidget(QLabel("Start Temperature (T₀):"), row_idx, 0)
        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(1.0, 2000.0) # Wider range for T0
        self.temperature_input.setValue(200.0)      # Higher default T0
        self.temperature_input.setDecimals(1)
        self.temperature_input.setSingleStep(10.0)
        self.temperature_input.setToolTip("Initial 'energy' of the system. Higher values allow the algorithm "
                                          "to accept worse solutions more readily, exploring more of the solution space.")
        grid.addWidget(self.temperature_input, row_idx, 1)
        row_idx += 1

        grid.addWidget(QLabel("Cooling Rate (α):"), row_idx, 0)
        self.cooling_input = QDoubleSpinBox()
        self.cooling_input.setRange(0.800, 0.999)
        self.cooling_input.setValue(0.985) # Slower cooling by default
        self.cooling_input.setDecimals(3)
        self.cooling_input.setSingleStep(0.001)
        self.cooling_input.setToolTip("Multiplier for temperature reduction at each step (e.g., 0.99 means T_new = T_old * 0.99). "
                                      "Values closer to 1.0 result in slower cooling.")
        grid.addWidget(self.cooling_input, row_idx, 1)
        row_idx += 1

        grid.addWidget(QLabel("Min Temperature (T_min):"), row_idx, 0)
        self.min_temp_input = QDoubleSpinBox()
        self.min_temp_input.setRange(0.001, 100.0) # Allow very low min temp
        self.min_temp_input.setValue(SA_MIN_TEMPERATURE) # Use global constant
        self.min_temp_input.setDecimals(3)
        self.min_temp_input.setSingleStep(0.01)
        self.min_temp_input.setToolTip("The lowest temperature the system can reach. "
                                       "Prevents temperature from becoming zero and effectively stops accepting worse solutions.")
        grid.addWidget(self.min_temp_input, row_idx, 1)
        row_idx += 1

        grid.addWidget(QLabel("Reheat Factor:"), row_idx, 0)
        self.reheat_factor_input = QDoubleSpinBox()
        self.reheat_factor_input.setRange(1.0, 10.0) # Wider range for reheating
        self.reheat_factor_input.setValue(SA_REHEAT_FACTOR) # Use global constant
        self.reheat_factor_input.setDecimals(1)
        self.reheat_factor_input.setSingleStep(0.1)
        self.reheat_factor_input.setToolTip("If the algorithm gets stuck (no improvement), temperature can be multiplied by this factor "
                                            "to 'reheat' the system and allow more exploration.")
        grid.addWidget(self.reheat_factor_input, row_idx, 1)
        row_idx += 1

        grid.addWidget(QLabel("Reheat Threshold Temp:"), row_idx, 0)
        self.reheat_threshold_input = QDoubleSpinBox()
        self.reheat_threshold_input.setRange(0.1, 200.0)
        self.reheat_threshold_input.setValue(SA_REHEAT_THRESHOLD_TEMP) # Use global constant
        self.reheat_threshold_input.setDecimals(1)
        self.reheat_threshold_input.setSingleStep(1.0)
        self.reheat_threshold_input.setToolTip("If temperature drops below this value AND the solution hasn't improved for a while, "
                                               "reheating may be triggered.")
        grid.addWidget(self.reheat_threshold_input, row_idx, 1)
        # Removed params_group.setLayout(grid) as it's done by QGroupBox constructor

        layout.addWidget(params_group)

        # Optimization Approach Selection GroupBox
        approach_group = QGroupBox("Optimization Strategy")
        approach_layout = QVBoxLayout(approach_group) # Set layout for this group box
        self.approach_buttons = QButtonGroup(self)

        # Define approaches: (Display Text, Detailed Tooltip, Weight Profile ID/Name)
        approaches = [
            ("Balanced (Default)", "Aims for a good overall compromise between safety, operational efficiency, and site comfort.", "balanced"),
            ("Safety Priority", "Strongly prioritizes adherence to safety distances, minimizing overlaps, and reducing hazard proximities.", "safety"),
            ("Efficiency Priority", "Focuses on minimizing material travel distances, ensuring good crane coverage, and optimizing accessibility for key facilities.", "efficiency"),
            ("Comfort & Spacing", "Emphasizes comfortable placement of offices/dorms away from noise/hazards, and promotes well-spaced, organized layouts.", "comfort")
        ]

        for i, (text, tooltip, profile_id) in enumerate(approaches):
            radio_button = QRadioButton(text)
            radio_button.setToolTip(tooltip)
            if i == 0: # Select "Balanced" by default
                radio_button.setChecked(True)
            # Store profile_id with the button if needed for more complex weight management,
            # but here we use index `i` for simplicity with `update_optimization_weights`.
            self.approach_buttons.addButton(radio_button, i)
            approach_layout.addWidget(radio_button)

        self.approach_buttons.idClicked.connect(self.update_optimization_weights) # Use idClicked for the ID
        # Removed approach_group.setLayout(approach_layout)
        layout.addWidget(approach_group)

        # Progress and Status Section
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True) # Show percentage on bar
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Strategy: Balanced (Default)") # Initial status reflecting default
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"font-size: 9pt; padding: 3px; color: {DARK_THEME_INFO_TEXT_COLOR.name()};")
        layout.addWidget(self.status_label)

        # Optimize Button
        self.optimize_button = QPushButton(QIcon(), "Run Optimization") # Placeholder for icon
        self.optimize_button.clicked.connect(self.emit_start_optimization_signal)
        self.optimize_button.setStyleSheet(
            # "QPushButton { background-color: #28a745; color: white; padding: 8px; font-size: 10pt; border-radius: 4px; }"
            # "QPushButton:hover { background-color: #218838; }"
            # "QPushButton:pressed { background-color: #1e7e34; }"
            # "QPushButton:disabled { background-color: #e9ecef; color: #6c757d; }"
            f"QPushButton {{ background-color: {DARK_THEME_SUCCESS_BUTTON_BG_COLOR.name()}; color: white; padding: 8px; font-size: 10pt; border-radius: 4px; }}"
            f"QPushButton:hover {{ background-color: {DARK_THEME_SUCCESS_BUTTON_HOVER_BG_COLOR.name()}; }}"
            f"QPushButton:pressed {{ background-color: {DARK_THEME_SUCCESS_BUTTON_PRESSED_BG_COLOR.name()}; }}"
            f"QPushButton:disabled {{ background-color: {QColor(60,60,60).name()}; color: {QColor(127,127,127).name()}; }}"
        )
        self.optimize_button.setMinimumHeight(35) # Make button a bit taller
        layout.addWidget(self.optimize_button)

        layout.addStretch(1) # Push everything to the top
        self.setLayout(layout) # Already done by QVBoxLayout(self)


    def update_optimization_weights(self, approach_id: int):
        """
        Updates the internal self.optimization_weights dictionary based on the selected approach.
        Emits a signal so the main application/optimizer can fetch these weights.
        """
        logger.debug(f"Updating optimization weights for approach ID: {approach_id}")
        # Define base weights for various scoring criteria.
        # These keys MUST match those used in OptimizationEngine.evaluate_layout().
        base_weights = {
            'overlap': 2500.0,
            'out_of_bounds': 2000.0,
            'movable_on_building_penalty': 3000.0, # Very high critical penalty
            'storage_welding_dist_penalty': 3.0,
            'storage_welding_dist_bonus': 2.0,
            'fuel_welding_dist_penalty': 15.0,
            'fuel_welding_safety_bonus': 1.5,
            'fuel_dorm_dist_penalty': 20.0,
            'fuel_dorm_safety_bonus': 2.0,
            'crane_coverage_penalty': 25.0,
            'crane_full_coverage_bonus': 50.0,
            'comfort_safety_penalty': 4.0,
            'comfort_safety_bonus': 0.8,
            'spacing_bonus': 1.0, # Bonus for exceeding safety_buffer
            'spacing_penalty': 2.5, # Penalty for being < safety_buffer (but not overlapping)
            'office_proximity_penalty': 0.5, # Penalty for offices being too far apart
            'office_proximity_bonus': 0.3,   # Bonus for offices being close
            'security_entrance_bonus': 1.2,
            'security_entrance_penalty': 0.6,
            'accessibility_bonus_factor': 1.0, # Multiplier for proximity to entrance for certain elements
            'vehicle_path_length_penalty': 0.20,
            'pedestrian_path_length_penalty': 0.10,
            'route_interference_penalty': 5.0
        }

        self.optimization_weights = base_weights.copy() # Start with base for all approaches

        selected_button = self.approach_buttons.button(approach_id)
        selected_approach_name = selected_button.text() if selected_button else "Unknown"
        self.status_label.setText(f"Strategy: {selected_approach_name}")


        if approach_id == 0: # Balanced (Default)
            # Base weights are considered balanced. No specific overrides here, or minor tweaks.
            pass # self.optimization_weights already holds base_weights

        elif approach_id == 1:  # Safety Priority
            self.optimization_weights.update({
                'overlap': 2500.0, # Significantly higher
                'out_of_bounds': 4000.0, # Significantly higher
                'movable_on_building_penalty': 5000.0, # Max critical
                'fuel_welding_dist_penalty': 30.0,
                'fuel_welding_safety_bonus': 2.5,
                'fuel_dorm_dist_penalty': 40.0,
                'fuel_dorm_safety_bonus': 3.0,
                'comfort_safety_penalty': 8.0, # Higher penalty for being close to hazards
                'comfort_safety_bonus': 1.0,   # Moderate bonus for extra distance
                'spacing_penalty': 5.0,        # Higher penalty for violating buffer
                'route_interference_penalty': 15.0 # High penalty for path crossings
            })

        elif approach_id == 2:  # Efficiency Priority
            self.optimization_weights.update({
                'overlap': 2500.0, # Significantly higher
                'out_of_bounds': 4000.0, # Significantly higher
                'movable_on_building_penalty': 1000.0, # Max critical
                'storage_welding_dist_penalty': 6.0, # Higher penalty if far
                'storage_welding_dist_bonus': 5.0,   # Higher bonus if close
                'crane_full_coverage_bonus': 75.0,   # Stronger emphasis on crane coverage
                'accessibility_bonus_factor': 2.5,   # Stronger pull for storage/parking to entrance
                'vehicle_path_length_penalty': 0.5,  # Penalize long vehicle paths more
                'pedestrian_path_length_penalty': 0.2, # Penalize long pedestrian paths more
                'route_interference_penalty': 3.0    # Lowered if it improves overall flow for efficiency
            })

        elif approach_id == 3: # Comfort & Spacing
            self.optimization_weights.update({
                'overlap': 2500.0, # Significantly higher
                'out_of_bounds': 4000.0, # Significantly higher
                'movable_on_building_penalty': 1000.0, # Max critical
                'comfort_safety_penalty': 2.0, # Lower penalty, focus on bonus
                'comfort_safety_bonus': 2.0,   # Higher bonus for good distance from hazards
                'spacing_bonus': 3.0,          # Strong bonus for good spacing (exceeding buffer)
                'spacing_penalty': 1.5,        # Lower penalty for slightly violating buffer
                'pedestrian_path_length_penalty': 0.05, # Very short pedestrian paths desired
                'route_interference_penalty': 8.0, # Path interference is uncomfortable
                'office_proximity_bonus': 0.8,     # Stronger bonus for offices being grouped
                'office_proximity_penalty': 0.2,   # Lower penalty if they are not
            })

        logger.info(f"Optimization weights updated for approach: {selected_approach_name}. Weights: {self.optimization_weights}")
        self.request_weights_update.emit(self.optimization_weights.copy())


    def emit_start_optimization_signal(self):
        """Gathers parameters and emits the signal to start the optimization process."""
        iterations = self.iterations_input.value()
        temperature = self.temperature_input.value()
        cooling_rate = self.cooling_input.value()
        min_temp = self.min_temp_input.value()
        reheat_factor = self.reheat_factor_input.value()
        reheat_threshold = self.reheat_threshold_input.value()

        # Update status label before emitting signal
        current_approach_text = self.approach_buttons.checkedButton().text() if self.approach_buttons.checkedButton() else "N/A"
        self.status_label.setText(f"Starting optimization with: {current_approach_text}...")
        self.progress_bar.setValue(0) # Reset progress bar

        logger.info(f"Emitting start_optimization signal with params: Iter={iterations}, T0={temperature}, "
                    f"Alpha={cooling_rate}, Tmin={min_temp}, ReheatF={reheat_factor}, ReheatT={reheat_threshold}")
        self.start_optimization.emit(iterations, temperature, cooling_rate, min_temp, reheat_factor, reheat_threshold)

    def get_optimization_weights(self) -> Dict[str, float]:
        """Return the current set of optimization weights based on selected approach."""
        # This method might be redundant if request_weights_update signal is used,
        # but can be useful for direct polling if needed.
        return self.optimization_weights.copy() # Return a copy

    def set_progress(self, value: int):
        """Update the progress bar and status label during optimization."""
        self.progress_bar.setValue(value)
        if 0 <= value < 100:
            current_status = self.status_label.text()
            if "Starting optimization" in current_status or "Optimization in progress..." not in current_status :
                 self.status_label.setText("Optimization in progress...")
        elif value >= 100:
            self.status_label.setText("Optimization complete! Review the layout.")
            # Re-enable controls is typically handled by the main app when the worker thread finishes.

    def enable_controls(self, enabled: bool):
        """Enable or disable all input controls and the start button."""
        self.iterations_input.setEnabled(enabled)
        self.temperature_input.setEnabled(enabled)
        self.cooling_input.setEnabled(enabled)
        self.min_temp_input.setEnabled(enabled)
        self.reheat_factor_input.setEnabled(enabled)
        self.reheat_threshold_input.setEnabled(enabled)
        self.optimize_button.setEnabled(enabled)

        for button in self.approach_buttons.buttons():
            button.setEnabled(enabled)

        if enabled:
            # Reset progress bar and status label text if controls are re-enabled (e.g., after completion)
            # This might be better handled by the main app logic upon completion.
            self.progress_bar.setValue(0)
            current_approach_text = self.approach_buttons.checkedButton().text() if self.approach_buttons.checkedButton() else "N/A"
            self.status_label.setText(f"Strategy: {current_approach_text}")
        else:
            # Status label will be updated by emit_start_optimization_signal or set_progress
            pass
        logger.debug(f"Optimization controls {'enabled' if enabled else 'disabled'}.")

class DisplayOptionsWidget(QWidget):
    """
    Widget for controlling various display options on the SiteCanvas,
    such as the visibility of grid lines, element labels, dimensions,
    crane operational radius, and markers for internal elements.
    """
    # Signal emitting the state of all display options.
    # Order: show_grid, show_dims, show_labels, show_crane_radius, show_internal_markers
    options_changed = pyqtSignal(bool, bool, bool, bool, bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._is_programmatic_update: bool = False # Flag to prevent signals during programmatic changes
        self.initUI()
        logger.debug("DisplayOptionsWidget initialized.")

    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5,5,5,5)

        title_label = QLabel("Canvas Display Options")
        font = title_label.font()
        font.setPointSize(13)
        font.setBold(True)
        title_label.setFont(font)
        main_layout.addWidget(title_label)

        info_label = QLabel(
            "Control the visibility of various informational overlays and visual aids on the site layout canvas."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"font-size: 9pt; color: {DARK_THEME_INFO_TEXT_COLOR.name()};")
        main_layout.addWidget(info_label)

        # GroupBox for better organization
        options_group = QGroupBox("Toggle Visibility")
        group_layout = QVBoxLayout(options_group)
        group_layout.setSpacing(8) # Spacing between checkboxes

        # --- Checkbox for Grid Lines ---
        self.grid_cb = QCheckBox("Show Grid Lines")
        self.grid_cb.setChecked(True) # Default state
        self.grid_cb.setToolTip("Toggle the visibility of the measurement grid on the canvas background.")
        self.grid_cb.toggled.connect(self._emit_options_changed_if_not_programmatic)
        group_layout.addWidget(self.grid_cb)

        # --- Checkbox for Element Dimensions ---
        self.dims_cb = QCheckBox("Show Element Dimensions (W×H)")
        self.dims_cb.setChecked(True) # Default state
        self.dims_cb.setToolTip("Toggle the display of width and height dimensions on each site element.")
        self.dims_cb.toggled.connect(self._emit_options_changed_if_not_programmatic)
        group_layout.addWidget(self.dims_cb)

        # --- Checkbox for Element Labels ---
        self.labels_cb = QCheckBox("Show Element Labels (Names)")
        self.labels_cb.setChecked(True) # Default state
        self.labels_cb.setToolTip("Toggle the display of names on each site element.")
        self.labels_cb.toggled.connect(self._emit_options_changed_if_not_programmatic)
        group_layout.addWidget(self.labels_cb)

        # --- Checkbox for Crane Operational Radius ---
        self.crane_radius_cb = QCheckBox("Show Crane Operational Radius")
        self.crane_radius_cb.setChecked(True) # Default state
        self.crane_radius_cb.setToolTip("Toggle the visibility of the tower crane's reach/operational radius circle.")
        self.crane_radius_cb.toggled.connect(self._emit_options_changed_if_not_programmatic)
        group_layout.addWidget(self.crane_radius_cb)

        # --- Checkbox for Internal Element Markers ---
        self.internal_markers_cb = QCheckBox("Show Internal Element Markers")
        self.internal_markers_cb.setChecked(True) # Default state
        self.internal_markers_cb.setToolTip("Toggle visibility of markers representing elements placed inside the main building.")
        self.internal_markers_cb.toggled.connect(self._emit_options_changed_if_not_programmatic)
        group_layout.addWidget(self.internal_markers_cb)

        # --- Checkbox for Out-of-Bounds Elements (Optional) ---
        # self.oob_elements_cb = QCheckBox("Show Out-of-Bounds Elements")
        # self.oob_elements_cb.setChecked(False) # Default state (usually not shown if fully OOB)
        # self.oob_elements_cb.setToolTip("If checked, elements partially or fully outside plot boundaries might still be rendered (for debugging).")
        # self.oob_elements_cb.toggled.connect(self._emit_options_changed_if_not_programmatic)
        # group_layout.addWidget(self.oob_elements_cb) # Add if this feature is implemented in SiteCanvas

        main_layout.addWidget(options_group)
        main_layout.addStretch(1) # Push options to the top

        # Emit initial state once UI is built, so connected slots can sync
        # This might be better handled by the main app explicitly calling getters after init.
        # For now, let's emit to ensure initial consistency if slots are connected early.
        # QTimer.singleShot(0, self._emit_options_changed_if_not_programmatic) # Emit after event loop starts

    def _emit_options_changed_if_not_programmatic(self):
        """
        Emits the `options_changed` signal with the current states of all checkboxes,
        but only if the change was not triggered programmatically.
        """
        if self._is_programmatic_update:
            return

        grid_visible = self.grid_cb.isChecked()
        dims_visible = self.dims_cb.isChecked()
        labels_visible = self.labels_cb.isChecked()
        crane_radius_visible = self.crane_radius_cb.isChecked()
        internal_markers_visible = self.internal_markers_cb.isChecked()
        # oob_visible = self.oob_elements_cb.isChecked() # If OOB checkbox is added

        logger.debug(f"Display options changed by user: Grid={grid_visible}, Dims={dims_visible}, "
                     f"Labels={labels_visible}, CraneR={crane_radius_visible}, InternalM={internal_markers_visible}")

        self.options_changed.emit(
            grid_visible,
            dims_visible,
            labels_visible,
            crane_radius_visible,
            internal_markers_visible
            # oob_visible # If OOB checkbox is added
        )

    # --- Public methods to get and set states programmatically ---

    def get_all_options_states(self) -> Tuple[bool, bool, bool, bool, bool]:
        """Returns a tuple with the current checked state of all display options."""
        return (
            self.grid_cb.isChecked(),
            self.dims_cb.isChecked(),
            self.labels_cb.isChecked(),
            self.crane_radius_cb.isChecked(),
            self.internal_markers_cb.isChecked()
            # self.oob_elements_cb.isChecked() # If OOB checkbox is added
        )

    def set_all_options_states(self, grid: bool, dims: bool, labels: bool, crane: bool, internal_markers: bool):
        """
        Programmatically sets the state of all display option checkboxes.
        This will prevent the `options_changed` signal from being emitted during the update.
        """
        self._is_programmatic_update = True
        logger.debug(f"Programmatically setting display options: Grid={grid}, Dims={dims}, "
                     f"Labels={labels}, CraneR={crane}, InternalM={internal_markers}")
        self.grid_cb.setChecked(grid)
        self.dims_cb.setChecked(dims)
        self.labels_cb.setChecked(labels)
        self.crane_radius_cb.setChecked(crane)
        self.internal_markers_cb.setChecked(internal_markers)
        # self.oob_elements_cb.setChecked(oob) # If OOB checkbox is added
        self._is_programmatic_update = False
        # After programmatic update, explicitly emit if a change occurred,
        # or let the caller handle propagation if this is part of a larger state load.
        # For simplicity, let's assume the caller might want to react.
        # self._emit_options_changed_if_not_programmatic() # This will now emit as flag is false.


    # Individual setters (useful for menu actions or direct control)
    def set_grid_visible(self, visible: bool):
        self._is_programmatic_update = True
        self.grid_cb.setChecked(visible)
        self._is_programmatic_update = False
        self._emit_options_changed_if_not_programmatic() # Emit if this specific call was the user action via menu

    def set_dims_visible(self, visible: bool):
        self._is_programmatic_update = True
        self.dims_cb.setChecked(visible)
        self._is_programmatic_update = False
        self._emit_options_changed_if_not_programmatic()

    def set_labels_visible(self, visible: bool):
        self._is_programmatic_update = True
        self.labels_cb.setChecked(visible)
        self._is_programmatic_update = False
        self._emit_options_changed_if_not_programmatic()

    def set_crane_radius_visible(self, visible: bool):
        self._is_programmatic_update = True
        self.crane_radius_cb.setChecked(visible)
        self._is_programmatic_update = False
        self._emit_options_changed_if_not_programmatic()

    def set_internal_markers_visible(self, visible: bool):
        self._is_programmatic_update = True
        self.internal_markers_cb.setChecked(visible)
        self._is_programmatic_update = False
        self._emit_options_changed_if_not_programmatic()

    # Add a similar setter for oob_elements_cb if it's implemented
    # def set_oob_elements_visible(self, visible: bool):
    #     self._is_programmatic_update = True
    #     self.oob_elements_cb.setChecked(visible)
    #     self._is_programmatic_update = False
    #     self._emit_options_changed_if_not_programmatic()


class ConstructionSiteOptimizerApp(QMainWindow):
    """
    Main application window for the Construction Site Layout Optimizer.
    It orchestrates the UI, data management, and optimization processes.
    """

    # Signal for when project settings (like plot dimensions) that affect optimizer are changed
    project_settings_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        logger.info("Initializing ConstructionSiteOptimizerApp...")
        self.current_project_filename: Optional[str] = None
        self.unsaved_changes: bool = False

        # Default plot dimensions and project settings (can be loaded or set by user)
        self.plot_width_m: float = 80.0
        self.plot_height_m: float = 50.0
        self.building_ratio_val: float = 0.45 # Proportion of plot area for building

        # These will hold the current state of constraints and requirements,
        # synchronized with the respective widgets or loaded from project files.
        self.current_constraints: Dict = DEFAULT_CONSTRAINTS.copy()
        self.current_space_requirements: Dict = DEFAULT_SPACE_REQUIREMENTS.copy()
        self.current_personnel_config: Dict = DEFAULT_PERSONNEL_CONFIG.copy() # Added for consistent loading/saving

        # Core components
        self.optimizer: Optional[OptimizationEngine] = None
        self.optimization_thread: Optional[OptimizationWorker] = None

        # Application settings (for window state, etc.)
        self.settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)

        self.initUI()
        self._connect_signals_slots()

        self._load_window_settings() # Load window size/pos
        self.startup_actions() # Perform actions like loading last project or creating new
        logger.info("ConstructionSiteOptimizerApp initialized and UI setup complete.")

    def initUI(self):
        """Initialize the main user interface layout and widgets."""
        logger.info("Initializing UI components...")
        self.setWindowTitle("Construction Site Layout Optimizer")
        self.setMinimumSize(1280, 768) # Adjusted minimum size
        # Attempt to load an application icon
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png') # Assuming icon.png is in the same dir
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            logger.warning(f"Application icon not found at {icon_path}. Using default.")
            self.setWindowIcon(QIcon())


        # Central Widget and Main Layout (using QSplitter for resizable panels)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget) # Main layout for the central widget

        # --- Left Panel (Controls & Settings) ---
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_widget.setMinimumWidth(380) # Minimum width for readability
        left_panel_widget.setMaximumWidth(500) # Max width to prevent it taking too much space

        # Plot Dimensions and Settings Group
        plot_group = QGroupBox("Site & Building Configuration")
        plot_layout = QGridLayout(plot_group) # Set layout for the group box
        plot_layout.setColumnStretch(1,1) # Allow input fields to expand

        plot_layout.addWidget(QLabel("Plot Width (m):"), 0, 0)
        self.plot_width_input = QDoubleSpinBox()
        self.plot_width_input.setRange(10, 2000); self.plot_width_input.setValue(self.plot_width_m)
        self.plot_width_input.setDecimals(1); self.plot_width_input.setSingleStep(1.0)
        self.plot_width_input.setToolTip("Total width of the construction site plot.")
        plot_layout.addWidget(self.plot_width_input, 0, 1)

        plot_layout.addWidget(QLabel("Plot Height (m):"), 1, 0)
        self.plot_height_input = QDoubleSpinBox()
        self.plot_height_input.setRange(10, 2000); self.plot_height_input.setValue(self.plot_height_m)
        self.plot_height_input.setDecimals(1); self.plot_height_input.setSingleStep(1.0)
        self.plot_height_input.setToolTip("Total height of the construction site plot.")
        plot_layout.addWidget(self.plot_height_input, 1, 1)

        plot_layout.addWidget(QLabel("Building Area Ratio:"), 2, 0)
        self.building_ratio_input = QDoubleSpinBox()
        self.building_ratio_input.setRange(0.05, 0.95); self.building_ratio_input.setSingleStep(0.01)
        self.building_ratio_input.setValue(self.building_ratio_val)
        self.building_ratio_input.setToolTip("Proportion of the total plot area allocated to the main building footprint (e.g., 0.75 for 75%).")
        plot_layout.addWidget(self.building_ratio_input, 2, 1)
        left_panel_layout.addWidget(plot_group)

        # Tab Widget for Detailed Settings
        self.settings_tabs = QTabWidget() # Make it an instance attribute
        self.space_widget = SpaceRequirementsWidget(
            site_elements_config=SITE_ELEMENTS,
            default_personnel_config=DEFAULT_PERSONNEL_CONFIG,
            default_space_requirements=DEFAULT_SPACE_REQUIREMENTS
        )
        self.settings_tabs.addTab(self.space_widget, "Space Requirements")

        self.constraints_widget = ConstraintsWidget()
        self.settings_tabs.addTab(self.constraints_widget, "Layout Constraints")

        self.optimization_control_widget = OptimizationControlWidget()
        self.settings_tabs.addTab(self.optimization_control_widget, "Optimization Controls")

        self.display_options_widget = DisplayOptionsWidget()
        self.settings_tabs.addTab(self.display_options_widget, "Display Options")
        left_panel_layout.addWidget(self.settings_tabs)

        # Element Properties Panel (Scrollable)
        properties_scroll_area = QScrollArea()
        properties_scroll_area.setWidgetResizable(True)
        properties_scroll_area.setFrameShape(QFrame.StyledPanel)
        self.element_properties_widget = ElementPropertiesWidget()
        properties_scroll_area.setWidget(self.element_properties_widget)
        properties_scroll_area.setMinimumHeight(200) # Ensure it has some vertical space
        left_panel_layout.addWidget(properties_scroll_area)

        left_panel_layout.addStretch(1) # Pushes buttons to bottom if any were here

        # Action Buttons at the bottom of the left panel
        action_buttons_layout = QHBoxLayout()
        self.init_layout_button = QPushButton(QIcon(), "Initialize / Reset Layout") # Added icon placeholder
        self.init_layout_button.setToolTip("Generate an initial layout based on current settings, or reset an existing one.")
        action_buttons_layout.addWidget(self.init_layout_button)

        self.run_opt_button_ui = QPushButton(QIcon(), "Run Optimization") # Added icon placeholder
        self.run_opt_button_ui.setToolTip("Start the automated layout optimization process.")
        # self.run_opt_button_ui.setStyleSheet( # Make it stand out slightly
        #     "QPushButton { background-color: #007bff; color: white; padding: 6px; border-radius: 3px;}"
        #     "QPushButton:hover { background-color: #0069d9; }"
        #     "QPushButton:disabled { background-color: #e9ecef; color: #6c757d; }"
        # )
        self.run_opt_button_ui.setStyleSheet(
            f"QPushButton {{ background-color: {DARK_THEME_SUCCESS_BUTTON_BG_COLOR.name()}; color: white; padding: 6px; border-radius: 3px;}}"
            f"QPushButton:hover {{ background-color: {DARK_THEME_SUCCESS_BUTTON_HOVER_BG_COLOR.name()}; }}"
            f"QPushButton:disabled {{ background-color: {QColor(60,60,60).name()}; color: {QColor(127,127,127).name()}; }}"
        )
        action_buttons_layout.addWidget(self.run_opt_button_ui)
        left_panel_layout.addLayout(action_buttons_layout)


        # --- Right Panel (Canvas and Visualization) ---
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(0,0,0,0) # No margins for the canvas panel

        # Canvas Title/Toolbar Area (optional) - kept minimal
        canvas_header_layout = QHBoxLayout()
        canvas_header_layout.setContentsMargins(5,5,5,2)
        self.canvas_title_label = QLabel("Site Layout Visualization")
        font = self.canvas_title_label.font(); font.setPointSize(14); font.setBold(True)
        self.canvas_title_label.setFont(font)
        canvas_header_layout.addWidget(self.canvas_title_label)
        canvas_header_layout.addStretch()
        # Example: Fit to view button for canvas
        fit_view_button = QPushButton(QIcon(), "Fit View") # Placeholder for icon
        fit_view_button.setToolTip("Adjust zoom and pan to fit the entire site plot into the view.")
        fit_view_button.clicked.connect(lambda: self.site_canvas.fit_plot_to_view())
        canvas_header_layout.addWidget(fit_view_button)
        right_panel_layout.addLayout(canvas_header_layout)

        # Main Site Canvas
        self.site_canvas = SiteCanvas()
        right_panel_layout.addWidget(self.site_canvas, 1) # Canvas takes most available vertical space

        # Status Bar-like Area at the bottom of the right panel (for layout metrics)
        metrics_bar_widget = QFrame() # Use QFrame for a slight visual separation
        metrics_bar_widget.setFrameShape(QFrame.StyledPanel)
        metrics_bar_widget.setFrameShadow(QFrame.Sunken)
        metrics_layout = QHBoxLayout(metrics_bar_widget)
        metrics_layout.setContentsMargins(8, 4, 8, 4) # Padding inside the metrics bar

        self.info_text_label = QLabel("Tip: Select elements to edit. Use Shift+Drag or Middle-Mouse to pan. Wheel to zoom.")
        self.info_text_label.setStyleSheet(f"font-size: 8pt; color: {DARK_THEME_INFO_TEXT_COLOR.name()};")
        metrics_layout.addWidget(self.info_text_label)
        metrics_layout.addStretch()

        self.score_label = QLabel("Score: N/A")
        self.score_label.setToolTip("Overall layout quality score (higher is better).")
        metrics_layout.addWidget(self.score_label)
        metrics_layout.addSpacing(15)

        self.vehicle_path_label = QLabel("Veh. Paths: N/A")
        self.vehicle_path_label.setToolTip("Total length of vehicle paths.")
        metrics_layout.addWidget(self.vehicle_path_label)
        metrics_layout.addSpacing(15)

        self.ped_path_label = QLabel("Ped. Paths: N/A")
        self.ped_path_label.setToolTip("Total length of pedestrian paths.")
        metrics_layout.addWidget(self.ped_path_label)
        metrics_layout.addSpacing(15)

        self.interference_label = QLabel("Interference: N/A")
        self.interference_label.setToolTip("Score indicating interference between vehicle and pedestrian paths (lower is better).")
        metrics_layout.addWidget(self.interference_label)
        right_panel_layout.addWidget(metrics_bar_widget)

        # Using QSplitter
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_panel_widget)
        self.splitter.addWidget(right_panel_widget)
        self.splitter.setStretchFactor(0, 0) # Left panel not to stretch initially beyond its size hint
        self.splitter.setStretchFactor(1, 1) # Right panel (canvas) takes available space
        self.splitter.setSizes([400, self.width() - 400]) # Initial sizes
        main_layout.addWidget(self.splitter)


        self._setup_menu_bar()
        self._setup_status_bar_app() # Main application status bar at the very bottom
        logger.info("UI Initialization sequence complete.")

    def _setup_menu_bar(self):
        """Sets up the main application menu bar and actions."""
        logger.debug("Setting up menu bar...")
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False) # Ensures menu bar is part of the window on macOS

        # --- File Menu ---
        file_menu = menubar.addMenu("&File")
        self.new_project_action = QAction(QIcon.fromTheme("document-new", QIcon()), "&New Project", self)
        self.new_project_action.setShortcut(Qt.CTRL + Qt.Key_N)
        self.new_project_action.setStatusTip("Create a new site layout project.")
        self.new_project_action.triggered.connect(lambda: self._create_new_project(confirm=True))
        file_menu.addAction(self.new_project_action)

        self.open_project_action = QAction(QIcon.fromTheme("document-open", QIcon()), "&Open Project...", self)
        self.open_project_action.setShortcut(Qt.CTRL + Qt.Key_O)
        self.open_project_action.setStatusTip("Open an existing site layout project file.")
        self.open_project_action.triggered.connect(self._load_project_dialog)
        file_menu.addAction(self.open_project_action)

        file_menu.addSeparator()

        self.save_project_action = QAction(QIcon.fromTheme("document-save", QIcon()), "&Save Project", self)
        self.save_project_action.setShortcut(Qt.CTRL + Qt.Key_S)
        self.save_project_action.setStatusTip("Save the current project.")
        self.save_project_action.triggered.connect(self._save_project)
        file_menu.addAction(self.save_project_action)

        self.save_project_as_action = QAction(QIcon.fromTheme("document-save-as", QIcon()), "Save Project &As...", self)
        self.save_project_as_action.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_S)
        self.save_project_as_action.setStatusTip("Save the current project to a new file.")
        self.save_project_as_action.triggered.connect(self._save_project_as_dialog)
        file_menu.addAction(self.save_project_as_action)

        file_menu.addSeparator()

        self.export_layout_action = QAction(QIcon.fromTheme("document-export", QIcon()), "E&xport Layout as Image...", self)
        self.export_layout_action.setStatusTip("Export the current site layout visualization as an image file.")
        self.export_layout_action.triggered.connect(self._export_layout_as_image)
        file_menu.addAction(self.export_layout_action)

        file_menu.addSeparator()

        self.exit_action = QAction(QIcon.fromTheme("application-exit", QIcon()), "E&xit", self)
        self.exit_action.setShortcut(Qt.CTRL + Qt.Key_Q)
        self.exit_action.setStatusTip("Exit the application.")
        self.exit_action.triggered.connect(self.close) # `self.close()` will trigger `closeEvent`
        file_menu.addAction(self.exit_action)

        # --- Edit Menu ---
        edit_menu = menubar.addMenu("&Edit")
        # Example: If you implement Undo/Redo
        # self.undo_action = QAction(QIcon.fromTheme("edit-undo"), "&Undo", self)
        # self.redo_action = QAction(QIcon.fromTheme("edit-redo"), "&Redo", self)
        # edit_menu.addAction(self.undo_action)
        # edit_menu.addAction(self.redo_action)
        # edit_menu.addSeparator()

        self.init_layout_action_menu = QAction(QIcon(), "Initialize Current Layout", self) # Placeholder for icon
        self.init_layout_action_menu.setStatusTip("Generate or reset the layout based on current settings.")
        self.init_layout_action_menu.triggered.connect(self.handle_initialize_layout)
        edit_menu.addAction(self.init_layout_action_menu)

        self.run_opt_action_menu = QAction(QIcon(), "Run Optimization", self) # Placeholder for icon
        self.run_opt_action_menu.setStatusTip("Start the automated layout optimization process.")
        # Connect directly to the optimization control widget's signal emitter
        self.run_opt_action_menu.triggered.connect(self.optimization_control_widget.emit_start_optimization_signal)
        edit_menu.addAction(self.run_opt_action_menu)

        # --- View Menu ---
        view_menu = menubar.addMenu("&View")
        self.toggle_grid_action = view_menu.addAction("Show Grid")
        self.toggle_grid_action.setCheckable(True)
        self.toggle_grid_action.setChecked(True) # Default
        self.toggle_grid_action.setStatusTip("Toggle visibility of the background grid on the canvas.")

        self.toggle_labels_action = view_menu.addAction("Show Element Labels")
        self.toggle_labels_action.setCheckable(True)
        self.toggle_labels_action.setChecked(True)
        self.toggle_labels_action.setStatusTip("Toggle visibility of labels on site elements.")

        self.toggle_dims_action = view_menu.addAction("Show Element Dimensions")
        self.toggle_dims_action.setCheckable(True)
        self.toggle_dims_action.setChecked(True)
        self.toggle_dims_action.setStatusTip("Toggle visibility of WxH dimensions on site elements.")

        self.toggle_crane_radius_action = view_menu.addAction("Show Crane Radius")
        self.toggle_crane_radius_action.setCheckable(True)
        self.toggle_crane_radius_action.setChecked(True)
        self.toggle_crane_radius_action.setStatusTip("Toggle visibility of the tower crane's operational radius.")

        self.toggle_internal_markers_action = view_menu.addAction("Show Internal Element Markers")
        self.toggle_internal_markers_action.setCheckable(True)
        self.toggle_internal_markers_action.setChecked(True)
        self.toggle_internal_markers_action.setStatusTip("Toggle visibility of markers for elements placed inside the main building.")
        view_menu.addSeparator()

        self.fit_view_action_menu = QAction(QIcon(), "Fit Plot to View", self)
        self.fit_view_action_menu.setStatusTip("Adjust zoom and pan to fit the entire site plot.")
        self.fit_view_action_menu.triggered.connect(self.site_canvas.fit_plot_to_view)
        view_menu.addAction(self.fit_view_action_menu)


        # --- Help Menu ---
        help_menu = menubar.addMenu("&Help")
        self.about_action = QAction(QIcon.fromTheme("help-about", QIcon()), "&About...", self)
        self.about_action.setStatusTip("Show application information.")
        self.about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(self.about_action)

        self.usage_guide_action = QAction(QIcon.fromTheme("help-contents", QIcon()), "&Usage Guide...", self)
        self.usage_guide_action.setStatusTip("Display a brief guide on how to use the application.")
        self.usage_guide_action.triggered.connect(self._show_usage_guide)
        help_menu.addAction(self.usage_guide_action)

    def _setup_status_bar_app(self):
        """Sets up the main application status bar (at the very bottom of QMainWindow)."""
        self.main_status_bar = self.statusBar() # Get the QMainWindow's status bar
        self.main_status_bar.showMessage("Ready. Create or open a project to begin.") # Initial message

    def _connect_signals_slots(self):
        """Connect signals from various UI components to their corresponding handler methods."""
        logger.debug("Connecting signals and slots...")

        # Plot settings changes
        self.plot_width_input.valueChanged.connect(self._handle_plot_dimension_or_ratio_change)
        self.plot_height_input.valueChanged.connect(self._handle_plot_dimension_or_ratio_change)
        self.building_ratio_input.valueChanged.connect(self._handle_plot_dimension_or_ratio_change)
        self.project_settings_changed.connect(self._on_project_settings_changed)

        # Space, Constraints, Optimization, Display widgets
        self.space_widget.requirements_changed.connect(self._handle_space_requirements_changed)
        self.constraints_widget.constraints_changed.connect(self._handle_constraints_changed)

        self.optimization_control_widget.start_optimization.connect(self.run_optimization_process)
        self.optimization_control_widget.request_weights_update.connect(self._provide_weights_to_optimizer_control)

        self.display_options_widget.options_changed.connect(self.site_canvas.set_display_options)

        # Element properties widget
        self.element_properties_widget.properties_changed.connect(self._handle_element_properties_changed)
        self.element_properties_widget.request_element_delete.connect(self._handle_element_delete_request)
        self.element_properties_widget.request_element_duplicate.connect(self._handle_element_duplicate_request)


        # SiteCanvas interactions
        self.site_canvas.element_selected.connect(self.element_properties_widget.set_element)
        self.site_canvas.element_moved.connect(self._handle_element_manual_move)
        self.site_canvas.view_changed.connect(self._update_canvas_related_ui) # e.g. scale bar, mouse coords


        # UI Buttons in main app UI (not within specific widgets)
        self.init_layout_button.clicked.connect(self.handle_initialize_layout)
        self.run_opt_button_ui.clicked.connect(self.optimization_control_widget.emit_start_optimization_signal)

        # View Menu Actions <-> DisplayOptionsWidget Checkboxes (two-way binding)
        # Action -> Checkbox
        self.toggle_grid_action.triggered.connect(self.display_options_widget.set_grid_visible)
        self.toggle_labels_action.triggered.connect(self.display_options_widget.set_labels_visible)
        self.toggle_dims_action.triggered.connect(self.display_options_widget.set_dims_visible)
        self.toggle_crane_radius_action.triggered.connect(self.display_options_widget.set_crane_radius_visible)
        self.toggle_internal_markers_action.triggered.connect(self.display_options_widget.set_internal_markers_visible)
        # Checkbox -> Action
        self.display_options_widget.grid_cb.toggled.connect(self.toggle_grid_action.setChecked)
        self.display_options_widget.labels_cb.toggled.connect(self.toggle_labels_action.setChecked)
        self.display_options_widget.dims_cb.toggled.connect(self.toggle_dims_action.setChecked)
        self.display_options_widget.crane_radius_cb.toggled.connect(self.toggle_crane_radius_action.setChecked)

        self.display_options_widget.internal_markers_cb.toggled.connect(self.toggle_internal_markers_action.setChecked)

        # Initial sync of view menu checkboxes with display options widget defaults
        self._sync_view_menu_to_display_options()

    def _sync_view_menu_to_display_options(self):
        """Synchronizes the checked state of View menu actions with DisplayOptionsWidget checkboxes."""
        self.toggle_grid_action.setChecked(self.display_options_widget.grid_cb.isChecked())
        self.toggle_labels_action.setChecked(self.display_options_widget.labels_cb.isChecked())
        self.toggle_dims_action.setChecked(self.display_options_widget.dims_cb.isChecked())
        self.toggle_crane_radius_action.setChecked(self.display_options_widget.crane_radius_cb.isChecked())
        self.toggle_internal_markers_action.setChecked(self.display_options_widget.internal_markers_cb.isChecked())

    def _update_canvas_related_ui(self):
        """Placeholder for actions needed when canvas view (zoom/pan) changes."""
        # For example, if there was a scale display or coordinate display tied to the main app UI
        pass

    def startup_actions(self):
        """Actions to perform on application startup."""
        logger.info("Performing startup actions...")
        # Attempt to load last opened project path from settings
        last_project_path = self.settings.value("lastProjectPath", "")
        if last_project_path and os.path.exists(last_project_path):
            reply = QMessageBox.question(self, "Load Last Project",
                                         f"Do you want to load the last opened project?\n({last_project_path})",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self._load_project_from_file(last_project_path)
                return # Exit startup_actions if project loaded

        # If no last project or user chose not to load, create a new default project.
        self._create_new_project(confirm=False) # Create a default project without confirmation dialog
        QTimer.singleShot(150, lambda: QMessageBox.information(
            self, f"Welcome to {APPLICATION_NAME}",
            f"Welcome to the {APPLICATION_NAME} v{APP_VERSION}!\n\n"
            "To get started:\n"
            "1. Adjust Site Configuration (plot dimensions, building ratio).\n"
            "2. Define Space Requirements for each facility.\n"
            "3. Review and adjust Layout Constraints.\n"
            "4. Click 'Initialize / Reset Layout'.\n"
            "5. Manually refine or use 'Run Optimization'.\n\n"
            "You can save/load projects using the File menu."
        ))


    def _mark_unsaved_changes(self, changed: bool = True):
        """Updates the window title to indicate unsaved changes."""
        if self.unsaved_changes == changed:
            return # No change in status

        self.unsaved_changes = changed
        title = APPLICATION_NAME
        if self.current_project_filename:
            title += f" - {os.path.basename(self.current_project_filename)}"
        if self.unsaved_changes:
            title += "*" # Asterisk indicates unsaved changes
        self.setWindowTitle(title)

    def _confirm_discard_changes(self) -> bool:
        """If there are unsaved changes, asks the user to save, discard, or cancel.
           Returns True if it's OK to proceed (saved or discarded), False if cancelled."""
        if not self.unsaved_changes:
            return True # No unsaved changes, OK to proceed

        reply = QMessageBox.warning(self, "Unsaved Changes",
                                     "The current project has unsaved changes. Do you want to save them?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                     QMessageBox.Save) # Default to Save

        if reply == QMessageBox.Save:
            return self._save_project() # _save_project returns True on success/no-file, False on cancel/error
        elif reply == QMessageBox.Discard:
            return True # User chose to discard changes
        else: # Cancel
            return False # User cancelled the operation

    # --- Project Management Methods ---
    def _create_new_project(self, confirm: bool = True):
        logger.info("Creating new project process started...")
        if confirm and not self._confirm_discard_changes():
            logger.info("New project creation cancelled by user due to unsaved changes.")
            return

        self.current_project_filename = None
        # Reset to application defaults
        self.plot_width_m = 80.0
        self.plot_height_m = 50.0
        self.building_ratio_val = 0.45

        # Update UI input fields to defaults, blocking signals temporarily
        self.plot_width_input.blockSignals(True)
        self.plot_height_input.blockSignals(True)
        self.building_ratio_input.blockSignals(True)
        self.plot_width_input.setValue(self.plot_width_m)
        self.plot_height_input.setValue(self.plot_height_m)
        self.building_ratio_input.setValue(self.building_ratio_val)
        self.plot_width_input.blockSignals(False)
        self.plot_height_input.blockSignals(False)
        self.building_ratio_input.blockSignals(False)


        # Reset constraints and space requirements to their global defaults
        self.current_constraints = DEFAULT_CONSTRAINTS.copy()
        self.current_space_requirements = DEFAULT_SPACE_REQUIREMENTS.copy()
        self.current_personnel_config = DEFAULT_PERSONNEL_CONFIG.copy()

        self.constraints_widget.set_constraints_data(self.current_constraints)
        self.space_widget.set_requirements_data(self.current_space_requirements, self.current_personnel_config)
        # OptimizationControlWidget defaults are set in its init. Weights are updated on selection.

        # Critical: Instantiate Optimizer with correct current values
        self.optimizer = OptimizationEngine(
            plot_width=self.plot_width_m,
            plot_height=self.plot_height_m,
            elements=[], # Start with no elements; optimizer will create Building
            constraints=self.current_constraints,
            space_requirements=self.current_space_requirements,
            building_ratio=self.building_ratio_val
        )
        # Ensure optimizer gets initial weights from the control widget
        self.optimizer.optimization_weights = self.optimization_control_widget.get_optimization_weights()

        self.site_canvas.set_plot_dimensions(self.plot_width_m, self.plot_height_m)
        self.site_canvas.set_constraints(self.current_constraints) # Pass current constraints to canvas

        self.handle_initialize_layout() # Create an initial layout for the new project
        self._mark_unsaved_changes(False) # New project is "clean"
        self.main_status_bar.showMessage("New project created with default settings. Ready.")
        logger.info("New project setup complete.")

    def _load_project_dialog(self):
        if not self._confirm_discard_changes():
            return # User cancelled due to unsaved changes

        filename, _ = QFileDialog.getOpenFileName(self, "Load Project",
                                                  self.settings.value("lastProjectDir", QDir.homePath()), # Start in last dir or home
                                                  "Site Layout Project (*.slyp);;All Files (*)")
        if filename:
            self.settings.setValue("lastProjectDir", QFileInfo(filename).absolutePath()) # Save directory
            self._load_project_from_file(filename)

    def _load_project_from_file(self, filename: str):
        logger.info(f"Loading project from: {filename}")
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # Load project settings
            self.plot_width_m = data.get('plot_width', 80.0)
            self.plot_height_m = data.get('plot_height', 50.0)
            self.building_ratio_val = data.get('building_ratio', 0.45)

            # Update UI for plot settings, blocking signals
            self.plot_width_input.blockSignals(True); self.plot_width_input.setValue(self.plot_width_m); self.plot_width_input.blockSignals(False)
            self.plot_height_input.blockSignals(True); self.plot_height_input.setValue(self.plot_height_m); self.plot_height_input.blockSignals(False)
            self.building_ratio_input.blockSignals(True); self.building_ratio_input.setValue(self.building_ratio_val); self.building_ratio_input.blockSignals(False)


            self.current_constraints = data.get('constraints', DEFAULT_CONSTRAINTS.copy())
            self.current_space_requirements = data.get('space_requirements', DEFAULT_SPACE_REQUIREMENTS.copy())
            self.current_personnel_config = data.get('personnel_config', DEFAULT_PERSONNEL_CONFIG.copy())

            # Update control widgets with loaded data
            self.constraints_widget.set_constraints_data(self.current_constraints)
            self.space_widget.set_requirements_data(self.current_space_requirements, self.current_personnel_config)
            # Note: Optimization parameters (iterations, temp, etc.) are typically not saved per project,
            # but are user preferences for the session. Could be added to project data if desired.

            loaded_elements_data = data.get('elements', [])
            loaded_elements_instances = []
            for el_data in loaded_elements_data:
                el = SiteElement(el_data['name'], el_data['x'], el_data['y'],
                                 el_data['width'], el_data['height'], el_data.get('rotation',0.0))
                # Color might be list or string, handle robustly
                color_data = el_data.get('color')
                if isinstance(color_data, list) and len(color_data) in [3,4]:
                    el.color = QColor(*color_data)
                elif isinstance(color_data, str):
                    el.color = QColor(color_data)
                else: # Fallback
                    el.color = SITE_ELEMENTS.get(el.name, {}).get('color', QColor(200,200,200))

                el.movable = el_data.get('movable', True)
                el.selected = el_data.get('selected', False) # Load selection state
                el.is_placed_inside_building = el_data.get('is_placed_inside_building', False)
                # Load other custom attributes if saved...
                loaded_elements_instances.append(el)

            # Re-initialize optimizer with loaded data
            self.optimizer = OptimizationEngine(
                plot_width=self.plot_width_m,
                plot_height=self.plot_height_m,
                elements=loaded_elements_instances, # Pass loaded elements
                constraints=self.current_constraints,
                space_requirements=self.current_space_requirements,
                building_ratio=self.building_ratio_val
            )
            self.optimizer.optimization_weights = self.optimization_control_widget.get_optimization_weights()

            self.site_canvas.set_plot_dimensions(self.plot_width_m, self.plot_height_m)
            self.site_canvas.set_constraints(self.current_constraints)
            self.site_canvas.set_elements(self.optimizer.elements) # Display loaded elements
            self._update_layout_and_score() # Update routes and score display

            self.current_project_filename = filename
            self.settings.setValue("lastProjectPath", filename) # Save as last loaded project
            self._mark_unsaved_changes(False) # Loaded project is "clean"
            self.main_status_bar.showMessage(f"Project '{os.path.basename(filename)}' loaded successfully.")
            logger.info(f"Project '{filename}' loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project file '{filename}':\n{str(e)}")
            logger.error(f"Error loading project '{filename}': {e}", exc_info=True)
            self._create_new_project(confirm=False) # Fallback to a new default project

    def _save_project(self) -> bool:
        """Saves the current project. If no filename, calls Save As."""
        if not self.current_project_filename:
            return self._save_project_as_dialog()
        else:
            return self._save_project_to_file(self.current_project_filename)

    def _save_project_as_dialog(self) -> bool:
        """Opens a dialog to save the project to a new file."""
        filename, _ = QFileDialog.getSaveFileName(self, "Save Project As",
                                                  self.settings.value("lastProjectDir", QDir.homePath()),
                                                  "Site Layout Project (*.slyp);;All Files (*)")
        if filename:
            if not filename.lower().endswith('.slyp'):
                filename += '.slyp'
            self.settings.setValue("lastProjectDir", QFileInfo(filename).absolutePath()) # Save directory
            return self._save_project_to_file(filename)
        return False # User cancelled

    def _save_project_to_file(self, filename: str) -> bool:
        logger.info(f"Saving project to: {filename}")
        if not self.optimizer:
            QMessageBox.warning(self, "Save Error", "No layout data to save. Initialize a layout first.")
            logger.warning("Attempted to save project with no optimizer instance.")
            return False

        project_data = {
            'version': APP_VERSION, # Store app version with project data
            'saved_date': QDate.currentDate().toString(Qt.ISODate),
            'plot_width': self.plot_width_m,
            'plot_height': self.plot_height_m,
            'building_ratio': self.building_ratio_val,
            'constraints': self.current_constraints,
            'space_requirements': self.current_space_requirements,
            'personnel_config': self.space_widget.get_current_personnel_config(), # Save personnel setup
            'elements': []
        }
        for el in self.optimizer.elements:
            project_data['elements'].append({
                'name': el.name, 'x': el.x, 'y': el.y,
                'width': el.width, 'height': el.height, 'rotation': el.rotation,
                'color': [el.color.red(), el.color.green(), el.color.blue(), el.color.alpha()], # RGBA list
                'movable': el.movable,
                'selected': el.selected, # Save selection state
                'is_placed_inside_building': getattr(el, 'is_placed_inside_building', False),
                # Add other relevant attributes of SiteElement if needed
            })

        try:
            with open(filename, 'w') as f:
                json.dump(project_data, f, indent=2) # indent for readability
            self.current_project_filename = filename
            self.settings.setValue("lastProjectPath", filename) # Update last saved project path
            self._mark_unsaved_changes(False) # Project is now saved
            self.main_status_bar.showMessage(f"Project saved to '{os.path.basename(filename)}'.")
            logger.info(f"Project successfully saved to '{filename}'.")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save project to '{filename}':\n{str(e)}")
            logger.error(f"Error saving project to '{filename}': {e}", exc_info=True)
            return False

    def _export_layout_as_image(self):
        if not self.optimizer or not self.optimizer.elements:
            QMessageBox.information(self, "Export Layout", "No layout to export. Please initialize a layout first.")
            return

        filename, selected_filter = QFileDialog.getSaveFileName(self, "Export Layout as Image",
                                                  self.settings.value("lastImageExportDir", QDir.homePath()),
                                                  "PNG Image (*.png);;JPEG Image (*.jpg);;SVG Image (*.svg)")
        if not filename:
            return

        self.settings.setValue("lastImageExportDir", QFileInfo(filename).absolutePath())
        logger.info(f"Exporting layout to image: {filename} with filter: {selected_filter}")

        try:
            # For higher quality, render to an intermediate QImage if not SVG
            if ".svg" not in selected_filter.lower():
                # Determine a suitable size for export, potentially larger than screen
                # For simplicity, using current canvas grab. For high-res, render to larger QImage.
                pixmap = self.site_canvas.grab()
                success = pixmap.save(filename)
            else:
                # SVG export requires QSVGGenerator, which is more involved
                # For now, indicate SVG is not directly supported this way or use a library
                QMessageBox.information(self, "SVG Export", "Direct SVG export is complex. Consider using PNG and converting, or implementing QSVGGenerator.")
                logger.warning("SVG export requested, but not fully implemented with QSVGGenerator in this example.")
                return # Or attempt simple pixmap save anyway

            if success:
                QMessageBox.information(self, "Export Successful", f"Layout exported to:\n{filename}")
                logger.info(f"Layout successfully exported to '{filename}'.")
            else:
                QMessageBox.warning(self, "Export Failed", f"Could not save image to:\n{filename}. Ensure file type is PNG or JPG.")
                logger.warning(f"Failed to save exported image to '{filename}'.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred during export:\n{str(e)}")
            logger.error(f"Error exporting layout image: {e}", exc_info=True)


    # --- Handler Methods for UI Interactions & Optimizer ---
    def _on_project_settings_changed(self):
        """Called when plot dimensions or building ratio changes fundamentally."""
        logger.info("Project settings changed (plot dimensions/ratio), re-initializing optimizer and layout.")
        if not self.optimizer:
            logger.warning("_on_project_settings_changed called with no optimizer. This indicates an issue if a project should be active.")
            self._create_new_project(confirm=False) # Ensure optimizer exists, without user prompt if mid-change
            if not self.optimizer: # If still no optimizer, critical failure
                logger.error("Failed to create optimizer instance in _on_project_settings_changed.")
                QMessageBox.critical(self, "Critical Error", "Failed to initialize optimization engine. Application might not function correctly.")
                return

        # Update optimizer with new dimensions
        self.optimizer.plot_width = self.plot_width_m
        self.optimizer.plot_height = self.plot_height_m
        self.optimizer.building_ratio = self.building_ratio_val
        # This will re-calculate building size/pos and update it in self.optimizer.elements
        self.optimizer.initialize_building()

        self.site_canvas.set_plot_dimensions(self.plot_width_m, self.plot_height_m)
        # The layout might be invalid now, so re-initialize it if elements (other than building) exist
        if len([el for el in self.optimizer.elements if el.name != 'Building']) > 0:
            logger.info("Re-initializing layout due to plot dimension changes.")
            self.handle_initialize_layout(confirm_reset=False) # Re-initialize without confirmation
        else: # Only building exists, just update canvas
            self.site_canvas.set_elements(self.optimizer.elements)
            self._update_layout_and_score()


        self._mark_unsaved_changes()


    def _handle_plot_dimension_or_ratio_change(self):
        """Handler for plot width, height, or building ratio input value changes."""
        # Check if the sender is one of the plot dimension inputs
        if self.sender() in [self.plot_width_input, self.plot_height_input, self.building_ratio_input]:
            new_width = self.plot_width_input.value()
            new_height = self.plot_height_input.value()
            new_ratio = self.building_ratio_input.value()

            # Only proceed if values have actually changed to avoid redundant updates
            if abs(new_width - self.plot_width_m) > 1e-3 or \
               abs(new_height - self.plot_height_m) > 1e-3 or \
               abs(new_ratio - self.building_ratio_val) > 1e-3:
                self.plot_width_m = new_width
                self.plot_height_m = new_height
                self.building_ratio_val = new_ratio
                self.project_settings_changed.emit() # Emit signal that fundamental settings changed
            else:
                logger.debug("Plot dimension/ratio input changed, but value is same as current. No action.")


    def _handle_space_requirements_changed(self, new_requirements: Dict[str, float]):
        self.current_space_requirements = new_requirements
        if self.optimizer:
            self.optimizer.space_requirements = self.current_space_requirements.copy() # Give optimizer a copy
            # When space reqs change, element sizes might need to update.
            # An ideal way is to have elements update their W/H based on new area and aspect ratio.
            # For now, re-initializing layout is a direct way to reflect changes.
            # Consider a less disruptive update if possible in the future.
            self.handle_initialize_layout(confirm_reset=False) # Re-initialize without prompt
        self._mark_unsaved_changes()
        logger.debug(f"Space requirements updated by widget: {new_requirements}")

    def _handle_constraints_changed(self, new_constraints: Dict[str, float]):
        self.current_constraints = new_constraints
        if self.optimizer:
            self.optimizer.constraints = self.current_constraints.copy()
        if self.site_canvas:
            self.site_canvas.set_constraints(self.current_constraints.copy())
        self._update_layout_and_score() # Re-evaluate score and paths based on new constraints
        self._mark_unsaved_changes()
        logger.debug(f"Constraints updated by widget: {new_constraints}")

    def _provide_weights_to_optimizer_control(self, current_weights: Dict[str, float]):
        """Handles request from OptimizationControlWidget to update its weights.
           (Or this can be a direct fetch: self.optimizer.optimization_weights = self.optimization_control_widget.get_optimization_weights())
        """
        if self.optimizer:
            self.optimizer.optimization_weights = current_weights
            logger.debug(f"Optimization weights updated in engine: {current_weights}")
            self._update_layout_and_score() # Score might change with new weights
            self._mark_unsaved_changes() # Changing weights is a project change if it affects outcome

    def _handle_element_properties_changed(self, changed_element: SiteElement):
        """Called when an element's properties are changed in the ElementPropertiesWidget."""
        if not self.optimizer or changed_element not in self.optimizer.elements:
            # This check might fail if changed_element is a copy.
            # Better to find by ID or assume properties widget always holds a ref from optimizer.elements.
            # For now, assume changed_element is the actual instance from self.optimizer.elements.
            logger.warning(f"Properties changed for element {changed_element.name}, but it's not in current optimizer list.")
            return

        # The element object itself should have been modified by ElementPropertiesWidget.
        # We need to ensure the canvas and score reflect this.
        self._update_layout_and_score() # This updates routes, canvas (implicitly via set_elements if needed), and score
        self._mark_unsaved_changes()
        logger.debug(f"Properties changed for element: {changed_element.name}. New pos: ({changed_element.x}, {changed_element.y})")


    def _handle_element_manual_move(self, moved_element: SiteElement):
        """Called when an element is moved on the canvas by the user (drag or key)."""
        # Element position is already updated in the self.optimizer.elements list by SiteCanvas logic
        # (SiteCanvas holds references to elements from self.optimizer.elements).
        self._update_layout_and_score()
        self._mark_unsaved_changes()
        logger.debug(f"Element manually moved: {moved_element.name} to ({moved_element.x:.1f}, {moved_element.y:.1f})")

    def _handle_element_delete_request(self, element_to_delete: SiteElement):
        if not self.optimizer: return
        if element_to_delete in self.optimizer.elements:
            if element_to_delete.name == "Building":
                QMessageBox.warning(self, "Action Denied", "The main 'Building' element cannot be deleted.")
                return

            reply = QMessageBox.question(self, "Confirm Delete",
                                         f"Are you sure you want to delete element '{element_to_delete.name}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.optimizer.elements.remove(element_to_delete)
                if self.site_canvas.selected_element_obj == element_to_delete:
                    self.site_canvas.clear_selection() # Also clears properties panel
                else: # Element might be deleted without being selected in panel
                    self.element_properties_widget.set_element(None)

                self._update_layout_and_score()
                self._mark_unsaved_changes()
                logger.info(f"Element '{element_to_delete.name}' deleted.")
        else:
            logger.warning(f"Request to delete element '{element_to_delete.name}' which is not in optimizer list.")


    def _handle_element_duplicate_request(self, element_to_duplicate: SiteElement):
        if not self.optimizer: return
        if element_to_duplicate.name == "Building":
            QMessageBox.warning(self, "Action Denied", "The main 'Building' element cannot be duplicated.")
            return
        if getattr(element_to_duplicate, 'is_placed_inside_building', False):
            QMessageBox.warning(self, "Action Denied", "Elements placed inside the building cannot be duplicated directly through this panel.")
            return


        # Create a true deep copy for the new element
        new_element = self.optimizer.clone_elements(source_elements_list=[element_to_duplicate])[0]

        # Modify name to indicate it's a copy (e.g., "Contractor Office Copy 1")
        copy_count = 1
        base_name = new_element.name.split(" Copy")[0] # Get base name if already a copy
        while any(el.name == f"{base_name} Copy {copy_count}" for el in self.optimizer.elements):
            copy_count += 1
        new_element.name = f"{base_name} Copy {copy_count}"

        # Offset the new element slightly to avoid exact overlap
        new_element.x += new_element.width * 0.2 + 2.0 # Offset by 20% of its width + 2m
        new_element.y += new_element.height * 0.2 + 2.0
        new_element.selected = False # Duplicated element is not initially selected

        # Clamp to plot boundaries (simple AABB clamp)
        new_element.x = max(0, min(new_element.x, self.plot_width_m - new_element.width))
        new_element.y = max(0, min(new_element.y, self.plot_height_m - new_element.height))


        self.optimizer.elements.append(new_element)
        self._update_layout_and_score() # Updates canvas elements and routes
        self._mark_unsaved_changes()
        logger.info(f"Element '{element_to_duplicate.name}' duplicated as '{new_element.name}'.")
        # Optionally, select the new element on canvas and in properties
        self.site_canvas.selected_element_obj = new_element # This needs careful handling to ensure selection updates
        new_element.selected = True
        self.element_properties_widget.set_element(new_element)
        self.site_canvas.update()



    def handle_initialize_layout(self, confirm_reset: bool = True):
        """Generates a new initial layout or resets the current one."""
        logger.info("Initializing/Resetting layout...")
        if not self.optimizer:
            QMessageBox.warning(self, "Error", "Optimizer not initialized. Cannot create/reset layout. Please create or open a project.")
            logger.error("Attempted to initialize layout with no optimizer instance.")
            return

        if confirm_reset and len([el for el in self.optimizer.elements if el.name != 'Building']) > 0:
            reply = QMessageBox.question(self, "Confirm Reset Layout",
                                         "This will clear all existing elements (except the Building) and generate a new initial layout. Are you sure?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                logger.info("Layout initialization/reset cancelled by user.")
                return

        # Ensure optimizer has the latest weights, space reqs, constraints
        self.optimizer.optimization_weights = self.optimization_control_widget.get_optimization_weights()
        self.optimizer.space_requirements = self.current_space_requirements.copy()
        self.optimizer.constraints = self.current_constraints.copy()
        self.optimizer.plot_width = self.plot_width_m # Ensure plot dims are current
        self.optimizer.plot_height = self.plot_height_m
        self.optimizer.building_ratio = self.building_ratio_val


        self.optimizer.create_initial_layout() # This populates/re-populates optimizer.elements

        self.site_canvas.set_elements(self.optimizer.elements) # Update canvas with new elements
        self._update_layout_and_score() # This will update routes and score display
        self.element_properties_widget.set_element(None) # Clear selection in properties panel
        self._mark_unsaved_changes()
        self.main_status_bar.showMessage("Layout initialized/reset successfully.")
        logger.info("Layout initialization/reset complete.")

    def run_optimization_process(self, iterations: int, temperature: float, cooling_rate: float,
                                 min_temp: float, reheat_factor: float, reheat_threshold: float):
        logger.info(f"Request to start optimization: Iter={iterations}, T0={temperature:.1f}, Cool={cooling_rate:.3f}, MinT={min_temp:.2f}")
        if not self.optimizer:
            QMessageBox.critical(self, "Error", "Optimizer not initialized. Cannot run optimization.")
            logger.error("Attempted to run optimization with no optimizer instance.")
            return
        if not self.optimizer.elements or len([e for e in self.optimizer.elements if e.name !="Building"]) == 0 :
            QMessageBox.information(self, "Optimization", "Please initialize a layout with some elements (other than just the Building) before running optimization.")
            logger.warning("Optimization run attempted without an initialized layout or only Building present.")
            self.optimization_control_widget.enable_controls(True)
            self.optimization_control_widget.set_progress(0)
            return
        if self.optimization_thread and self.optimization_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Optimization is already in progress.")
            logger.warning("Attempted to start new optimization while previous one is running.")
            return


        self.optimization_control_widget.enable_controls(False) # Disable UI during optimization
        self.main_status_bar.showMessage("Optimization in progress... Please wait.")

        # Ensure optimizer has the latest weights and other parameters
        self.optimizer.optimization_weights = self.optimization_control_widget.get_optimization_weights()
        self.optimizer.constraints = self.current_constraints.copy() # Ensure it has latest
        self.optimizer.space_requirements = self.current_space_requirements.copy()


        self.optimization_thread = OptimizationWorker(
            self.optimizer, iterations, temperature, cooling_rate,
            min_temp, reheat_factor, reheat_threshold
        )
        self.optimization_thread.progress_updated.connect(self.optimization_control_widget.set_progress)
        self.optimization_thread.optimization_complete.connect(self._handle_optimization_complete)
        self.optimization_thread.optimization_failed.connect(self._handle_optimization_failed)
        self.optimization_thread.finished.connect(self._handle_optimization_thread_finished) # For cleanup
        self.optimization_thread.start()
        logger.info("Optimization worker thread started.")


    def _handle_optimization_complete(self, optimized_elements_list: List[SiteElement]):
        logger.info("Optimization process reported complete by worker.")
        # The optimizer instance (self.optimizer) was modified in-place by the worker.
        # The optimized_elements_list is a copy of its final state.
        # We can use self.optimizer.elements directly.
        self.site_canvas.set_elements(self.optimizer.elements) # Update canvas
        self._update_layout_and_score() # Update routes and score display
        self.element_properties_widget.set_element(None) # Clear selection in properties panel
        self._mark_unsaved_changes()
        self.main_status_bar.showMessage("Optimization complete. Layout updated.")
        QMessageBox.information(self, "Optimization Finished", "The layout optimization process has completed.")


    def _handle_optimization_failed(self, error_message: str):
        logger.error(f"Optimization process failed: {error_message}")
        self.main_status_bar.showMessage(f"Optimization failed: {error_message[:100]}...") # Show truncated error
        QMessageBox.critical(self, "Optimization Error",
                             f"The optimization process encountered an error:\n\n{error_message}\n\n"
                             "The layout may not be optimal. Please check logs for details.")
        # Controls are re-enabled in _handle_optimization_thread_finished


    def _handle_optimization_thread_finished(self):
        logger.debug("Optimization worker thread finished signal received.")
        self.optimization_control_widget.enable_controls(True) # Re-enable controls
        if self.optimization_thread: # Ensure it's not None before trying to delete
            self.optimization_thread.deleteLater() # Schedule for deletion
            self.optimization_thread = None
        # Status bar message would have been set by _handle_optimization_complete or _failed.
        # If neither was called but thread finished (e.g. early exit), reset progress.
        if self.optimization_control_widget.progress_bar.value() < 100 and "failed" not in self.main_status_bar.currentMessage().lower():
             self.optimization_control_widget.set_progress(100) # Ensure bar shows complete if not failed.
             self.main_status_bar.showMessage("Optimization process terminated.")


    def _update_layout_and_score(self):
        """Central method to update routes on canvas and refresh score display."""
        if not self.optimizer:
            logger.debug("_update_layout_and_score called with no optimizer. Skipping.")
            return

        # This updates self.optimizer.vehicle_routes, self.optimizer.pedestrian_routes
        # and also updates the obstruction grid in path_planner.
        self.optimizer.update_routes()
        self.site_canvas.set_routes(self.optimizer.vehicle_routes, self.optimizer.pedestrian_routes)
        # self.site_canvas.set_elements(self.optimizer.elements) # Ensure canvas has latest elements if modified elsewhere
        self.site_canvas.update() # Force redraw of canvas

        # Update score and path metric labels
        # This also calls update_routes if the optimizer deems it necessary, but we called it above.
        current_score = self.optimizer.evaluate_layout()
        self.score_label.setText(f"<b>Score:</b> {current_score:,.0f}") # Formatted score

        if self.optimizer.path_planner:
            veh_len = self.optimizer.path_planner.get_total_path_length('vehicle')
            ped_len = self.optimizer.path_planner.get_total_path_length('pedestrian')
            interf = self.optimizer.path_planner.calculate_route_interference()
            self.vehicle_path_label.setText(f"Veh. Paths: {veh_len:,.0f}m")
            self.ped_path_label.setText(f"Ped. Paths: {ped_len:,.0f}m")
            self.interference_label.setText(f"Interference: {interf:,.0f}")
        else:
            self.vehicle_path_label.setText("Veh. Paths: N/A")
            self.ped_path_label.setText("Ped. Paths: N/A")
            self.interference_label.setText("Interference: N/A")
        # logger.debug(f"Layout and score updated. Current Score: {current_score:.2f}")


    # --- Help Dialogs & About ---
    def _show_about_dialog(self):
            current_year = QDate.currentDate().year()
            about_text = (
                f"<h3>{APPLICATION_NAME}</h3>"
                f"<p>Version: {APP_VERSION}</p>"
                "<p>A sophisticated GUI application for optimizing construction site facilities layout.</p>"
                "<p>Developed by: <b>Hamoon Soleimani</b></p>" # ADDED/MODIFIED
                "<p><b>Features:</b></p>"
                "<ul>"
                "<li>Interactive site element placement and editing.</li>"
                "<li>Automated layout optimization using Simulated Annealing.</li>"
                "<li>Configurable space requirements and layout constraints.</li>"
                "<li>Path planning for vehicle and pedestrian routes.</li>"
                "<li>Project saving and loading capabilities.</li>"
                "</ul>"
                # UPDATED COPYRIGHT LINE
                f"<p>Copyright © {current_year} Hamoon Soleimani. All rights reserved.</p>"
                "<p>This software is provided 'as-is', without any express or implied warranty.</p>" # Optional legal-like disclaimer
                "<p>Further contact: hamoon.s2@gmail.com</p>" # Optional
            )
            QMessageBox.about(self, f"About {APPLICATION_NAME}", about_text)

    def _show_usage_guide(self):
        guide_text = (
            "<h4>Quick Start Guide:</h4>"
            "<ol>"
            "<li><b>Configure Site:</b> Use the 'Site & Building Configuration' panel to set the overall plot width, height, and the ratio of area dedicated to the main building.</li>"
            "<li><b>Define Needs:</b> In the 'Space Requirements' tab, specify the area needed for each type of temporary facility. For offices and dormitories, you can base this on personnel counts.</li>"
            "<li><b>Set Constraints:</b> In the 'Layout Constraints' tab, adjust parameters like minimum safety distances between facilities, crane reach, and path widths.</li>"
            "<li><b>Initialize Layout:</b> Click the 'Initialize / Reset Layout' button. This will place elements based on your configurations. You can also load an existing project from the File menu.</li>"
            "<li><b>Manual Adjustments (Optional):</b>"
            "<ul><li>Click and drag elements on the canvas to move them.</li>"
            "<li>Select an element to edit its precise properties (position, size, rotation, color) in the 'Element Properties' panel on the left.</li>"
            "<li>Use Shift+Drag or Middle Mouse Button to pan the canvas. Use the mouse wheel to zoom.</li></ul></li>"
            "<li><b>Optimize Layout:</b>"
            "<ul><li>Go to the 'Optimization Controls' tab.</li>"
            "<li>Adjust algorithm parameters like Iterations, Start Temperature, and Cooling Rate if needed.</li>"
            "<li>Select an 'Optimization Strategy' that best fits your project's priorities (e.g., Safety, Efficiency).</li>"
            "<li>Click the 'Run Optimization' button (either in this tab or the main button below the left panel).</li>"
            "<li>Monitor progress on the progress bar. The layout will update upon completion.</li></ul></li>"
            "<li><b>Review & Iterate:</b> Examine the optimized layout and the metrics (Score, Path Lengths). You can make further manual adjustments or re-run the optimization with different parameters or strategies.</li>"
            "<li><b>Save/Export:</b> Use the File menu to save your project (`.slyp` file) or export the current layout as an image.</li>"
            "</ol>"
            "<p><b>Display Options:</b> Use the 'Display Options' tab or the View menu to toggle visibility of grid lines, element labels, dimensions, etc., on the canvas.</p>"
        )
        # Use a QDialog for better formatting if needed, or a QMessageBox for simplicity
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Usage Guide")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setTextFormat(Qt.RichText) # Allow HTML formatting
        msg_box.setText(guide_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


    # --- Window State Management & Closing ---
    def _load_window_settings(self):
        """Loads window geometry (size, position) from QSettings."""
        logger.debug("Loading window settings...")
        geometry = self.settings.value("windowGeometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
        splitter_state = self.settings.value("splitterSizes")
        if splitter_state:
            self.splitter.restoreState(splitter_state)


    def _save_window_settings(self):
        """Saves current window geometry to QSettings."""
        logger.debug("Saving window settings...")
        self.settings.setValue("windowGeometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("splitterSizes", self.splitter.saveState())


    def closeEvent(self, event: QCloseEvent):
        """Handles the application close event. Prompts to save unsaved changes."""
        logger.info("Application close event triggered.")
        if self.optimization_thread and self.optimization_thread.isRunning():
            reply = QMessageBox.question(self, "Optimization in Progress",
                                         "An optimization process is currently running. Exiting now will terminate it. Are you sure you want to exit?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore()
                logger.info("Application close cancelled because optimization is running.")
                return
            else: # User chose to exit despite running optimization
                self.optimization_thread.requestInterruption() # Politely ask thread to stop
                if not self.optimization_thread.wait(1000): # Wait up to 1 sec
                    logger.warning("Optimization thread did not terminate gracefully. Forcing termination.")
                    self.optimization_thread.terminate() # Force if necessary
                    self.optimization_thread.wait() # Wait for termination
                logger.info("Optimization thread stopped due to application close.")


        if self._confirm_discard_changes():
            self._save_window_settings() # Save window state before closing
            event.accept()
            logger.info("Application closing.")
        else:
            event.ignore()
            logger.info("Application close cancelled by user due to unsaved changes or cancel action.")

# Main execution block (if this file is run directly)
def main():
    app = QApplication(sys.argv)
    # Apply a style (optional, 'Fusion' is good cross-platform)
    app.setStyle('Fusion')

    # --- Dark Theme Palette ---
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, DARK_THEME_TEXT_COLOR)
    dark_palette.setColor(QPalette.Base, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipText, DARK_THEME_TEXT_COLOR)
    dark_palette.setColor(QPalette.Text, DARK_THEME_TEXT_COLOR)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, DARK_THEME_TEXT_COLOR)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))

    app.setPalette(dark_palette)
    app.setStyleSheet(
        "QToolTip { color: #ffffff; background-color: #2a2a2a; border: 1px solid #32414B; }"
        "QGroupBox { background-color: transparent; border: 1px solid " + DARK_THEME_BORDER_COLOR.name() + "; margin-top: 2ex; }"
        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; background-color: " + DARK_THEME_GROUPBOX_BG_COLOR.name() + "; color:" + DARK_THEME_TEXT_COLOR.name() + ";}"
        "QTabWidget::pane { border: 1px solid " + DARK_THEME_BORDER_COLOR.name() + "; }"
        "QTabBar::tab { background: " + DARK_THEME_GROUPBOX_BG_COLOR.name() + "; color: " + DARK_THEME_TEXT_COLOR.name() + "; padding: 5px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px;}"
        "QTabBar::tab:selected { background: " + QColor(66, 66, 66).name() + "; }" # Slightly lighter for selected tab
        "QTabBar::tab:hover { background: " + QColor(77, 77, 77).name() + "; }"
        "QScrollArea { border: 1px solid " + DARK_THEME_BORDER_COLOR.name() + "; background-color: " + QColor(42, 42, 42).name() + ";}"
        "QScrollBar:vertical { border: 1px solid " + DARK_THEME_BORDER_COLOR.name() + "; background: " + QColor(53, 53, 53).name() + "; width: 10px; margin: 0px 0px 0px 0px; }"
        "QScrollBar::handle:vertical { background: " + QColor(80,80,80).name() + "; min-height: 20px; border-radius: 5px; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background: none; border: none; height: 0px; }"
        "QScrollBar:horizontal { border: 1px solid " + DARK_THEME_BORDER_COLOR.name() + "; background: " + QColor(53, 53, 53).name() + "; height: 10px; margin: 0px 0px 0px 0px; }"
        "QScrollBar::handle:horizontal { background: " + QColor(80,80,80).name() + "; min-width: 20px; border-radius: 5px; }"
        "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { background: none; border: none; width: 0px; }"
        "QPushButton { background-color: " + DARK_THEME_BUTTON_BG_COLOR.name() + "; border: 1px solid " + DARK_THEME_BORDER_COLOR.name() + "; padding: 5px; border-radius: 3px; }"
        "QPushButton:hover { background-color: " + DARK_THEME_BUTTON_HOVER_BG_COLOR.name() + "; }"
        "QPushButton:pressed { background-color: " + DARK_THEME_BUTTON_PRESSED_BG_COLOR.name() + "; }"
        "QPushButton:disabled { background-color: " + QColor(60,60,60).name() + "; color: " + QColor(127,127,127).name() + "; }"
        "QSpinBox, QDoubleSpinBox { padding-right: 15px; border: 1px solid " + DARK_THEME_BORDER_COLOR.name() + "; border-radius: 3px; background-color: " + QColor(42,42,42).name() + "; }"

    )

    main_window = ConstructionSiteOptimizerApp()
    main_window.show()
    # Issue 3: Default "Fit View" - Call after window is shown and layout potentially initialized
    QTimer.singleShot(200, lambda: main_window.site_canvas.fit_plot_to_view())
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
