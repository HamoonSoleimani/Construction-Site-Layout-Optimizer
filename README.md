# Construction Site Layout Optimizer

A sophisticated GUI application designed for optimizing the layout of temporary facilities on a construction site. It provides tools for defining site parameters, facility requirements, constraints, and uses a Simulated Annealing algorithm to find efficient layouts.

![image](https://github.com/user-attachments/assets/7aae1ccb-b1e0-487e-9b72-acc01dd1549b)

## Features

*   **Graphical User Interface:** Intuitive and interactive interface built with PyQt5 for managing all aspects of the site layout.
*   **Site Configuration:** Define overall plot dimensions (width, height) and the proportion of area allocated to the main building.
*   **Customizable Site Elements:**
    *   Define various types of temporary facilities (e.g., offices, storage, workshops, accommodation).
    *   Each element type has default properties: color, movability, placement priority, resizability, aspect ratio, or fixed dimensions.
    *   Support for elements that can be placed inside the main building.
*   **Space Requirements Definition:**
    *   Specify total area requirements for each facility.
    *   For personnel-based facilities (offices, dormitories), calculate area based on the number of people and area per person, or set total area manually.
    *   View suggested dimensions (Width × Height) based on area and typical aspect ratios.
*   **Layout Constraints Management:**
    *   Define critical constraints: minimum/maximum distances between specific facilities (e.g., fuel tank to dormitory).
    *   Set operational parameters like tower crane reach.
    *   Configure general safety buffers and path widths.
*   **Interactive 2D Canvas:**
    *   Visualize the site layout, including elements, the main building, and plot boundaries.
    *   Pan and zoom the view for detailed inspection.
    *   Select elements by clicking on them.
    *   Manually move selected elements using drag-and-drop or keyboard arrow keys.
    *   Rotate selected movable elements.
*   **Element Properties Editing:**
    *   Detailed panel to view and edit properties of the selected element:
        *   Position (X, Y), Dimensions (Width, Height), Rotation.
        *   Display of calculated area.
        *   Toggle movability for optimization.
        *   Change element color.
    *   Actions to duplicate or delete elements (Building element is protected).
*   **Automated Layout Optimization:**
    *   Employs a Simulated Annealing (SA) algorithm to find an optimized layout.
    *   Configurable SA parameters: iterations, start temperature, cooling rate, minimum temperature, reheat factor, and reheat threshold.
    *   Choose from different optimization strategies (e.g., Balanced, Safety Priority, Efficiency Priority, Comfort & Spacing) that adjust the importance of various scoring criteria.
    *   Optimization process runs in a background thread to keep the GUI responsive, with a progress bar.
*   **Path Planning & Visualization:**
    *   Automatic generation of indicative vehicle and pedestrian routes between key facilities and site entry.
    *   A* algorithm used for pathfinding on a grid, considering obstacles (site elements) and safety buffers.
    *   Paths are smoothed using the Ramer-Douglas-Peucker algorithm.
    *   Visual display of routes on the canvas.
*   **Display Options:**
    *   Toggle visibility of: grid lines, element dimensions, element labels/names, crane operational radius, and markers for elements placed inside the main building.
    *   Dynamic scale bar and mouse cursor coordinates (site units) displayed on the canvas.
*   **Layout Evaluation & Metrics:**
    *   Comprehensive scoring system evaluates layout quality based on:
        *   Overlaps and out-of-bounds violations.
        *   Adherence to defined constraints (distances, buffers, crane coverage).
        *   Proximity/grouping of related facilities (e.g., offices).
        *   Accessibility of key facilities.
        *   Total lengths of vehicle and pedestrian paths.
        *   Interference between vehicle and pedestrian routes.
    *   Metrics (Score, Path Lengths, Interference) are displayed and updated.
*   **Project Management:**
    *   Create new projects with default settings.
    *   Save and load projects in a `.slyp` file (JSON format).
    *   Saves all site configurations, element states, requirements, and constraints.
    *   Remembers last opened/saved project directory.
*   **Export:** Export the current site layout visualization as a PNG or JPG image.
*   **Cross-Platform:** Developed with Python and PyQt5, aiming for cross-platform compatibility.

## Core Technologies

*   **Python 3.x**
*   **PyQt5:** For the graphical user interface (QtWidgets, QtCore, QtGui).
*   **NumPy:** For numerical operations, especially in path planning and grid management.
*   **Matplotlib:** (Backend is imported, potentially for future plotting extensions or if specific components use it).

## Installation

1.  **Prerequisites:**
    *   Python 3 (e.g., Python 3.7 or newer recommended).
    *   `pip` (Python package installer).

2.  **Install Dependencies:**
    Open a terminal or command prompt and run:
    ```bash
    pip install PyQt5 numpy matplotlib
    ```

3.  **Run the Application:**
    Navigate to the directory containing the application script (e.g., `main_app_script.py`) and run:
    ```bash
    python 'Construction Site Layout Optimizer.py'
    ```
    *(Replace `main_app_script.py` with the actual name of the main Python file.)*

## Usage Overview

1.  **Launch Application:** Run the main script.
2.  **Project Setup:**
    *   Go to **File > New Project** or **File > Open Project...** to load an existing `.slyp` file.
    *   Configure overall site dimensions and main building area ratio in the **Site & Building Configuration** panel.
3.  **Define Facility Needs:**
    *   In the **Space Requirements** tab, specify the area for each temporary facility. Use personnel counts for offices/dormitories if desired.
4.  **Set Layout Rules:**
    *   In the **Layout Constraints** tab, adjust safety distances, crane reach, path widths, and other operational rules.
5.  **Initialize Layout:**
    *   Click the **Initialize / Reset Layout** button (or **Edit > Initialize Current Layout**). This places elements based on your configurations.
6.  **Manual Adjustments (Optional):**
    *   Click and drag elements on the canvas to move them.
    *   Select an element to see its details in the **Element Properties** panel (left side) and edit its position, size, rotation, or color.
    *   Use **Shift + Left-Click + Drag** or the **Middle Mouse Button + Drag** to pan the canvas. Use the mouse wheel to zoom.
7.  **Automated Optimization:**
    *   Go to the **Optimization Controls** tab.
    *   Adjust Simulated Annealing parameters (Iterations, Temperature, etc.) if needed.
    *   Select an **Optimization Strategy** (e.g., Balanced, Safety Priority).
    *   Click the **Run Optimization** button.
    *   Monitor the progress bar. The layout will update upon completion.
8.  **Review and Iterate:**
    *   Examine the optimized layout and the metrics displayed (Score, Path Lengths, Interference).
    *   Make further manual adjustments or re-run the optimization with different parameters or strategies if desired.
9.  **Save/Export:**
    *   Use **File > Save Project** or **File > Save Project As...** to save your work.
    *   Use **File > Export Layout as Image...** to save a snapshot of the canvas.
10. **Display Customization:**
    *   Use the **Display Options** tab or the **View** menu to toggle the visibility of various visual aids on the canvas (grid, labels, dimensions, etc.).

## Key Components (Code Structure Overview)

*   `SiteElement`: Represents individual facilities with properties like position, size, color, and behavior rules.
*   `PathPlanner`: Employs an A* algorithm to find and manage vehicle/pedestrian routes, considering obstacles and constraints. It generates an obstruction grid and smooths paths.
*   `OptimizationEngine`: Orchestrates the layout process using Simulated Annealing. It creates initial layouts, evaluates layout quality against defined objectives and constraints, and iteratively seeks better arrangements.
*   `SiteCanvas`: The interactive 2D visualization widget. It handles drawing of the site, elements, routes, grid, and supports user interactions like panning, zooming, element selection, and movement.
*   `ElementPropertiesWidget`: A panel for viewing and editing the detailed properties of a currently selected `SiteElement`.
*   `SpaceRequirementsWidget`: Allows users to define the area needed for each type of facility, either directly or based on personnel counts.
*   `ConstraintsWidget`: Provides UI for users to set various layout constraints (e.g., minimum distances, crane reach, safety buffers).
*   `OptimizationControlWidget`: Allows users to configure parameters for the Simulated Annealing algorithm and choose an overall optimization strategy (which adjusts scoring weights).
*   `DisplayOptionsWidget`: A panel with checkboxes to control the visibility of different visual elements on the `SiteCanvas`.
*   `OptimizationWorker (QThread)`: Executes the computationally intensive optimization process in a separate thread to keep the GUI responsive.
*   `ConstructionSiteOptimizerApp (QMainWindow)`: The main application window. It integrates all UI components, manages project data (loading/saving), and orchestrates the interaction between the UI, the `OptimizationEngine`, and other modules.

## Configuration

### Default Element Definitions
The application uses pre-defined defaults for site elements, their properties, space requirements, and layout constraints. These are typically found as global Python dictionaries within the source code (e.g., `SITE_ELEMENTS`, `DEFAULT_SPACE_REQUIREMENTS`, `DEFAULT_CONSTRAINTS`, `DEFAULT_PERSONNEL_CONFIG`). These defaults are used when creating a new project or resetting configurations.

`SITE_ELEMENTS` defines for each facility type:
*   `color`: Default display color.
*   `movable`: Whether it can be moved by the optimizer or user.
*   `priority`: Influences initial placement order.
*   `placeable_inside_building`: If true, this element can be notionally placed within the main building's footprint.
*   `resizable`: If true, its dimensions can change.
*   `aspect_ratio`: Preferred width-to-height ratio if resizable.
*   `fixed_width`, `fixed_height`: Specific dimensions if not resizable.

### Element Properties (SiteElement class attributes)
Each `SiteElement` instance in the layout stores:
*   `name`: The type of the facility (e.g., 'Contractor Office').
*   `x`, `y`: Top-left position in world coordinates (meters).
*   `width`, `height`: Dimensions in meters.
*   `rotation`: Angle in degrees.
*   `color`: `QColor` for display.
*   `movable`: Boolean indicating if it can be moved.
*   `selected`: Boolean indicating if it's currently selected on the canvas.
*   `priority`, `placeable_inside_building`: (Inherited from type definition).
*   `is_placed_inside_building`: Boolean status.
*   `original_external_x, y, width, height, rotation`: Stores original geometry if temporarily moved inside a building.

### Project Files (`.slyp`)
*   Projects are saved in `.slyp` files, which are JSON-formatted text files.
*   These files store:
    *   Application version at the time of saving.
    *   Save date.
    *   Plot dimensions (width, height) and the main building's area ratio.
    *   All configured layout constraints.
    *   All defined space requirements, including any personnel-based configurations.
    *   A list of all site elements, including their name, position, dimensions, rotation, color, movability, selection state, and internal placement status.

### Application Settings
*   Window size, position, and the state of UI splitters are saved automatically using `QSettings`.
    *   These settings are stored based on `ORGANIZATION_NAME = "HamoonSoleimani"` and `APPLICATION_NAME = "ConstructionSiteOptimizer"`.
*   The file paths to the last opened/saved project and the last image export directory are also remembered across sessions.

## Main Interface Components

### Left Panel
The left panel typically houses configuration and control widgets:

#### Site & Building Configuration
*   Inputs for **Plot Width (m)**, **Plot Height (m)**, and **Building Area Ratio**.

#### Tabbed Settings
A `QTabWidget` organizes more detailed settings:
*   **Space Requirements:** Configure area for each facility type (see `SpaceRequirementsWidget`).
*   **Layout Constraints:** Adjust safety distances, operational limits, etc. (see `ConstraintsWidget`).
*   **Optimization Controls:** Set Simulated Annealing parameters and choose an optimization strategy (see `OptimizationControlWidget`).
*   **Display Options:** Toggle visibility of canvas aids like grid, labels, dimensions (see `DisplayOptionsWidget`).

#### Element Properties
*   A dedicated panel displays and allows editing of the currently selected `SiteElement`'s properties (position, size, rotation, color, movability). Also includes buttons for duplicating or deleting the element.

#### Action Buttons
*   **Initialize / Reset Layout:** Generates an initial layout or resets the current one based on settings.
*   **Run Optimization:** Starts the automated layout optimization (mirrors the button in "Optimization Controls" tab).

### Right Panel
The right panel is primarily for visualization:

#### Site Layout Visualization (Canvas)
*   The main interactive `SiteCanvas` where the 2D site layout is displayed and can be manipulated.
*   Includes a small header with the canvas title and a "Fit View" button.

#### Metrics Bar
*   Located below the canvas, this bar displays:
    *   Current layout **Score**.
    *   Total **Vehicle Path Length**.
    *   Total **Pedestrian Path Length**.
    *   A **Route Interference** score.
    *   General tips or status messages.

## Menu Bar

### File Menu
*   **New Project:** Creates a new, empty project with default settings.
*   **Open Project...:** Loads an existing project from a `.slyp` file.
*   **Save Project:** Saves the current project. If unsaved, prompts for a filename.
*   **Save Project As...:** Saves the current project to a new `.slyp` file.
*   **Export Layout as Image...:** Saves the current canvas view as a PNG or JPG image.
*   **Exit:** Closes the application (prompts to save unsaved changes).

### Edit Menu
*   **Initialize Current Layout:** Clears existing elements (except Building) and generates a new initial layout.
*   **Run Optimization:** Starts the automated optimization process.

### View Menu
*   Toggle visibility of:
    *   **Show Grid**
    *   **Show Element Labels**
    *   **Show Element Dimensions**
    *   **Show Crane Radius**
    *   **Show Internal Element Markers**
*   **Fit Plot to View:** Adjusts zoom and pan to show the entire site plot.

### Help Menu
*   **About...:** Displays information about the application (version, developer).
*   **Usage Guide...:** Shows a quick start guide on how to use the application.

## About

*   **Application Name:** Construction Site Layout Optimizer
*   **Version:** 1.2.0
*   **Description:** A sophisticated GUI application for optimizing construction site facilities layout.
*   **Developer:** Hamoon Soleimani
*   **Contact:** hamoon.s2@gmail.com

## License

Copyright © 2024 Hamoon Soleimani. All rights reserved.

This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software. Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:
1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

