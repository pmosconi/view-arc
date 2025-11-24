# View Arc Obstacle Detection

Finds the obstacle with largest visible angular coverage within a field-of-view arc from a viewer point.

## Installation

```bash
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

For development:
```bash
uv pip install -e ".[dev]"
```

## Usage

```python
import numpy as np
from view_arc import find_largest_obstacle

# Viewer position in image coordinates
viewer = np.array([100.0, 100.0], dtype=np.float32)

# View direction as unit vector (e.g., pointing UP in image space)
view_direction = np.array([0.0, 1.0], dtype=np.float32)

# Obstacle contours in image coordinates
contours = [
    np.array([[90, 150], [110, 150], [100, 170]], dtype=np.float32),
    np.array([[80, 200], [120, 200], [100, 230]], dtype=np.float32)
]

result = find_largest_obstacle(viewer, view_direction, 30.0, 150.0, contours)
print(f"Winner: obstacle {result.obstacle_id}, coverage: {result.angular_coverage:.2f} rad")
```

## Algorithm

See `docs/obstacle_arc_spec.md` for detailed algorithm specification.
