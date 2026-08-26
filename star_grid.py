
"""
Generate a grid of objects with random magnitudes in a square area.

Usage: python star_grid.py <area> <output_file.feather>
Example: python star_grid.py 100.0 objects.feather
"""

import numpy as np
import pandas as pd
import sys
import math

# ============================================================================
# CONFIGURATION
# ============================================================================
OBJECT_DENSITY = 11211.07  # objects per square degree (adjust as needed) 
#11211.07 -> 34 arcsec spacing between adj. stars

# Magnitude range
MAG_MIN = 19.0
MAG_MAX = 23.0

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_grid_objects(area, output_file):
    """
    Generate a grid of objects tiling a square area.
    
    Args:
        area (float): Total area in square degrees
        output_file (str): Path to output .feather file
    """
    
    side_length = math.sqrt(area)
    n_objects = int(area * OBJECT_DENSITY)
    n_per_side = int(math.sqrt(n_objects))  # approximate square grid
    tile_size = side_length / n_per_side
    
    print(f"Area: {area} square degrees")
    print(f"Side length: {side_length:.3f} degrees")
    print(f"Object density: {OBJECT_DENSITY} per square degree")
    print(f"Total objects: {n_objects}")
    print(f"Grid: {n_per_side} × {n_per_side}")
    print(f"Tile size: {tile_size:.6f} degrees")
    
    # Generate grid coordinates (centers of tiles)
    indices = []
    ras = []
    decs = []
    rs = []
    
    index = 0
    for i in range(n_per_side):
        for j in range(n_per_side):
            ra = (i + 0.5) * tile_size  # tile center
            dec = (j + 0.5) * tile_size
            r = np.random.uniform(MAG_MIN, MAG_MAX)

            indices.append(index)
            ras.append(ra)
            decs.append(dec)
            rs.append(r)
            
            index += 1
    
    df = pd.DataFrame({
        'index': indices,
        'ra': ras,
        'dec': decs,
        'r': rs
    })
    df.to_feather(output_file)
    
    print(f"\nSaved {len(df)} objects to {output_file}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nRA range: [{df['ra'].min():.6f}, {df['ra'].max():.6f}]")
    print(f"Dec range: [{df['dec'].min():.6f}, {df['dec'].max():.6f}]")
    print(f"r range: [{df['r'].min():.3f}, {df['r'].max():.3f}]")
    
    return df


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_grid.py <area> <output_file.feather>")
        print("Example: python generate_grid.py 100.0 objects.feather")
        print(f"\nCurrent object density: {OBJECT_DENSITY} per square degree")
        sys.exit(1)
    
    try:
        area = float(sys.argv[1])
        output_file = sys.argv[2]
        
        if area <= 0:
            print("Error: Area must be positive")
            sys.exit(1)
        
        if not output_file.endswith('.feather'):
            print("Warning: Output file should have .feather extension")
        
        # Set random seed for reproducibility (optional)
        np.random.seed(42)
        
        generate_grid_objects(area, output_file)
        
    except ValueError:
        print("Error: Area must be a valid number")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
