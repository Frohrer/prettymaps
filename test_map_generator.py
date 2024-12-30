import pytest
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import logging
from pathlib import Path
from prettymaps.draw import plot

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Fixtures
@pytest.fixture
def test_output_dir():
    """Create and return a temporary test output directory."""
    test_dir = Path("static/test_maps")
    test_dir.mkdir(exist_ok=True, parents=True)
    return test_dir

@pytest.fixture
def default_map_params():
    """Return default parameters for map generation."""
    return {
        'address': 'San Francisco, CA',
        'radius': 200
    }

# Tests
def test_plot_default_map(test_output_dir, default_map_params):
    """Test plotting a map with default style."""
    output_path = test_output_dir / "test_default.svg"
    
    try:
        # Generate the plot
        fig, ax = plot(
            default_map_params['address'],
            radius=default_map_params['radius']
        )
        
        # Save the figure
        fig.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0)
        
        # Assertions
        assert output_path.exists(), f"Output file not found at {output_path}"
        assert output_path.stat().st_size > 0, "Generated file is empty"
        
        logger.info(f"Successfully generated map at {output_path}")
        logger.info(f"File size: {output_path.stat().st_size} bytes")
        
    except Exception as e:
        logger.error(f"Error during map generation: {str(e)}")
        raise

def test_plot_invalid_location(test_output_dir):
    """Test map generation with invalid location."""
    output_path = test_output_dir / "test_invalid.svg"
    
    with pytest.raises(Exception):
        fig, ax = plot(
            'ThisIsNotARealLocation12345',
            radius=200
        )
        fig.savefig(output_path, format='svg')

def test_plot_different_radius(test_output_dir, default_map_params):
    """Test map generation with different radius values."""
    address = default_map_params['address']
    
    for radius in [100, 500]:
        output_path = test_output_dir / f"test_radius_{radius}.svg"
        
        try:
            fig, ax = plot(
                address,
                radius=radius
            )
            fig.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0)
            
            assert output_path.exists(), f"Output file not found at {output_path}"
            assert output_path.stat().st_size > 0, f"Generated file is empty for radius {radius}"
            
            logger.info(f"Successfully generated map with radius {radius}")
            
        except Exception as e:
            logger.error(f"Error generating map with radius {radius}: {str(e)}")
            raise

def test_plot_with_custom_style(test_output_dir, default_map_params):
    """Test plotting with custom style parameters."""
    output_path = test_output_dir / "test_custom_style.svg"
    
    try:
        layers = {
            'perimeter': {},  # Empty dict for default perimeter settings
            'streets': {
                'custom_filter': '["highway"~"motorway|trunk|primary|secondary|tertiary|residential|service|unclassified"]',
                'width': {
                    'motorway': 5,
                    'trunk': 4,
                    'primary': 3,
                    'secondary': 2,
                    'tertiary': 1,
                    'residential': 0.5,
                    'service': 0.25,
                    'unclassified': 0.25
                }
            },
            'building': {'tags': {'building': True}},
            'water': {'tags': {'natural': ['water', 'bay']}},
            'green': {'tags': {'landuse': ['grass', 'park'], 'natural': ['wood']}}
        }
        
        fig, ax = plot(
            default_map_params['address'],
            radius=default_map_params['radius'],
            layers=layers
        )
        
        fig.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0)
        
        assert output_path.exists(), f"Output file not found at {output_path}"
        assert output_path.stat().st_size > 0, "Generated file is empty"
        
        logger.info(f"Successfully generated custom styled map at {output_path}")
        
    except Exception as e:
        logger.error(f"Error during custom style map generation: {str(e)}")
        raise

def test_plot_circular_perimeter(test_output_dir, default_map_params):
    """Test plotting with circular perimeter."""
    output_path = test_output_dir / "test_circular.svg"
    
    try:
        layers = {
            'perimeter': {'circle': True},  # Set circular perimeter in layers
            'streets': {},
            'building': {'tags': {'building': True}},
            'water': {'tags': {'natural': ['water', 'bay']}},
            'green': {'tags': {'landuse': ['grass', 'park'], 'natural': ['wood']}}
        }
        
        fig, ax = plot(
            default_map_params['address'],
            radius=default_map_params['radius'],
            layers=layers
        )
        
        fig.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0)
        
        assert output_path.exists(), f"Output file not found at {output_path}"
        assert output_path.stat().st_size > 0, "Generated file is empty"
        
        logger.info(f"Successfully generated circular perimeter map")
        
    except Exception as e:
        logger.error(f"Error during circular perimeter map generation: {str(e)}")
        raise