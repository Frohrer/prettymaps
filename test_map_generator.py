import pytest
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import logging
from pathlib import Path
from map_generator import generate_map

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
        'location': 'San Francisco, CA',
        'radius': 200,
        'circle': True
    }

# Tests
def test_generate_default_map(test_output_dir, default_map_params):
    """Test generating a map with default style."""
    output_path = test_output_dir / "test_default.svg"
    
    success = generate_map(
        **default_map_params,
        output_path=str(output_path)
    )
    
    # Assertions
    assert success, "Map generation failed"
    assert output_path.exists(), f"Output file not found at {output_path}"
    assert output_path.stat().st_size > 0, "Generated file is empty"
    
    logger.info(f"Successfully generated map at {output_path}")
    logger.info(f"File size: {output_path.stat().st_size} bytes")

def test_generate_map_invalid_location(test_output_dir):
    """Test map generation with invalid location."""
    output_path = test_output_dir / "test_invalid.svg"
    
    with pytest.raises(Exception):
        generate_map(
            location='ThisIsNotARealLocation12345',
            radius=200,
            circle=True,
            output_path=str(output_path)
        )

def test_generate_map_different_radius(test_output_dir, default_map_params):
    """Test map generation with different radius values."""
    params = default_map_params.copy()
    
    for radius in [100, 500]:
        output_path = test_output_dir / f"test_radius_{radius}.svg"
        params['radius'] = radius
        
        success = generate_map(
            **params,
            output_path=str(output_path)
        )
        
        assert success, f"Map generation failed for radius {radius}"
        assert output_path.exists(), f"Output file not found at {output_path}"
        assert output_path.stat().st_size > 0, f"Generated file is empty for radius {radius}"