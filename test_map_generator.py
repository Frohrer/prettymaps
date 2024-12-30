import pytest
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
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
def sf_coordinates():
    """Return the test coordinates for San Francisco."""
    return (37.79072260209851, -122.39110457256962)

@pytest.mark.usefixtures("test_output_dir")
def test_plot_sf_map(test_output_dir, sf_coordinates):
    """Test plotting a simple map of San Francisco."""
    output_path = test_output_dir / "test_sf.svg"
    
    try:
        # Clear any existing plots
        plt.clf()
        
        # Generate the plot using exact coordinates
        plot(
            sf_coordinates,
            radius=200,
            circle=True,
            figsize=(12, 12)
        )
        
        # Save the current figure
        plt.savefig(str(output_path), format='svg', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Assertions
        assert output_path.exists(), f"Output file not found at {output_path}"
        assert output_path.stat().st_size > 0, "Generated file is empty"
        
        logger.info(f"Successfully generated map at {output_path}")
        logger.info(f"File size: {output_path.stat().st_size} bytes")
        
    except Exception as e:
        logger.error(f"Error during map generation: {str(e)}")
        plt.close()
        raise