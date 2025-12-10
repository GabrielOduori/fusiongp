"""
Tests for data loading and preprocessing.

Run with: pytest tests/test_data.py -v
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from fusiongp.data.loader import DataLoader, FusionData, FusionDataset
from fusiongp.data.preprocessor import DataPreprocessor, Scalers


@pytest.fixture
def sample_csv():
    """Create a temporary CSV file with sample data."""
    n_samples = 100
    
    data = {
        'grid_id': np.repeat(np.arange(10), 10),
        'latitude': np.random.uniform(40.7, 40.8, n_samples),
        'longitude': np.random.uniform(-74.1, -74.0, n_samples),
        'timestamp': np.tile(pd.date_range('2024-01-01', periods=10, freq='D'), 10),
        'satellite_values': np.random.uniform(10, 30, n_samples),
        'low_cost_data': np.random.uniform(8, 35, n_samples),
        'epa_no2': np.random.uniform(12, 28, n_samples),
    }
    
    # Add some missing values
    data['satellite_values'][::5] = np.nan
    data['epa_no2'][::3] = np.nan
    
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        return f.name


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_load_basic(self, sample_csv):
        """Test basic data loading."""
        loader = DataLoader(sample_csv)
        data = loader.load()
        
        assert isinstance(data, FusionData)
        assert data.n_observations == 100
        assert data.coords.shape == (100, 2)
        assert len(data.sources) == 3
    
    def test_load_with_missing(self, sample_csv):
        """Test that missing values are handled correctly."""
        loader = DataLoader(sample_csv)
        data = loader.load()
        
        # Check masks
        assert data.source_masks['satellite'].sum() < 100  # Some missing
        assert data.source_masks['epa'].sum() < 100
        assert data.source_masks['low_cost'].sum() == 100  # None missing
    
    def test_file_not_found(self):
        """Test error on missing file."""
        loader = DataLoader("nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_summary(self, sample_csv):
        """Test summary generation."""
        loader = DataLoader(sample_csv)
        data = loader.load()
        summary = data.summary()
        
        assert "FusionData Summary" in summary
        assert "100" in summary  # Total observations


class TestFusionData:
    """Tests for FusionData class."""
    
    def test_get_valid_observations(self, sample_csv):
        """Test extraction of valid observations."""
        loader = DataLoader(sample_csv)
        data = loader.load()
        
        coords, times, values = data.get_valid_observations('low_cost')
        
        assert len(coords) == len(values)
        assert not np.any(np.isnan(values))
    
    def test_n_observations(self, sample_csv):
        """Test n_observations property."""
        loader = DataLoader(sample_csv)
        data = loader.load()
        
        assert data.n_observations == len(data.coords)


class TestFusionDataset:
    """Tests for FusionDataset (PyTorch Dataset)."""
    
    def test_dataset_creation(self, sample_csv):
        """Test dataset creation."""
        loader = DataLoader(sample_csv)
        data = loader.load()
        dataset = FusionDataset(data)
        
        assert len(dataset) == 100
    
    def test_getitem(self, sample_csv):
        """Test dataset indexing."""
        loader = DataLoader(sample_csv)
        data = loader.load()
        dataset = FusionDataset(data)
        
        coords, timestamp, obs, masks, idx = dataset[0]
        
        assert coords.shape == (2,)
        assert timestamp.dim() == 0
        assert obs.shape == (3,)  # 3 sources
        assert masks.shape == (3,)
    
    def test_get_input_tensor(self, sample_csv):
        """Test combined input tensor."""
        loader = DataLoader(sample_csv)
        data = loader.load()
        dataset = FusionDataset(data)
        
        x = dataset.get_input_tensor()
        
        assert x.shape == (100, 3)  # lat, lon, time


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    def test_fit_transform(self, sample_csv):
        """Test fit_transform method."""
        loader = DataLoader(sample_csv)
        data = loader.load()
        
        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.fit_transform(data)
        
        # Check split sizes
        total = train.n_observations + val.n_observations + test.n_observations
        assert total == data.n_observations
        
        # Check normalization
        assert train.coords.min() >= 0
        assert train.coords.max() <= 1
    
    def test_normalization(self, sample_csv):
        """Test coordinate normalization."""
        loader = DataLoader(sample_csv)
        data = loader.load()
        
        preprocessor = DataPreprocessor(normalize_coords=True)
        preprocessor.fit(data)
        transformed = preprocessor.transform(data)
        
        assert transformed.coords.min() >= 0
        assert transformed.coords.max() <= 1
        assert transformed.timestamps.min() >= 0
        assert transformed.timestamps.max() <= 1
    
    def test_split_ratios(self, sample_csv):
        """Test custom split ratios."""
        loader = DataLoader(sample_csv)
        data = loader.load()
        
        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.fit_transform(
            data, 
            train_ratio=0.6, 
            val_ratio=0.2, 
            test_ratio=0.2
        )
        
        # Approximate check (exact depends on rounding)
        assert train.n_observations > val.n_observations
        assert train.n_observations > test.n_observations


class TestScalers:
    """Tests for Scalers class."""
    
    def test_inverse_transform_coords(self):
        """Test coordinate inverse transformation."""
        scalers = Scalers(
            coord_min=np.array([40.0, -74.0]),
            coord_max=np.array([41.0, -73.0]),
            time_min=0.0,
            time_max=10.0,
        )
        
        # Normalized coords
        normalized = np.array([[0.5, 0.5]])
        
        original = scalers.inverse_transform_coords(normalized)
        
        np.testing.assert_array_almost_equal(original, [[40.5, -73.5]])
    
    def test_inverse_transform_time(self):
        """Test time inverse transformation."""
        scalers = Scalers(
            coord_min=np.array([0.0, 0.0]),
            coord_max=np.array([1.0, 1.0]),
            time_min=0.0,
            time_max=30.0,
        )
        
        normalized = np.array([0.0, 0.5, 1.0])
        original = scalers.inverse_transform_time(normalized)
        
        np.testing.assert_array_almost_equal(original, [0.0, 15.0, 30.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
