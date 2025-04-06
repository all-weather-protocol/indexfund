"""
Unit tests for the data_loading module.
"""

import io
import os
import tempfile
from datetime import datetime
from unittest.mock import patch

from core.data_loading import (
    align_data_timestamps,
    extract_current_data,
    filter_data_by_start_date,
    load_fear_greed_index,
    load_historical_data,
    load_token_data,
    process_fear_greed_data,
)

# Define path to the test dataset
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
FEAR_GREED_PATH = os.path.join(DATASET_DIR, "fear_and_greed.json")


def test_load_token_data():
    """Test loading token data from a CSV file."""
    # Test with actual BTC data file
    btc_file = os.path.join(DATASET_DIR, "btc.csv")
    data = load_token_data(btc_file)

    # Check if data was loaded properly
    assert data is not None
    assert len(data) > 0
    assert len(data[0]) == 3  # [timestamp, price, mcap]

    # Test with non-existent file
    with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
        data = load_token_data("nonexistent_file.csv")
        output = fake_stdout.getvalue()

    assert data is None
    assert "Warning: nonexistent_file.csv not found" in output


def test_load_historical_data():
    """Test loading historical data for multiple tokens."""
    # Test with actual token data
    tokens = ["btc", "eth", "sol"]
    data = load_historical_data(tokens, DATASET_DIR)

    # Check if data was loaded properly
    assert len(data) == 3
    assert "btc" in data
    assert "eth" in data
    assert "sol" in data

    # All tokens should have data
    assert len(data["btc"]) > 0
    assert len(data["eth"]) > 0
    assert len(data["sol"]) > 0

    # Test with a non-existent token
    with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
        data = load_historical_data(["btc", "non_existent_token"], DATASET_DIR)
        output = fake_stdout.getvalue()

    assert "btc" in data
    assert "non_existent_token" not in data
    assert "Warning: Could not load data for non_existent_token" in output


def test_filter_data_by_start_date():
    """Test filtering historical data by start date."""
    # Load some historical data first
    tokens = ["btc", "eth"]
    historical_data = load_historical_data(tokens, DATASET_DIR)

    # Get the timestamp of the 10th data point
    start_timestamp = historical_data["btc"][10][0]

    # Filter data
    filtered = filter_data_by_start_date(historical_data, start_timestamp)

    # Check if filtering worked
    assert len(filtered["btc"]) == len(historical_data["btc"]) - 10
    assert len(filtered["eth"]) == len(historical_data["eth"]) - 10

    # First timestamp should match our start timestamp
    assert filtered["btc"][0][0] == start_timestamp

    # Test with a timestamp beyond the latest data point
    future_timestamp = historical_data["btc"][-1][0] + 1000000
    with patch("sys.stdout", new=io.StringIO()):
        filtered = filter_data_by_start_date(historical_data, future_timestamp)

    # Should return empty dict as no data points remain
    assert len(filtered) == 0


def test_align_data_timestamps():
    """Test aligning data timestamps across different tokens."""
    # Create unaligned data by loading full dataset for BTC and partial for ETH
    btc_data = load_token_data(os.path.join(DATASET_DIR, "btc.csv"))
    eth_data = load_token_data(os.path.join(DATASET_DIR, "eth.csv"))

    # Take only first 100 records for ETH to create misalignment
    eth_data_partial = eth_data[:100]

    unaligned_data = {"btc": btc_data, "eth": eth_data_partial}

    # Align the data
    with patch("sys.stdout", new=io.StringIO()):
        aligned = align_data_timestamps(unaligned_data)

    # Check that both tokens have the same number of data points
    assert len(aligned["btc"]) == len(aligned["eth"])

    # Check that timestamps match for a few random points
    for i in range(0, len(aligned["btc"]), 10):
        if i < len(aligned["btc"]):
            assert aligned["btc"][i][0] == aligned["eth"][i][0]

    # Test with empty data
    with patch("sys.stdout", new=io.StringIO()):
        result = align_data_timestamps({})
    assert result == {}


def test_extract_current_data():
    """Test extracting market caps and prices at a specific timestamp."""
    # Load historical data
    tokens = ["btc", "eth", "sol"]
    historical_data = load_historical_data(tokens, DATASET_DIR)

    # Get a timestamp that exists in the data
    timestamp = historical_data["btc"][5][0]

    # Extract data for that timestamp
    market_caps, prices = extract_current_data(historical_data, timestamp)

    # Check if extraction worked
    assert "btc" in market_caps
    assert "eth" in market_caps
    assert "sol" in market_caps

    assert "btc" in prices
    assert "eth" in prices
    assert "sol" in prices

    # Values should match the original data
    assert market_caps["btc"] == historical_data["btc"][5][2]
    assert prices["btc"] == historical_data["btc"][5][1]

    # Test with timestamp not in data
    invalid_timestamp = timestamp + 1  # Just after the timestamp we know exists
    market_caps, prices = extract_current_data(historical_data, invalid_timestamp)

    # Should return empty dictionaries
    assert len(market_caps) == 0
    assert len(prices) == 0


def test_load_fear_greed_index():
    """Test loading fear and greed index from JSON file."""
    # Test with actual fear and greed data
    data = load_fear_greed_index(FEAR_GREED_PATH)

    # Check if data was loaded properly
    assert data is not None
    assert len(data) > 0
    assert len(data[0]) == 3  # [timestamp, value, classification]

    # Make sure data types are correct
    assert isinstance(data[0][0], int)  # timestamp
    assert isinstance(data[0][1], float)  # value
    assert isinstance(data[0][2], str)  # classification

    # Test with non-existent file
    with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
        data = load_fear_greed_index("nonexistent_file.json")
        output = fake_stdout.getvalue()

    assert data is None
    assert "Warning: nonexistent_file.json not found" in output

    # Test with malformed JSON
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as bad_file:
        bad_file.write('{"name": "Fear and Greed Index"}')  # Missing 'data' key

    try:
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            data = load_fear_greed_index(bad_file.name)
            output = fake_stdout.getvalue()

        assert data is None
        assert "Warning: 'data' key not found" in output
    finally:
        os.unlink(bad_file.name)


def test_process_fear_greed_data():
    """Test processing fear and greed data with filtering and alignment."""
    # Load historical data for alignment
    tokens = ["btc", "eth"]
    historical_data = load_historical_data(tokens, DATASET_DIR)

    # Process fear and greed data with default settings
    processed = process_fear_greed_data(FEAR_GREED_PATH)

    # Check basic processing
    assert processed is not None
    assert len(processed) > 0

    # Get one timestamp from historical data to use as start date
    mid_point = len(historical_data["btc"]) // 2
    mid_timestamp = historical_data["btc"][mid_point][0]
    start_date = datetime.fromtimestamp(mid_timestamp / 1000)

    # Test filtering by start date
    with patch("sys.stdout", new=io.StringIO()):
        processed = process_fear_greed_data(FEAR_GREED_PATH, start_date=start_date)

    # Check that filtering worked - all timestamps should be >= mid_timestamp
    assert all(ts >= mid_timestamp for ts, _, _ in processed)

    # Test alignment with historical data
    with patch("sys.stdout", new=io.StringIO()):
        processed = process_fear_greed_data(
            FEAR_GREED_PATH, historical_data=historical_data
        )

    # All timestamps in processed should be in historical_data
    historical_timestamps = {entry[0] for entry in historical_data["btc"]}
    assert all(ts in historical_timestamps for ts, _, _ in processed)
