"""
Data loading functions for the indexfund package.
Contains functions for loading and preprocessing historical crypto data.
"""

import csv
import json
from datetime import datetime

from utils.utils import validate_data_length_consistency


def load_token_data(token_filename):
    """
    Load historical price and market cap data for a token from a CSV file.

    Args:
        token_filename (str): Path to the CSV file containing token data

    Returns:
        list: List of [timestamp, price, market_cap] entries, sorted by timestamp
        None: If there was an error loading the file
    """
    try:
        with open(token_filename, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            token_data = []

            for row in reader:
                # Parse the ISO timestamp to convert to milliseconds timestamp
                time_str = row["timeOpen"]
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                timestamp = int(dt.timestamp() * 1000)  # Convert to milliseconds

                # Get price and market cap from appropriate columns
                price = float(row["open"])
                market_cap = float(row["marketCap"])

                token_data.append([timestamp, price, market_cap])

            # Sort by timestamp
            token_data.sort(key=lambda x: x[0])
            return token_data

    except FileNotFoundError:
        print(f"Warning: {token_filename} not found")
    except KeyError as e:
        # Check if there's a BOM character in the header
        if "timeOpen" in str(e) and "\ufeff" in str(e):
            print(
                f"Warning: BOM character detected in {token_filename}. Try opening the file with UTF-8 encoding."
            )
        else:
            print(f"Warning: Missing key {e} in {token_filename}")
    except Exception as e:
        print(f"Warning: Error processing {token_filename}: {e}")

    return None


def load_historical_data(tokens, data_dir="./"):
    """
    Load historical data for multiple tokens from CSV files.

    Args:
        tokens (list): List of token symbols to load data for
        data_dir (str): Directory containing the CSV files (default: current directory)

    Returns:
        dict: Dictionary mapping token symbols to their historical data
    """
    historical_data = {}

    for token in tokens:
        file_path = f"{data_dir}/{token}.csv"
        token_data = load_token_data(file_path)
        if token_data:
            historical_data[token] = token_data
        else:
            print(f"Warning: Could not load data for {token}")

    # Validate that all loaded data has the same length
    is_valid, message = validate_data_length_consistency(historical_data)
    if not is_valid:
        print(f"Warning: {message}")

    return historical_data


def filter_data_by_start_date(historical_data, start_timestamp):
    """
    Filter historical data to only include data points after the start timestamp.

    Args:
        historical_data (dict): Dictionary containing historical price and market cap data
        start_timestamp (int): Start timestamp in milliseconds

    Returns:
        dict: Filtered historical data
    """
    filtered_data = {}

    for token, data in historical_data.items():
        # Filter data points that are >= start_timestamp
        filtered_token_data = [entry for entry in data if entry[0] >= start_timestamp]

        # Only include tokens that have data after the start date
        if filtered_token_data:
            filtered_data[token] = filtered_token_data

    # Validate that all filtered data has the same length
    is_valid, message = validate_data_length_consistency(filtered_data)
    if not is_valid:
        print(f"Warning: {message}")

    return filtered_data


def align_data_timestamps(historical_data):
    """
    Align all token data to have the same timestamps by finding common timestamps.

    Args:
        historical_data (dict): Dictionary mapping token symbols to their historical data

    Returns:
        dict: Aligned historical data where all tokens have the same timestamps
    """
    if not historical_data or len(historical_data) <= 1:
        return historical_data

    # Find all unique timestamps across all tokens
    all_timestamps = set()
    for token_data in historical_data.values():
        all_timestamps.update(entry[0] for entry in token_data)

    # For each token, find which timestamps it has data for
    token_timestamp_map = {}
    for token, data in historical_data.items():
        token_timestamp_map[token] = set(entry[0] for entry in data)

    # Find common timestamps across all tokens
    common_timestamps = set.intersection(*token_timestamp_map.values())

    if not common_timestamps:
        print("Error: No common timestamps found across all tokens")
        return {}

    # Create new aligned data using only common timestamps
    aligned_data = {}
    for token, data in historical_data.items():
        # Create dictionary for fast lookup
        timestamp_to_data = {entry[0]: entry for entry in data}
        # Extract only common timestamps
        aligned_token_data = [
            timestamp_to_data[ts] for ts in common_timestamps if ts in timestamp_to_data
        ]
        # Sort by timestamp
        aligned_token_data.sort(key=lambda x: x[0])
        aligned_data[token] = aligned_token_data

    # Validate alignment result
    is_valid, message = validate_data_length_consistency(aligned_data)
    if is_valid:
        print(f"Data successfully aligned: {message}")
    else:
        print(f"Error: Data alignment failed - {message}")

    return aligned_data


def extract_current_data(historical_data, timestamp):
    """
    Extract market caps and prices for all tokens at a specific timestamp.

    Args:
        historical_data (dict): Dictionary of historical token data
        timestamp (int): Target timestamp to extract data for

    Returns:
        tuple: (market_caps_dict, prices_dict) of data at the timestamp
    """
    current_market_caps = {}
    current_prices = {}

    for token, data in historical_data.items():
        for ts, price, mcap in data:
            if ts == timestamp:
                current_market_caps[token] = mcap
                current_prices[token] = price
                break

    return current_market_caps, current_prices


def load_fear_greed_index(json_file_path):
    """
    Load fear and greed index data from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing fear and greed index data

    Returns:
        list: List of [timestamp, value, value_classification] entries, sorted by timestamp
        None: If there was an error loading the file
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            fear_greed_data = json.load(f)

        if "data" not in fear_greed_data:
            print(f"Warning: 'data' key not found in {json_file_path}")
            return None

        # Extract data
        result_data = []
        for entry in fear_greed_data["data"]:
            timestamp = int(entry["timestamp"]) * 1000  # Convert to milliseconds
            value = float(entry["value"])
            value_classification = entry["value_classification"]

            result_data.append([timestamp, value, value_classification])

        # Sort by timestamp
        result_data.sort(key=lambda x: x[0])
        return result_data

    except FileNotFoundError:
        print(f"Warning: {json_file_path} not found")
    except KeyError as e:
        print(f"Warning: Missing key {e} in {json_file_path}")
    except Exception as e:
        print(f"Warning: Error processing {json_file_path}: {e}")

    return None


def process_fear_greed_data(json_file_path, start_date=None, historical_data=None):
    """
    Load fear and greed index data from a JSON file and process it with filtering and timestamp alignment.

    Args:
        json_file_path (str): Path to the JSON file containing fear and greed index data
        start_date (str or datetime, optional): Start date to filter data from
        historical_data (dict, optional): Historical token data to align with

    Returns:
        list: Processed fear and greed index data
    """
    # Load fear and greed data
    fear_greed_data = load_fear_greed_index(json_file_path)

    if not fear_greed_data:
        print("Error: No fear and greed data could be loaded.")
        return []

    # Filter data by start date if provided
    if start_date:
        if isinstance(start_date, str):
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_date_obj = start_date

        start_timestamp = int(start_date_obj.timestamp() * 1000)
        fear_greed_data = [
            entry for entry in fear_greed_data if entry[0] >= start_timestamp
        ]

    # Align with historical data if requested
    if historical_data:
        # Get all timestamps from historical data
        all_timestamps = set()
        for token_data in historical_data.values():
            all_timestamps.update(entry[0] for entry in token_data)

        # Filter fear greed data to only include timestamps in historical data
        fear_greed_data = [
            entry for entry in fear_greed_data if entry[0] in all_timestamps
        ]

    return fear_greed_data


def load_and_prepare_data(tokens, data_dir, start_date=None, fear_greed_file=None):
    """
    Load and prepare all necessary data for analysis.

    Args:
        tokens (list): List of token symbols to analyze
        data_dir (str): Directory containing the data files
        start_date (str or datetime, optional): Start date for analysis
        fear_greed_file (str, optional): Path to fear and greed index file

    Returns:
        tuple: (historical_data, fear_greed_data) or (None, None) if data loading fails
    """
    # Load historical data
    historical_data = load_historical_data(tokens, data_dir)

    if not historical_data:
        print("Error: No historical data could be loaded.")
        return None, None

    # Filter data by start date if provided
    if start_date:
        if isinstance(start_date, str):
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_date_obj = start_date

        start_timestamp = int(start_date_obj.timestamp() * 1000)
        historical_data = filter_data_by_start_date(historical_data, start_timestamp)

    # Align timestamps across all tokens
    historical_data = align_data_timestamps(historical_data)

    if not historical_data:
        print("Error: No usable data after filtering/alignment.")
        return None, None

    # Load and process fear and greed data if provided
    fear_greed_data = None
    if fear_greed_file:
        print(f"Loading fear and greed index data from {fear_greed_file}")
        fear_greed_data = process_fear_greed_data(
            fear_greed_file,
            start_date=start_date,
            historical_data=historical_data,
        )
        if fear_greed_data:
            print(
                f"Successfully loaded {len(fear_greed_data)} fear and greed data points"
            )
        else:
            print("Warning: No usable fear and greed data after processing")

    # Perform validation
    is_valid, message = validate_data_length_consistency(historical_data)
    print(f"Data validation: {message}")

    return historical_data, fear_greed_data
