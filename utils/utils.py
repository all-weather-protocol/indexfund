"""
Utility functions for the indexfund package.
Contains helper functions for validation and data processing.
"""


def validate_data_length_consistency(historical_data):
    """
    Validate that all tokens in the historical data have the same number of data points.

    Args:
        historical_data (dict): Dictionary mapping token symbols to their historical data

    Returns:
        tuple: (is_valid, validation_message)
            - is_valid (bool): True if all token data has the same length, False otherwise
            - validation_message (str): Description of validation results
    """
    if not historical_data or len(historical_data) <= 1:
        return True, "Only one token present or no data available"

    lengths = {token: len(data) for token, data in historical_data.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) == 1:
        # All tokens have the same length
        return (
            True,
            f"All tokens have the same length: {next(iter(unique_lengths))} data points",
        )
    else:
        # Different lengths detected
        length_details = ", ".join(
            [f"{token}: {length}" for token, length in lengths.items()]
        )
        return False, f"Tokens have different data lengths: {length_details}"
