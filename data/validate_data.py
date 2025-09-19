"""
Script to validate the IMDB dataset using Great Expectations.
"""
import great_expectations as ge
from datasets import load_dataset
import pandas as pd

def validate_imdb_data():
    # Load the dataset directly from Hugging Face
    dataset = load_dataset("imdb")
    
    # Convert to pandas DataFrames
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Convert to Great Expectations DataFrames
    train_ge = ge.from_pandas(train_df)
    test_ge = ge.from_pandas(test_df)
    
    # Define and validate expectations
    results = []
    
    # Check column existence
    results.append(train_ge.expect_column_to_exist("text"))
    results.append(train_ge.expect_column_to_exist("label"))
    
    # Validate text content
    results.append(train_ge.expect_column_values_to_not_be_null("text"))
    results.append(train_ge.expect_column_values_to_be_of_type("text", "str"))
    results.append(train_ge.expect_column_value_lengths_to_be_between("text", min_value=1))
    
    # Validate labels
    results.append(train_ge.expect_column_values_to_be_in_set("label", [0, 1]))
    
    # Check class balance (should be roughly 50/50)
    results.append(train_ge.expect_column_distinct_values_to_equal_set("label", [0, 1]))
    
    # Print validation results
    print("\nValidation Results:")
    print("==================")
    for result in results:
        print(f"\nExpectation: {result.expectation_config.expectation_type}")
        print(f"Success: {result.success}")
        if not result.success:
            print(f"Details: {result.result}")
    
    # Check if all validations passed
    all_passed = all(result.success for result in results)
    print(f"\nOverall Validation {'Passed' if all_passed else 'Failed'}")
    
    return all_passed

if __name__ == "__main__":
    validate_imdb_data()
