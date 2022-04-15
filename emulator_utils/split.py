from sklearn.model_selection import train_test_split

def holdout_split(input_data, output_data, split_fraction):
    """
    Used for train-test splitting of data. The datapoints are randomly selection. 

    """

    train_data, test_data, train_target, test_target = train_test_split(input_data, output_data, test_size=split_fraction)
    return train_data, test_data, train_target, test_target

def holdout_split_nonrandom():
    return NotImplemented




