from sklearn.model_selection import train_test_split

__all__ = ("random_holdout", "nonrandom_holdout", )



def random_holdout(input_data, output_data, split_fraction):
    """
    Used for train-test splitting of data. The datapoints are randomly selection. 
    
    TO-DO: fix random seed?

    """

    train_data, test_data, train_target, test_target = train_test_split(input_data, output_data, test_size=split_fraction)
    return train_data, test_data, train_target, test_target

def nonrandom_holdout():
    return NotImplemented




