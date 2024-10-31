def splits(dataset, train_split=0.8, val_split=0.9):
    num_samples = len(dataset)
    train_split = int(num_samples * train_split)
    val_split = int(num_samples * val_split)
    train_data = dataset[:train_split]
    val_data = dataset[train_split:val_split]
    test_data = dataset[val_split:]
    return train_data, val_data, test_data
