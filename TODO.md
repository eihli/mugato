# Training loops test

A way to quickly try out different sequence models, embeddings, etc...

A test that builds some really small models and runs a training loop and creates some type of output where we can review how different settings affect the training.

First things first, right now the only way that we're setup to train is through the train.py file at the project root. It would be nice to have a train.py module that could be imported by test files.

What kinds of things will we need in this module? I guess we'll need a "Trainer" class to keep track of the state that is kept track of in globals in train.py.

Let's start there. A simple almost 1:1 refactor of what's in the train.py script in the root directory to a train.py module that can be imported and run from tests.
