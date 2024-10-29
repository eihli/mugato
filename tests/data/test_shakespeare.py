from unittest import TestCase
from mugato.data.shakespeare import initialize, create_dataloader, tokenize

class TestShakespeareDataset(TestCase):
    def setUp(self):
        self.dataset = initialize()
        self.dataloader = create_dataloader()
