import pytest
from mugato.data.four_rooms import initialize, create_dataloader
from mugato.tokenizer import Tokenizer
import tiktoken


@pytest.fixture
def tokenizer():
    return Tokenizer(tiktoken.get_encoding("r50k_base"))


def test_initialize():
    data = initialize()
    assert "train" in data
    assert "val" in data
    assert "test" in data
    print(len(data["train"]), len(data["val"]), len(data["test"]))
    assert len(data["train"]) > 0
    assert len(data["val"]) > 0
    assert len(data["test"]) > 0


def test_create_dataloader(tokenizer):
    batch_size = 4
    dataloader = create_dataloader(tokenizer, batch_size=batch_size)

    # Check batch has expected format
    batch = next(iter(dataloader))
    xs, ys, mask = batch
    assert isinstance(xs, dict)
    assert isinstance(ys, dict)
    assert isinstance(mask, dict)

    # Check expected keys exist
    assert "mission" in xs
    assert "direction" in xs
    assert "image" in xs
    assert "action" in xs

    # Check corresponding ys and masks exist
    for key in xs:
        assert key in ys
        assert key in mask
        assert ys[key].shape[0] == batch_size
        assert mask[key].shape[0] == batch_size

    raw_data = initialize()
    raw_sample = raw_data["train"][0]
    raw_mission = raw_sample.observations["mission"][0]
    mission = xs["mission"][0, 0]
    decoded_mission = tokenizer.decode_text(mission)
    assert decoded_mission == raw_mission

    # It's not straightforward to compare actions because the tokenizer
    # selects a random subsequence of episodes to tokenize; this makes sure
    # the entire concatenated sequence fits in the context window.
    # We can test the `mission` above simply because *every* mission is the same
    # across all episodes - the agent's mission doesn't change episode-to-episode.
    # That's just a fact I happen to know about this dataset.
    # So, I'll leave this commented out for now. Maybe we could do something clever
    # in the future. But also, it's probably not worth testing.
    #
    # raw_actions = raw_sample.actions
    # decoded_actions = tokenizer.decode_discrete(ys['action'][0].squeeze(-1))
    # assert decoded_actions == raw_actions
