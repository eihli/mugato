from typing import Protocol

class TextTokenizer(Protocol):
    n_vocab: int
    eot_token: int
    def encode(self, text: str) -> List[int]:
        ...
    def decode(self, tokens: List[int]) -> str:
        ...

class Tokenizer:
    def __init__(self, text_tokenizer: TextTokenizer):
        self.text_tokenizer = text_tokenizer
        self.eot_token_id = text_tokenizer.eot_token
        self.eot_token = text_tokenizer.decode([self.eot_token_id])
        self.n_text = text_tokenizer.n_vocab
        self.n_discrete = 1024
        self.separator = self.boa_token = 1023  # Separator between observation and action.
        self.vocab_size = self.n_text + self.n_discrete

    def encode_text(self, text):
        return encode_text(self.text_tokenizer, text).unsqueeze(-1)

    def decode_text(self, tokens):
        return decode_text(self.text_tokenizer, tokens.squeeze(-1))

    def encode_discrete(self, xs):
        return encode_discrete(xs, self.n_text).unsqueeze(-1)

    def decode_discrete(self, tokens):
        return decode_discrete(tokens.squeeze(-1), self.n_text)

    def encode_continuous(self, xs, is_action=False):
        return encode_continuous(xs, self.n_text).unsqueeze(-1)

    def decode_continuous(self, tokens):
        return decode_continuous(tokens.squeeze(-1), self.n_text, original_min, original_max)

    def encode_image(self, image, patch_size=16):
        return encode_image(image, patch_size)

    def decode_image(self, tokens, image_shape=(3, 192, 192), patch_size=16):
        return decode_image(tokens, image_shape, patch_size)
