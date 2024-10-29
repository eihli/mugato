# μGATO - Mini Unofficial GATO

A small, simple implementation of what's described in [DeepMind's GATO paper](https://arxiv.org/abs/2205.06175).

Goals:

- Clarity
- Simplicity
- Iterability
- Experimentability

Non-goals:

- Runtime performance
- Eval performance

# GATO

The GATO paper only describes a way of tokenizing and sequencing data so that we can feed it into a sequence model. They use a transformer for the model. But the architecture isn't really the point of the paper.

Tokenizing and sequencing data is what the bulk of the paper is about.

Tokenizing and sequencing data is what the bulk of this repo is about.


# Tokenizing and shaping data for input

## Agent and robotics modalities

Consider the following example, typical of an agent/robotics dataset:

``` python
examples = [
    {
        # Continuous/Discrete/Text looks like this:
        #
        #                                 Episodes
        #                                 |  Tokens
        #                                 |  |  Channels
        #                                 |  |  |
        'mission': torch.arange(2*4).view(2, 4, 1),
        # Images look like this:
        #
        #                                      Episodes
        #                                      |  Height
        #                                      |    |    Width
        #                                      |    |    |  Channels
        #                                      |    |    |  |
        'image': torch.randn(2*256*256*3).view(2, 256, 256, 3),
        'action': torch.arange(2*1).view(2, 1, 1),
    },
    {
        #                                 Episodes
        #                                 |  Tokens
        #                                 |  |  Channels
        #                                 |  |  |
        'mission': torch.arange(3*3).view(3, 3, 1),
        #                                      Episodes
        #                                      |  Height
        #                                      |    |    Width
        #                                      |    |    |  Channels
        #                                      |    |    |  |
        'image': torch.randn(3*256*256*3).view(3, 256, 256, 3),
        'action': torch.arange(3*1).view(3, 1, 1),
    },
]
```

Those examples would be batched (collated?) to actually look like the following.

Notice that padding.

- The first example had its episodes padded to 3.
- The second example had its mission tokens padded to 4.

``` python
batch = {
    # Continuous/Discrete/Text looks like this:
    #                                   Batch
    #                                   |  Episodes
    #                                   |  |  Tokens
    #                                   |  |  |  Channels
    #                                   |  |  |  |
    'mission': torch.arange(2*3*4).view(2, 3, 4, 1),
    # Images look like this:
    #                                        Batch
    #                                        |  Episodes
    #                                        |  |  Height
    #                                        |  |    |  Width
    #                                        |  |    |    |  Channels
    #                                        |  |    |    |  |
    'image': torch.randn(2*3*256*256*3).view(2, 3, 256, 256, 3),
    'action': torch.arange(2*3*1).view(2, 3, 1, 1),
}
```

Eventually, we'll want to `concat` along the `tokens` rank. (At this point, their more like "timesteps" than "tokens". But it's easy to get "timesteps" conflated with "episodes" when dealing with agent/robotics datasets. Just keep that in mind.)

Before we can concat along the tokens rank, we'll have to give everything the same number of channel dimensions. That means we can only do this after we've embedded. But once we do that, then the batch will look like this before we concat along the tokens dimension.

``` python
batch = {
    #                                        Batch
    #                                        |  Episodes
    #                                        |  |    Tokens
    #                                        |  |    |    Channels
    #                                        |  |    |    |
    'mission': torch.arange(2*3*4).view(     2, 3,   4, 768),
    'image': torch.randn(2*3*256*256*3).view(2, 3, 256, 768),
    'action': torch.arange(2*3*1).view(      2, 3,   1, 768),
}
```

Once we concat, we'll have:

``` python
#                    Batch
#                    |   Episodes
#                    |   |    Mission Tokens
#                    |   |    |    Image Tokens
#                    |   |    |    |    Action Tokens
#                    |   |    |    |    |      Channels 
#                    |   |    |    |    |      |
batch = torch.arange(2 * 3 * (4 + 256 + 1) * 768).view(...),
```

Which can simply be reshaped to:

``` python
batch = torch.arange(2 * 3 * (4 + 256 + 1) * 768).view(2, 783, 768),
```

And now we've got something in the shape to send to something like a transformer.

## Text and other modalities 

With text, we don't *need* to jump through nearly as many hoops to get it into a format that we can feed into something like a transformer.

A batch of text data might start out as something like:

``` python
[
  ["Four score and seven years ago..."],
  ["There once was a man from Nantucket..."],
  ["Lorem ipsum dolor sit amet..."],
]
```

Which might get tokenized (with padding) as a shape (3, 11) batch of tensors:

```
tensor([[15137,  4776,   290,  3598,   812,  2084,   986,     0,     0,     0,     0],
        [ 1858,  1752,   373,   257,   582,   422,   399,   415, 38811,   986,     0],
        [   43, 29625,   220,  2419,   388,   288, 45621,  1650,   716,   316,   986]])
```

That's a `(3, 11)` tensor that we can immediately embed as a `(3, 768)` batch of embeddings and bam! We're done.

But:

- Embedding needs to happen inside the model, an `nn.Module` (if we want to easily take advantage of the niceties that offers).
- We'll need a bunch of conditional logic in our model to handle the different types of modalities.

We can avoid some complexity in our model if we have a unified shape to our data.

It might seem silly at first to give an "episode" dimension and a "channel" dimension to our text. But I've noticed it makes the rest of the code a lot simpler.

Therefore, I propose, instead of our text looking like:

``` python
[
  ["Four score and seven years ago..."],
  ["There once was a man from Nantucket..."],
  ["Lorem ipsum dolor sit amet..."],
]
```

It actually looks like this:

``` python
examples = [
    {
        'text': "Four score and seven years ago..."
    },
    {
        'text': "There once was a man from Nantucket..."
    },
    {
        'text': "Lorem ipsum dolor sit amet..."
    },
]
```

And then this:

``` python
examples = [
    {
        #
        #                              Episodes
        #                              |   Tokens
        #                              |   |  Channels
        #                              |   |  |
        'text': torch.arange(2*4).view(1,  7, 1),
    },
    {
        'text': torch.arange(2*4).view(1, 10, 1),
    },
    {
        'text': torch.arange(2*4).view(1, 11, 1),
    },
]
```

Which gets padded and batched as:

``` python
batch = {
    #                                   Batch
    #                                   |  Episodes
    #                                   |  |   Tokens
    #                                   |  |   |  Channels
    #                                   |  |   |  |
    'text': torch.arange(2*1*11*1).view(2, 1, 11, 1),
}
```

Once we concat all of our keys along the tokens dimension, then we'll... well... concatting in this case will basically be a no-op because we only have a single key: "text".

``` python
#                    Batch
#                    |   Episodes
#                    |   |    Mission Tokens
#                    |   |    |    Image Tokens
#                    |   |    |    |    Action Tokens
#                    |   |    |    |    |      Channels 
#                    |   |    |    |    |      |
batch = torch.arange(2 * 3 * (4 + 256 + 1) * 768).view(...),
```

Let's bring it all together by viewing side-by-side a simple modality, like text, and a complex modality, like robotics.


``` python
# Text:**
text_batch = {
#                                            Batch
#                                            |  Episodes
#                                            |  |    Tokens
#                                            |  |    |    Channels
#                                            |  |    |    |
    'text': torch.arange(2*1*11*1).view(     2, 1,  11, 768),
} #                                          |  |    |    |
robotics_batch = { #                         |  |    |    |
    'mission': torch.arange(2*3*4).view(     2, 3,   4, 768),
    'image': torch.randn(2*3*256*256*3).view(2, 3, 256, 768),
    'action': torch.arange(2*3*1).view(      2, 3,   1, 768),
}
```

So. Again. I know it might feel kind of weird to have a useless 1-dimension "episode" rank in text/VQA batches. But it makes life easier for our model. We don't have to wrangle a bunch of conditional reshaping.
