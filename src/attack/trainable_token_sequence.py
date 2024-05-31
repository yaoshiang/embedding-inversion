import torch
import torch.nn.functional as F


class TrainableTokenSequence(torch.nn.Module):
    """A class to implement trainable tokens.

    This is engineered for E5 models.

    The first token is always [CLS] or 101.
    The second is either 'query' 23032 or 'passage' 6019.
    The third is always ':' or 1024.
    After the valid tokens, the penultimate token is [SEP] or 102.
    The final tokens, if any, are always [PAD] or 0.
    """

    def __init__(self, batch_size, sequence_length, vocab_size, dropout):
        """Initialize the TrainableTokenSequence class.

        The batch_size arg must be even. Half the batch_size is used for queries and the other half for passages.


        Args:
            batch_size (int): The batch size.
            sequence_length (int): The sequence length.
            vocab_size (int): The size of the vocabulary.

        """
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even.")

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        super().__init__()
        # self.tokens = F.one_hot(torch.arange(vocab_size)

        cls = F.one_hot(torch.tensor(101), vocab_size).float().requires_grad_(True).reshape((1, 1, -1))
        cls = torch.tile(cls, (batch_size // 2, 1, 1))

        query = F.one_hot(torch.tensor(23032), vocab_size).float().requires_grad_(False).reshape((1, 1, -1))
        query = torch.tile(query, (batch_size // 2, 1, 1))

        passage = F.one_hot(torch.tensor(6019), vocab_size).float().requires_grad_(False).reshape((1, 1, -1))
        passage = torch.tile(passage, (batch_size // 2, 1, 1))

        colon = F.one_hot(torch.tensor(1024), vocab_size).float().requires_grad_(False).reshape(1, 1, -1)
        colon = torch.tile(colon, (batch_size // 2, 1, 1))

        sep = F.one_hot(torch.tensor(102), vocab_size).float().requires_grad_(False).reshape(1, 1, -1)
        sep = torch.tile(sep, (batch_size // 2, 1, 1))

        pad = F.one_hot(torch.tensor(0), vocab_size).float().requires_grad_(False).reshape(1, 1, -1)
        pad = torch.tile(pad, (batch_size // 2, 1, 1))

        # # Build the sep and pad influences.
        # arange = torch.arange(0, sequence_length - 4).reshape(1, -1, 1)

        self.register_buffer("cls", cls)
        self.register_buffer("query", query)
        self.register_buffer("passage", passage)
        self.register_buffer("colon", colon)
        self.register_buffer("sep", sep)
        self.register_buffer("pad", pad)
        # self.register_buffer('arange', arange)

        shape = (batch_size, sequence_length - 5, vocab_size)
        # token_logits = torch.normal(torch.zeros(*shape), torch.ones(*shape))
        # token_logits = torch.normal(torch.zeros(*shape + (2,)), torch.ones(*shape + (2,)))
        token_logits = torch.ones(*shape + (1,)) # Final dim will be reduce summed. 
        token_logits.requires_grad_(True)
        self.token_logits = torch.nn.Parameter(token_logits)

        # self.sep_inflection_point = torch.nn.Parameter(torch.Tensor([batch_size, 0.0]))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.softmax = torch.nn.Sigmoid()

    def forward(self) -> torch.Tensor:
        """Forward pass.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, vocab_size)
            containing the one-hot representation of trainable "input" tokens to
            a sequence model.
        """
        # print('cls', self.cls.shape, self.cls.device)
        # print('colon', self.colon.shape)
        # print('token_logits', self.token_logits.shape)

        queries_prefix = torch.cat(
            [
                self.cls,
                self.query,
                self.colon,
            ],
            dim=1,
        )

        passages_prefix = torch.cat(
            [
                self.cls,
                self.passage,
                self.colon,
            ],
            dim=1,
        )

        prefix = torch.cat([queries_prefix, passages_prefix], dim=0)

        queries_postfix = torch.cat(
            [
                self.sep,
                self.pad,
            ],
            dim=1,
        )

        passages_postfix = torch.cat(
            [
                self.sep,
                self.pad,
            ],
            dim=1,
        )

        postfix = torch.cat([queries_postfix, passages_postfix], dim=0)

        # print('prefix', prefix.shape, prefix.device)
        # print('token_logits', self.token_logits.shape,
        #       self.token_logits.device)

        token_logits = self.token_logits

        if self.training:
            token_logits = self.dropout(token_logits)

        token_logits = torch.sum(token_logits, dim=-1)

        pred = torch.cat([prefix, token_logits, postfix], dim=1)

        # print('pred', pred.shape, pred.device)

        pred = self.softmax(pred)

        # pred = pred / torch.sum(pred, dim=-1, keepdim=True)

        assert pred.shape == (self.batch_size, self.sequence_length, self.vocab_size), pred.shape
        return pred
