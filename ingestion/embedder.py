import torch
from torch.nn import functional as F
from ingestion.chunker import MODEL_NAME, tokenizer
from transformers import AutoModel

# Load model once
_model = AutoModel.from_pretrained(MODEL_NAME)
_model.eval()


def _mean_pool(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute a pooled vector for a sequence of tokens.
    :param outputs:
    :param attention_mask:
    :return:
    """
    # Get get all sequence (tokens) vectors
    vectors = outputs.last_hidden_state  # -> (1, n_tokens, model_vector_size)

    # Get attention mask and unsqueezed it in compatible dimension (add a dimension at the end to have it 3D as vectors)
    mask = attention_mask.unsqueeze(-1).float()  # contains 1's for real tokens or 0 for padding [PAD] special characters (which are added when the sequence is shorter than max_length)

    # Return result vector
    return ((vectors * mask)  # multiply between each token vector and mask,
            .sum(dim=1)  # sum all up,
            / mask.sum(dim=1))  # divide by the number of real tokens to get the average (one vector representing the whole sentence)


def embed_text(text: str) -> torch.Tensor:
    """
    Embed a text into a vector.
    :param text: chunk or query
    :return: embedded 1D vector
    """
    # Encode text into a token vector
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)  # every abstract is well under 512 tokens-size

    with torch.no_grad():  # Forward pass, no gradients needed
        outputs = _model(**encoded)

    # Pool the token into a single sentence vector
    embedding = _mean_pool(outputs, encoded['attention_mask'])  # -> (1, model_vector_size)

    # Normalize embedding to unit length (i.e. magnitude = 1) and return; normalization is needed for cosine similarity with FAISS IndexFlatIP
    return F.normalize(embedding, dim=-1).squeeze(0)  # remove dimension (batch) ->  (model_vector_size, )
