from transformers import AutoTokenizer


# Load the model once
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def chunk_text(
        text: str,
        chunk_size: int = 256,
        overlap: int = 32
) -> list[str]:
    # Use encode of tokenizer to get the list of tokens ids
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Iterate over tokens to create chunks based on chunk_size and overlap
    chunks = []
    chunk_start = 0
    while chunk_start < len(token_ids):
        chunk_ids = token_ids[chunk_start : chunk_start+chunk_size]

        # Decode chunk ids back to words and add to list
        chunk_str = tokenizer.decode(chunk_ids)
        chunks.append(chunk_str)

        # Increment start id
        chunk_start += chunk_size - overlap

    return chunks

