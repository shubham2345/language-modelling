from datasets import load_dataset
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
from model import TransformerLanguageModel
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import argparse
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PTBDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=90):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]["sentence"]
        # Add start and end tokens
        sentence = f"<s> {sentence} </s>"
        token_ids = self.tokenizer.encode(sentence, out_type=int)
        token_ids = token_ids[:self.max_len]  # Truncate to max length

        if len(token_ids) < 2:  # Skip short sequences
            token_ids = [self.tokenizer.pad_id(), self.tokenizer.eos_id()]
        
        x = token_ids[:-1]  # Input sequence
        y = token_ids[1:]   # Target sequence
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch):
    x_batch = [x[0] for x in batch]
    y_batch = [x[1] for x in batch]
    x_batch = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=0)
    y_batch = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=0)
    return x_batch, y_batch


def train_model(model, train_loader, val_loader, criterion, num_epochs, optimizer):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0  # Accumulate train loss
        total_train_tokens = 0  # Accumulate train tokens
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * y_batch.numel()  # Multiply by number of tokens in the batch
            total_train_tokens += y_batch.numel()  # Count total tokens in the batch

        train_loss = total_train_loss / total_train_tokens  # Average train loss
        train_perplexity = math.exp(train_loss)  # Compute train perplexity

        # Validation
        model.eval()
        total_val_loss = 0  # Accumulate validation loss
        total_val_tokens = 0  # Accumulate validation tokens
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
                total_val_loss += loss.item() * y_batch.numel()  # Multiply by number of tokens in the batch
                total_val_tokens += y_batch.numel()  # Count total tokens in the batch

        val_loss = total_val_loss / total_val_tokens  # Average validation loss
        val_perplexity = math.exp(val_loss)  # Compute validation perplexity

        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, "
            f"Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}"
        )
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(epoch, train_losses, label="Training")
    # plt.plot(epoch, val_losses, label="Validation")
    # plt.xlabel("Epochs")
    # plt.ylabel("Losses")
    # plt.title("Training and Validation Losses")
    # plt.legend()
    # plt.grid()
    # plt.show()


def compute_sentence_perplexities(model, dataset, criterion):
    model.eval()
    sentence_perplexities = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            x, y = dataset[idx]
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)  # Add batch dimension
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            ppl = math.exp(loss.item())
            sentence_perplexities.append({"ID": idx, "ppl": ppl})
    return sentence_perplexities


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a language model.")

    parser.add_argument(
        "output",
        type=str,
        help="Path to the output data file"
    )
    args = parser.parse_args()
    ptb = load_dataset('ptb-text-only/ptb_text_only')
    train_data = ptb['train']
    test_data = ptb['test']
    val_data = ptb['validation']

    # Train SentencePiece tokenizer
    with open("train_sentences.txt", "w") as f:
        for sentence in train_data["sentence"]:
            f.write(sentence + "\n")

    spm.SentencePieceTrainer.train(
        input="train_sentences.txt",
        model_prefix="sentencepiece",
        vocab_size=10000,
        model_type="bpe",
        user_defined_symbols=["<pad>", "<s>", "</s>"]
    )

    # Load the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("sentencepiece.model")

    print(f"Vocabulary size: {sp.vocab_size()}")
    print(f"Padding token index: {sp.pad_id()}, Start token index: {sp.bos_id()}, End token index: {sp.eos_id()}")

    # Create datasets
    train_dataset = PTBDataset(train_data, sp)
    val_dataset = PTBDataset(val_data, sp)
    test_dataset = PTBDataset(test_data, sp)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Initialize model, optimizer, and loss
    model = TransformerLanguageModel(
        vocab_size=sp.vocab_size(),
        d_model=512,
        n_head=4,
        n_layer=6,
        max_len=90,
        padding_idx=sp.pad_id(),  # Use the correct padding index
        dropout=0.2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id())  # Ignore <pad> token

    train_model(model, train_loader, val_loader, criterion, num_epochs=15, optimizer=optimizer)

    sentence_perplexities = compute_sentence_perplexities(model, test_dataset, criterion)

    output_file=args.output
    df = pd.DataFrame(sentence_perplexities)
    df.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")

if __name__ == "__main__":
    main()
