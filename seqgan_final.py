import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt

class PoemDataset(Dataset):
    def __init__(self, csv_file, seq_length=50):
        self.data = pd.read_csv(csv_file)
        self.poems = self.data['content'].astype(str).tolist()
        self.seq_length = seq_length

        # Preprocess poems
        self.poems = [self._preprocess(poem) for poem in self.poems]

        # Build vocabulary
        text = ' '.join(self.poems)
        self.vocab = sorted(set(text))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        # Convert poems to indices
        self.encoded_poems = []
        for poem in self.poems:
            encoded = [self.char_to_idx[char] for char in poem]
            self.encoded_poems.append(encoded)

    def _preprocess(self, text):
        bengali_chars = r'[\u0980-\u09FF\s,.?!-]'
        clean_text = re.findall(bengali_chars, text)
        return ''.join(clean_text).strip()

    def __len__(self):
        return len(self.encoded_poems)

    def __getitem__(self, idx):
        poem = self.encoded_poems[idx]
        if len(poem) < self.seq_length:
            poem = poem + [self.char_to_idx[' ']] * (self.seq_length - len(poem))
        elif len(poem) > self.seq_length:
            start_idx = random.randint(0, len(poem) - self.seq_length)
            poem = poem[start_idx:start_idx + self.seq_length]

        return torch.tensor(poem, dtype=torch.long)

class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, noise, prev_chars):
        batch_size, seq_len = prev_chars.size(0), prev_chars.size(1)
        char_embeds = self.embedding(prev_chars)
        noise = noise.unsqueeze(1).expand(batch_size, seq_len, -1)
        combined_input = torch.cat([char_embeds, noise], dim=2)
        lstm_out, _ = self.lstm(combined_input)
        output = self.fc(lstm_out)
        return output

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, prev_chars):
        char_embeds = self.embedding(prev_chars)
        lstm_out, _ = self.lstm(char_embeds)
        pooled = torch.max(lstm_out, dim=1)[0]
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        output = torch.sigmoid(self.fc2(x))
        return output

class SeqGAN:
    def __init__(self, dataset, latent_dim=64, embedding_dim=128, hidden_dim=256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.latent_dim = latent_dim

        self.generator = Generator(
            dataset.vocab_size, embedding_dim, hidden_dim, latent_dim
        ).to(self.device)

        self.discriminator = Discriminator(
            dataset.vocab_size, embedding_dim, hidden_dim
        ).to(self.device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0004)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        self.criterion = nn.BCELoss()

        self.d_losses = []
        self.g_losses = []

    def calculate_metrics(self, real_poems, fake_poems_idx):
        real = real_poems.view(-1).cpu().numpy()
        fake = fake_poems_idx.view(-1).cpu().numpy()
        min_len = min(len(real), len(fake))
        real, fake = real[:min_len], fake[:min_len]

        acc = accuracy_score(real, fake)
        precision = precision_score(real, fake, average='macro', zero_division=0)
        recall = recall_score(real, fake, average='macro', zero_division=0)
        f1 = f1_score(real, fake, average='macro', zero_division=0)

        real_sentences = [self.dataset.idx_to_char[idx] for idx in real]
        fake_sentences = [self.dataset.idx_to_char[idx] for idx in fake]

        smoothing_function = SmoothingFunction().method1
        bleu = sentence_bleu([real_sentences], fake_sentences, smoothing_function=smoothing_function)

        return acc, precision, recall, f1, bleu

    def train_discriminator(self, real_poems, batch_size):
        self.d_optimizer.zero_grad()
        current_batch_size = real_poems.size(0)

        output_real = self.discriminator(real_poems)
        label_real = torch.full_like(output_real, 0.9, device=self.device)
        d_loss_real = self.criterion(output_real, label_real)

        noise = torch.randn(current_batch_size, self.latent_dim).to(self.device)
        fake_poems = self.generator(noise, real_poems)
        fake_poems_idx = torch.multinomial(F.softmax(fake_poems, dim=-1).view(-1, fake_poems.size(-1)), 1).view(current_batch_size, -1)

        output_fake = self.discriminator(fake_poems_idx.detach())
        label_fake = torch.zeros_like(output_fake, device=self.device)
        d_loss_fake = self.criterion(output_fake, label_fake)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.d_optimizer.step()

        return d_loss.item()

    def train_generator(self, real_poems, batch_size):
        self.g_optimizer.zero_grad()
        current_batch_size = real_poems.size(0)

        noise = torch.randn(current_batch_size, self.latent_dim).to(self.device)
        fake_poems = self.generator(noise, real_poems)
        fake_poems_idx = torch.multinomial(F.softmax(fake_poems, dim=-1).view(-1, fake_poems.size(-1)), 1).view(current_batch_size, -1)

        output_fake = self.discriminator(fake_poems_idx)
        label_real = torch.FloatTensor(current_batch_size, 1).uniform_(0.8, 1.0).to(self.device)
        g_loss = self.criterion(output_fake, label_real)
        g_loss.backward()
        nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.g_optimizer.step()

        return g_loss.item(), fake_poems_idx

    def train(self, num_epochs, batch_size=32):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            d_loss_sum = 0
            g_loss_sum = 0

            for batch_idx, real_poems in enumerate(dataloader):
                real_poems = real_poems.to(self.device)
                d_loss = self.train_discriminator(real_poems, batch_size)
                g_loss, fake_poems_idx = self.train_generator(real_poems, batch_size)

                d_loss_sum += d_loss
                g_loss_sum += g_loss

                if batch_idx % 100 == 0:
                    acc, precision, recall, f1, bleu = self.calculate_metrics(real_poems, fake_poems_idx)
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}], D_Loss: {d_loss:.4f}, G_Loss: {g_loss:.4f}, '
                          f'Acc: {acc:.4f}, Prec: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, BLEU: {bleu:.4f}')

            self.d_losses.append(d_loss_sum / len(dataloader))
            self.g_losses.append(g_loss_sum / len(dataloader))

        self.plot_losses()

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.plot(self.g_losses, label='Generator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_poem(self, seq_length=50, temperature=1.0):
        self.generator.eval()
        with torch.no_grad():
            current_seq = torch.tensor([[self.dataset.char_to_idx[' ']] * seq_length]).to(self.device)
            noise = torch.randn(1, self.latent_dim).to(self.device)

            output = self.generator(noise, current_seq)
            probs = F.softmax(output[0] / temperature, dim=1)
            predicted_chars = torch.multinomial(probs, 1)

            poem = ''.join([self.dataset.idx_to_char[idx.item()] for idx in predicted_chars])
            return poem

def main():
    dataset = PoemDataset('poems.csv', seq_length=50)
    model = SeqGAN(dataset)
    model.train(num_epochs=100)

    for i in range(5):
        poem = model.generate_poem(seq_length=50, temperature=0.8)
        print(f"\nGenerated Poem {i + 1} (with lower temperature):")
        print(poem)
        print("-" * 50)

        poem = model.generate_poem(seq_length=50, temperature=1.2)
        print(f"\nGenerated Poem {i + 1} (with higher temperature):")
        print(poem)
        print("-" * 50)

if __name__ == "__main__":
    main()
