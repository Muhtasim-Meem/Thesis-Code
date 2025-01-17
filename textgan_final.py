import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import torch.nn.functional as F
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt

class PoemDataset(Dataset):
    def __init__(self, csv_file, seq_length=50):
        self.data = pd.read_csv(csv_file)
        self.poems = self.data['content'].astype(str).tolist()
        self.seq_length = seq_length

        # Preprocess the poems
        self.poems = [self.preprocess_text(poem) for poem in self.poems]

        # Build vocabulary
        text = ' '.join(self.poems)
        self.vocab = sorted(list(set(text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        # Convert poems to indices
        self.encoded_poems = []
        for poem in self.poems:
            encoded = [self.char_to_idx[char] for char in poem]
            self.encoded_poems.append(encoded)

    def preprocess_text(self, text):
        text = text.replace('\xa0', ' ')  # Replace non-breaking spaces
        text = re.sub(r'[^\u0980-\u09FF\s।\n]', '', text)  # Retain Bangla characters and punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        return text

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
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, noise, prev_chars):
        char_embeds = self.embedding(prev_chars)
        noise = noise.unsqueeze(1).expand(-1, prev_chars.size(1), -1)
        combined_input = torch.cat([char_embeds, noise], dim=2)
        lstm_out, _ = self.lstm(combined_input)
        return self.fc(lstm_out)

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))

class TextGAN:
    def __init__(self, dataset, latent_dim=64, embedding_dim=128, hidden_dim=256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.latent_dim = latent_dim

        self.generator = Generator(dataset.vocab_size, embedding_dim, hidden_dim, latent_dim).to(self.device)
        self.discriminator = Discriminator(dataset.vocab_size, embedding_dim, hidden_dim).to(self.device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0004)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        self.criterion = nn.BCELoss()

    def train(self, num_epochs, batch_size=32):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        d_losses, g_losses = [], []
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'bleu': []}

        for epoch in range(num_epochs):
            d_loss_epoch, g_loss_epoch = 0, 0
            all_labels, all_preds = [], []

            for real_poems in dataloader:
                real_poems = real_poems.to(self.device)
                batch_size = real_poems.size(0)
                label_real = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.0).to(self.device)
                label_fake = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.2).to(self.device)

                # Train Discriminator
                self.d_optimizer.zero_grad()
                output_real = self.discriminator(real_poems)
                d_loss_real = self.criterion(output_real, label_real)

                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_poems = self.generator(noise, real_poems)
                fake_poems_idx = torch.argmax(fake_poems, dim=2)
                output_fake = self.discriminator(fake_poems_idx.detach())
                d_loss_fake = self.criterion(output_fake, label_fake)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # Train Generator
                self.g_optimizer.zero_grad()
                output_fake = self.discriminator(fake_poems_idx)
                g_loss = self.criterion(output_fake, label_real)
                g_loss.backward()
                self.g_optimizer.step()

                d_loss_epoch += d_loss.item()
                g_loss_epoch += g_loss.item()
                all_labels.extend(label_real.cpu().numpy().flatten())
                all_preds.extend(output_real.cpu().detach().numpy().flatten())

            # Convert to binary for metrics calculation
            binary_labels = [1 if label > 0.5 else 0 for label in all_labels]
            binary_preds = [1 if pred > 0.5 else 0 for pred in all_preds]

            accuracy = accuracy_score(binary_labels, binary_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(binary_labels, binary_preds, average='binary')
            generated_poem = self.generate_poem(seq_length=50)
            reference_poem = random.choice(self.dataset.poems)
            bleu_score = sentence_bleu([reference_poem], generated_poem, weights=(0.5, 0.5))

            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['bleu'].append(bleu_score)

            d_losses.append(d_loss_epoch / len(dataloader))
            g_losses.append(g_loss_epoch / len(dataloader))

            print(f'Epoch {epoch + 1}/{num_epochs}, D_Loss: {d_losses[-1]:.4f}, G_Loss: {g_losses[-1]:.4f}, '
                  f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, BLEU: {bleu_score:.4f}')

        # Plot the losses
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_epochs), d_losses, label='Discriminator Loss')
        plt.plot(range(num_epochs), g_losses, label='Generator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Generator and Discriminator Losses')
        plt.show()

        # Print average metrics
        print("Average Metrics:")
        print(f"Accuracy: {np.mean(metrics['accuracy']):.4f}")
        print(f"Precision: {np.mean(metrics['precision']):.4f}")
        print(f"Recall: {np.mean(metrics['recall']):.4f}")
        print(f"F1 Score: {np.mean(metrics['f1']):.4f}")
        print(f"BLEU Score: {np.mean(metrics['bleu']):.4f}")

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
    model = TextGAN(dataset)
    model.train(num_epochs=100)
    for i in range(5):
        poem = model.generate_poem()
        print(f"\nGenerated Poem {i + 1}:")
        print(poem)
        print("-" * 50)

if __name__ == "__main__":
    main()
