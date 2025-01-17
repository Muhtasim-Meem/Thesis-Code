import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Dropout, LeakyReLU, Flatten, Reshape
from tensorflow.keras.models import Model, Sequential
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Manually define a list of Bangla stopwords
bangla_stopwords = set([
    'আমি', 'তুমি', 'সে', 'এটি', 'একটি', 'এই', 'তাদের', 'আমরা', 'আপনি', 'ও', 'এর', 'করে', 'হওয়া', 'ছিল', 'ছিলেন', 
    'কেন', 'এবং', 'থেকে', 'এরপর', 'যখন', 'বাবদ', 'তবে', 'অথবা', 'যত', 'পরে', 'ওই', 'তাদের', 'তা', 'এখন', 'সে', 'যাহা'
    
])

import re
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def preprocess_bangla_text(text):
    
    text = text.lower()
    
   
    text = re.sub(r'http\S+', '', text)
    

    text = re.sub(r'[^a-zA-Z0-9\s\u0980-\u09FF]', '', text)
    
 
    words = word_tokenize(text)
    
 
    words = [word for word in words if word not in bangla_stopwords]
    
    return ' '.join(words)

# Load preprocessed data
X = pd.read_csv(r'G:\4.2\thesis\thesis code\poem.csv')
X = X['content'].values  # Assuming the poems are in a column named 'content'
vocab_size = 5000  # Define vocabulary size based on your data preprocessing

# Preprocess the text data
X_preprocessed = [preprocess_bangla_text(text) for text in X]

# Tokenize the text data
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), max_features=vocab_size)
X_tokenized = vectorizer.fit_transform(X_preprocessed).toarray()

# Define GAN parameters
embedding_dim = 100
latent_dim = 256
batch_size = 64

# Generator
def build_generator(vocab_size, embedding_dim, latent_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=latent_dim))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Discriminator
def build_discriminator(vocab_size):
    model = Sequential()
    model.add(Dense(256, input_dim=vocab_size))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# GAN model combining Generator and Discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_poem = generator(gan_input)
    gan_output = discriminator(generated_poem)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

# Initialize Generator and Discriminator
generator = build_generator(vocab_size, embedding_dim, latent_dim)
discriminator = build_discriminator(vocab_size)
gan = build_gan(generator, discriminator)

# Training the GAN
def train_gan(generator, discriminator, gan, X, epochs=200, batch_size=64):
    # Store loss and accuracy values for plotting
    d_losses = []
    g_losses = []
    d_accuracies = []

    for epoch in range(epochs):
        # Train Discriminator
        real_samples = X[np.random.randint(0, X.shape[0], batch_size)]
        real_samples = real_samples.astype(float)
        real_labels = np.ones(batch_size)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Store losses and accuracy for plotting
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        d_accuracies.append(d_loss[1])

        # Display progress
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f} | G Loss: {g_loss:.4f}")
    
    # Plot the losses and accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_accuracies, label='Discriminator Accuracy')
    plt.title('GAN Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.show()

# Train the GAN
train_gan(generator, discriminator, gan, X_tokenized)

# Generate a new poem
def generate_poem(generator, latent_dim, vectorizer):
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated = generator.predict(noise)
    generated_tokens = np.argmax(generated, axis=1)
    reverse_vocab = {i: word for word, i in vectorizer.vocabulary_.items()}
    
    # Convert tokens to words and generate poem
    poem = ' '.join([reverse_vocab.get(token, '') for token in generated_tokens if token in reverse_vocab])
    return poem

# Example usage
new_poem = generate_poem(generator, latent_dim, vectorizer)
print("Generated Poem:\n", new_poem)
