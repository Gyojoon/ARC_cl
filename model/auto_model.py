import torch
import torch.nn as nn

class Autoencoder_origin(nn.Module):
  def __init__(self):
    super().__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(1, 10, 5, padding=0),
        nn.ReLU(),
        nn.Conv2d(10, 20, 5, padding=0),
        nn.ReLU(),
        )

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(10, 10, kernel_size = 5, stride = 1, padding=0),
        nn.ReLU(),
        )

    self.action_vector = nn.Parameter(torch.ones((20,22,22)))
    self.proj = nn.Linear(10, 11)

  def forward(self, x):
    feature_map = self.encoder(x)
    decoder_input = feature_map #+ self.action_vector
    output = self.decoder(decoder_input)

    return output

class Autoencoder_Linear_origin(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(11, 512)

        self.encoder = nn.Sequential(
            nn.Linear(900, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(300, 600),
            nn.ReLU(),
            nn.Linear(600, 900),
            nn.ReLU(),
        )

        self.proj = nn.Linear(512, 11)

    def forward(self, x):
        embed_x = self.embedding(x.reshape(1, 900).to(torch.long)).transpose(1,2)
        feature_map = self.encoder(embed_x)
        decoder_input = feature_map  # + self.action_vector
        output = self.decoder(decoder_input).transpose(1,2)
        output = self.proj(output).reshape(1,30,30,11).permute(0,3,1,2)

        return output

class Autoencoder_batch1(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(10, 20, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            )

        self.proj = nn.Linear(1, 11)

    def forward(self, x):
        feature_map = self.encoder(x)
        output = self.decoder(feature_map)

        return output

class Autoencoder_batch1_c10(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(10, 20, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            )

        self.proj = nn.Linear(10, 11)

    def forward(self, x):
        feature_map = self.encoder(x)
        output = self.decoder(feature_map)

        return output

class vae_Linear_origin(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(11, 512)

        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(128, 128)
        self.sigma_layer = nn.Linear(128, 128)

        self.proj = nn.Linear(512, 11)

    def forward(self, x):
        if len(x.shape) > 3:
            batch_size = x.shape[0]
            embed_x = self.embedding(x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_x = self.embedding(x.reshape(1, 900).to(torch.long))
        feature_map = self.encoder(embed_x)
        mu = self.mu_layer(feature_map)
        sigma = self.mu_layer(feature_map)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_vector = mu + std * eps
        output = self.decoder(latent_vector)
        output = self.proj(output).reshape(-1,30,30,11).permute(0,3,1,2)

        return output

class vae_batch1_c10(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(10, 20, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            )

        self.mu_layer = nn.Linear(20*22*22, 20*22*22)
        self.sigma_layer = nn.Linear(20 * 22 * 22, 20 * 22 * 22)
        self.proj = nn.Linear(10, 11)

    def forward(self, x):
        feature_map = self.encoder(x)
        if len(feature_map.shape) > 3:
            mu = self.mu_layer(feature_map.reshape(feature_map.shape[0],-1))
            sigma = self.sigma_layer(feature_map.reshape(feature_map.shape[0],-1))
        else:
            mu = self.mu_layer(feature_map.reshape(-1))
            sigma = self.mu_layer(feature_map.reshape(-1))

        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)

        latent_vector = mu + std * eps

        if len(feature_map.shape) > 3:
            output = self.decoder(latent_vector.reshape(feature_map.shape[0],20, 22, 22))
        else:
            output = self.decoder(latent_vector.reshape(20,22,22))

        return output

class Autoencoder_batch1_embedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(10, 20, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            )

        self.encoder_embedding = nn.Linear(20*22*22, 10*22*22)

        self.decompose_embedding = nn.Linear(10*22*22, 20*22*22)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            )

        self.proj = nn.Linear(10, 11)

    def forward(self, x):
        feature_map = self.encoder(x)
        embedding_vector = self.encoder_embedding(feature_map.reshape(1, -1))
        decompose_vector = self.decompose_embedding(embedding_vector)
        output = self.decoder(decompose_vector.reshape(20, 22, 22))

        return output

# TODO - 사전에 embedding 시킨 값을 CNN에 적용한 모델 구현하기
class Autoencoder_batch1_embedding_v2(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(11, 512)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, 5, padding=0),
            nn.ReLU(),
            nn.Conv2d(10, 20, 5, padding=0),
            nn.ReLU(),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 10, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
        )

        self.proj = nn.Linear(512, 11)

    def forward(self, x):
        embed_x = self.embedding(x.reshape(1, 900).to(torch.long)).transpose(1, 2)
        feature_map = self.encoder(embed_x)
        decoder_input = feature_map  # + self.action_vector
        output = self.decoder(decoder_input).transpose(1, 2)
        output = self.proj(output).reshape(1, 30, 30, 11).permute(0, 3, 1, 2)

        return output

class Autoencoder_batch1_compact_embedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 2, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(2, 1, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            )

        self.encoder_embedding = nn.Linear(1*22*22, 512)

        self.decompose_embedding = nn.Linear(512, 1*22*22)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            )

        self.proj = nn.Linear(10, 11)

    def forward(self, x):
        feature_map = self.encoder(x)
        embedding_vector = self.encoder_embedding(feature_map.reshape(1, -1))
        decompose_vector = self.decompose_embedding(embedding_vector)
        output = self.decoder(feature_map)

        return output