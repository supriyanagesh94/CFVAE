from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn

#  This file contains a collection of PyTorch models used to solve the ML tasks in the paper.

features = 32


class MultiTaskMLPModel(nn.Module):
    def __init__(self, feat_dim: int, inp_emb: int, emb_dim1: int, emb_dim2: int) -> None:
        """
        Initialize a Multi-Layer perceptron model for solving the multi-task problem of (i) predicting whether or not
        a patient will receive an intervention and (ii) pairwise-ranking the patients. The ranking is produced by
        gluing together two copies of the network, each of which provides a severity score for a separate user,
        where the order is computed by a sigmoid of the difference.

        See the paragraph labeled "CF VAE Objective" in the paper, as well as figures 3 and 4.

        :param feat_dim: Number of features in the input.
        :param inp_emb: Size of the input to embedding (linear function of input).
        :param emb_dim1: Size of the first hidden embedding layer.
        :param emb_dim2: Size of second hidden embedding layer.
        """
        super(MultiTaskMLPModel, self).__init__()

        # initialize the layers of the MLP

        # ranking layers
        self.word_embeddings = nn.Linear(feat_dim, inp_emb)

        self.ln1 = nn.LayerNorm(inp_emb)
        self.fc1 = nn.Linear(inp_emb, emb_dim1)

        self.ln2 = nn.LayerNorm(emb_dim1)
        self.fc2 = nn.Linear(emb_dim1, emb_dim2)

        self.scorelayer = nn.Linear(emb_dim2, 1)
        self.scoreact = nn.Sigmoid()

        # prediction layer
        self.pred = nn.Linear(1, 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # First do forward pass for x1

        # Compute embedding
        x1_emb = self.word_embeddings(x1)
        x1_emb = self.ln1(x1_emb)

        # pass x1's embedding through the MLP
        x1_fc1 = F.relu(self.fc1(x1_emb))
        x1_fc1 = self.ln2(x1_fc1)
        x1_fc2 = F.relu(self.fc2(x1_fc1))

        # compute the score and the prediction for x1
        x1_score = self.scorelayer(x1_fc2)
        x1_pred = self.pred(x1_score)

        # Now we do the same for x2

        # compute embedding
        x2_emb = self.word_embeddings(x2)
        x2_emb = self.ln1(x2_emb)

        # pass x2's embedding through the MLP
        x2_fc1 = F.relu(self.fc1(x2_emb))
        x2_fc1 = self.ln2(x2_fc1)
        x2_fc2 = F.relu(self.fc2(x2_fc1))

        # compute the score and prediction for x2
        x2_score = self.scorelayer(x2_fc2)
        x2_pred = self.pred(x2_score)

        # compute the ranking of the two using the score difference
        rank_score = x1_score - x2_score
        rank_score = self.scoreact(rank_score)

        return rank_score, x1_pred, x2_pred


class CFVAE(nn.Module):

    def __init__(self, feat_dim: int, emb_dim1: int, _mlp_dim1: int, _mlp_dim2: int, _mlp_dim3: int, mlp_inpemb: int,
                 f_dim1: int, f_dim2: int) -> None:
        """
        Initialize the CF-VAE architecture, including the standard a VAE encoder-decoder pair for generating the
        reconstruction, and an intervention prediction MLP constraining it to be a counter-factual.
        See the paragraph labeled CF VAE objective in the original paper of the motivation.

        The architecture of the two parts are as follows. The non-linearities are ReLUs.
            - VAE: Two linear layers in each the encoder and decoder.
                   The outer layer is width feat_dim and the inner layer is 2*features.
            - MLP: We use two hidden layers, of size emb_dim1 and emb_dim2. The output of the second layer is then
                   passed into a scoring layer, which can be used as a score for the ranking model using
                   Rank = Sigmoid(score(patient A) - score(patient B))

    - MLP architecture:
    - Four layers in the MLP with output sizes: inp_emb, emb_dim1, emb_dim2, 1
    - Output of fourth layer can be used as a score for training a ranking model. Rank = Sigmoid(score(patient A) - score(patient B))
    - Final layer output for the binary classification problem: intervention required vs not required

        :param feat_dim: (VAE) width of the outer hidden layer of the VAE
        :param emb_dim1: (VAE) half-width of the inner hidden layer (encoding layer) of the VAE
        :param _mlp_dim1: unused
        :param _mlp_dim2: unused
        :param _mlp_dim3: unused
        :param mlp_inpemb: Dimension of the word embedding.
        :param f_dim1: Hidden units in first layer of MLP
        :param f_dim2: Hidden units in second layer of MLP.
        """
        super(CFVAE, self).__init__()

        # VAE branch of the network

        # encoder
        self.enc1 = nn.Linear(in_features=feat_dim, out_features=emb_dim1)
        self.enc2 = nn.Linear(in_features=emb_dim1, out_features=features * 2)

        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=emb_dim1)
        self.dec2 = nn.Linear(in_features=emb_dim1, out_features=feat_dim)

        # MLP branch of the network
        self.word_embeddings = nn.Linear(feat_dim, mlp_inpemb)
        self.ln1 = nn.LayerNorm(mlp_inpemb)

        self.fc1 = nn.Linear(mlp_inpemb, f_dim1)
        self.ln2 = nn.LayerNorm(f_dim1)

        self.fc2 = nn.Linear(f_dim1, f_dim2)

        self.scorelayer = nn.Linear(f_dim2, 1)
        self.pred = nn.Linear(1, 2)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # VAE branch forward pass

        # encoding
        enc = F.relu(self.enc1(seq))
        enc = self.enc2(enc).view(-1, 2, features)

        # get `mu` and `log_var`
        mu = enc[:, 0, :]  # the first feature values as mean
        log_var = enc[:, 1, :]  # the other feature values as variance

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        dec = F.relu(self.dec1(z))
        reconstruction = self.dec2(dec)

        # MLP branch forward pass

        # get embedding
        embeds = self.word_embeddings(reconstruction)
        embeds = self.ln1(embeds)

        # pass through hidden layers
        out1 = F.relu(self.fc1(embeds))
        out1 = self.ln2(out1)

        out2 = F.relu(self.fc2(out1))

        out3 = self.scorelayer(out2)
        pred_s1 = self.pred(out3)

        return reconstruction, mu, log_var, pred_s1
