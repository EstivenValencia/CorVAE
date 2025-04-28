"""
Variational Autoencoder con arquitectura Transformer
Versión optimizada con mejor legibilidad y eficiencia.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor

import typing as ty
import math
from matplotlib import pyplot as plt


import os
from torch.utils.tensorboard import SummaryWriter


class LossTracker:
    """Registra pérdidas de reconstrucción, KL, clasificación y la pérdida total
    tanto en entrenamiento como en validación.
    
    Los nombres de los grupos quedan así:
        • **Pretrain/** Recon | KL | Total
        • **Finetune/** Recon | KL | Class | Total
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer: SummaryWriter = SummaryWriter(log_dir)
        self.epoch: int = 0

    # ---------------------------------------------------------------------
    # Pre‑training (sin cabeza) -------------------------------------------
    # ---------------------------------------------------------------------
    def log_pretrain(
        self,
        train_mse: float,
        train_ce: float,
        train_kld: float,
        train_total: float,
        val_mse: float,
        val_ce: float,
        val_kld: float,
        val_total: float,
    ) -> None:
        recon_train = train_mse + train_ce
        recon_val   = val_mse   + val_ce

        self.writer.add_scalars("Pretrain/Recon", {"Train": recon_train, "Val": recon_val}, self.epoch)
        self.writer.add_scalars("Pretrain/KL",    {"Train": train_kld,     "Val": val_kld},    self.epoch)
        self.writer.add_scalars("Pretrain/Total", {"Train": train_total,   "Val": val_total},  self.epoch)

        self.epoch += 1

    # ---------------------------------------------------------------------
    # Fine‑tuning (con cabeza) --------------------------------------------
    # ---------------------------------------------------------------------
    def log_finetune(
        self,
        train_mse: float,
        train_ce: float,
        train_kld: float,
        train_class: float,
        train_total: float,
        val_mse: float,
        val_ce: float,
        val_kld: float,
        val_class: float,
        val_total: float,
    ) -> None:
        recon_train = train_mse + train_ce
        recon_val   = val_mse   + val_ce

        self.writer.add_scalars("Finetune/Recon", {"Train": recon_train, "Val": recon_val}, self.epoch)
        self.writer.add_scalars("Finetune/KL",    {"Train": train_kld,   "Val": val_kld},  self.epoch)
        self.writer.add_scalars("Finetune/Class", {"Train": train_class, "Val": val_class}, self.epoch)
        self.writer.add_scalars("Finetune/Total", {"Train": train_total, "Val": val_total}, self.epoch)

        self.epoch += 1

    # ---------------------------------------------------------------------
    # Limpieza -------------------------------------------------------------
    # ---------------------------------------------------------------------
    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


def compute_loss(
    X_num: torch.Tensor,
    X_cat: torch.Tensor,
    recon_num: torch.Tensor,
    recon_cat: list[torch.Tensor],
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcula las pérdidas de un VAE para datos mixtos (numéricos + categóricos).

    Args:
        X_num      (Tensor): [batch, n_num]  Datos numéricos reales.
        X_cat      (Tensor): [batch, n_cat]  Etiquetas categóricas reales (índices).
        recon_num  (Tensor): [batch, n_num]  Reconstrucción numérica (continua).
        recon_cat  (List[Tensor]): Lista de n_cat tensores de forma [batch, n_classes] con logits.
        mu         (Tensor): [batch, z_dim]  Media del codificador.
        logvar     (Tensor): [batch, z_dim]  Log-varianza del codificador.
        beta       (float):  Peso para el término KLD (beta‑VAE). Por defecto 1.0.

    Returns:
        total_loss (Tensor): Suma de mse + ce + beta * kld.
        mse_loss   (Tensor): Pérdida MSE en parte numérica.
        ce_loss    (Tensor): Pérdida CE promedio en parte categórica.
        kld_loss   (Tensor): Divergencia KL promedio.
        acc        (Tensor): Precisión global de reconstrucción categórica.
    """

    # 1. Reconstrucción numérica: MSE
    #    Promedia automáticamente sobre batch y features.
    mse_loss = F.mse_loss(recon_num, X_num)

    # 2. Reconstrucción categórica: CE + accuracy
    ce_loss = 0.0
    correct  = 0
    total    = 0
    # Contamos sólo las features categóricas presentes
    n_cat_features = sum(1 for logits in recon_cat if logits is not None)

    for i, logits in enumerate(recon_cat):
        if logits is None:
            continue

        # CE entre los logits y la etiqueta real
        ce_loss += F.cross_entropy(logits, X_cat[:, i])

        # Predicción más probable
        preds = logits.argmax(dim=1)
        # Suma de correctos en todo el batch
        correct += (preds == X_cat[:, i]).sum().item()
        total   += X_cat.size(0)

    # Promediar CE por número de features categóricas
    ce_loss = ce_loss / max(n_cat_features, 1)

    # Precisión como fracción de aciertos
    acc = correct / total if total > 0 else torch.tensor(0.0)

    # 3. Divergencia KL entre N(mu, sigma^2) y N(0,1):
    #    KLD = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)), luego media por batch
    kld_element = 1 + logvar - mu.pow(2) - logvar.exp()
    kld_loss = -0.5 * torch.mean(kld_element.mean(-1).mean())

    return mse_loss, ce_loss, kld_loss, acc

class Tokenizer(nn.Module):
    """
    Convierte características numéricas y categóricas en embeddings de tokens,
    añadiendo un token [CLS] al inicio de cada secuencia.
    """
    def __init__(
        self, 
        numeric_dim: int, 
        categories: Optional[List[int]], 
        token_dim: int, 
        use_bias: bool = True
    ):
        super().__init__()
        
        # Configuración de dimensiones
        self.numeric_dim = numeric_dim
        self.token_dim = token_dim
        self.use_bias = use_bias
        
        # Determinar dimensión total considerando categorías
        if categories is None:
            bias_dim = numeric_dim
            self.categorical_offsets = None
            self.categorical_embeddings = None
        else:
            bias_dim = numeric_dim + len(categories)
            offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('categorical_offsets', offsets)
            total_categories = sum(categories)
            self.categorical_embeddings = nn.Embedding(total_categories, token_dim)
            nn.init.kaiming_uniform_(self.categorical_embeddings.weight, a=math.sqrt(5))

        # Parámetros para embedding numérico + token [CLS]
        self.embedding_weights = nn.Parameter(torch.Tensor(numeric_dim + 1, token_dim))
        self.token_biases = nn.Parameter(torch.Tensor(bias_dim, token_dim)) if use_bias else None

        # Inicialización de pesos
        self._init_parameters()

    def _init_parameters(self):
        """Inicializa los parámetros del modelo usando Kaiming uniform."""
        nn.init.kaiming_uniform_(self.embedding_weights, a=math.sqrt(5))
        if self.token_biases is not None:
            nn.init.kaiming_uniform_(self.token_biases, a=math.sqrt(5))

    def forward(
        self, 
        numeric_features: Optional[torch.Tensor], 
        categorical_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Convierte características a embeddings.
        
        Args:
            numeric_features: Tensor(batch, num_features) o None
            categorical_features: Tensor(batch, num_categories) o None
            
        Returns:
            Tensor(batch, tokens, token_dim)
        """
        # Determinar batch_size y dispositivo
        if categorical_features is not None:
            device = categorical_features.device
        elif numeric_features is not None:
            device = numeric_features.device
        else:
            # Opcional: Manejar el caso improbable de que ambos sean None
            # Puedes lanzar un error o asignar un dispositivo por defecto si tiene sentido en tu lógica
            raise ValueError("Ambas características (categóricas y numéricas) son None. No se puede determinar el dispositivo.")
        
        batch_size = numeric_features.size(0) if numeric_features is not None else categorical_features.size(0)
        
        # Construir token [CLS] y embeddings numéricos
        cls_token = torch.ones(batch_size, 1, device=device)
        
        if numeric_features is not None:
            numeric_input = torch.cat([cls_token, numeric_features], dim=1)
        else:
            numeric_input = cls_token
            
        # Embeddings numéricos: broadcast de pesos sobre batch
        numeric_embeddings = self.embedding_weights[None] * numeric_input[:, :, None]
        
        # Agregar embeddings categóricos si existen
        if self.categorical_embeddings is not None and categorical_features is not None:
            indices = categorical_features + self.categorical_offsets[None]
            indices = indices.long()
            categorical_embeddings = self.categorical_embeddings(indices)
            output = torch.cat([numeric_embeddings, categorical_embeddings], dim=1)
        else:
            output = numeric_embeddings
            
        # Añadir sesgo de token si está habilitado
        if self.token_biases is not None:
            biases = torch.cat([
                torch.zeros(1, self.token_biases.size(1), device=device),
                self.token_biases
            ], dim=0)
            output = output + biases[None]
            
        return output


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization = 'kaiming'):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression = None, value_compression = None):
  
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a/b , dim=-1)

        
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x
        
class Transformer(nn.Module):

    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout = 0.0,
        ffn_dropout = 0.0,
        residual_dropout = 0.0,
        activation = 'relu',
        prenormalization = True,
        initialization = 'kaiming',      
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
   
            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        # self.activation = lib.get_activation_fn(activation)
        # self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)


    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                x_residual,
                x_residual,
            )

            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x

class VAE(nn.Module):
    def __init__(self, d_numerical, categories, num_layers, hid_dim, n_head = 1, factor = 4, bias = True):
        super(VAE, self).__init__()
 
        self.d_numerical = d_numerical
        self.categories = categories
        self.hid_dim = hid_dim
        d_token = hid_dim
        self.n_head = n_head
 
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, use_bias = bias)

        self.encoder_mu = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.encoder_logvar = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

        self.decoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

    def get_embedding(self, x):
        return self.encoder_mu(x, x).detach() 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)

        mu_z = self.encoder_mu(x)
        std_z = self.encoder_logvar(x)

        z = self.reparameterize(mu_z, std_z)

        h = self.decoder(z[:,1:])
        
        return h, mu_z, std_z


class Reconstructor(nn.Module):
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))  
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h_num  = h[:, :self.d_numerical]
        h_cat  = h[:, self.d_numerical:]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):
      
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat


class ClassifierHead(nn.Module):
    """Classifier head with Batch Normalization and Dropout.

    Args:
        input_dim (int): Dimensionality of the latent feature vector.
        num_classes (int): Number of target classes.
        dropout (float, optional): Dropout probability applied after BatchNorm. Defaults to 0.3.
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is the latent representation (e.g., CLS token or mu_z)
        x = self.bn(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits



class ModelVAE(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head = 1, factor = 4,  bias = True, num_classes = None):
        super(ModelVAE, self).__init__()

        self.VAE = VAE(d_numerical, categories, num_layers, d_token, n_head = n_head, factor = factor, bias = bias)
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)

        self.classifier = None
        if num_classes is not None and num_classes > 0:
            # calcula cuántos tokens genera la VAE (1 CLS + num_cols + cat_cols)
            #seq_len = 1 + d_numerical + len(categories)
            # dimensión total al aplanar todos los tokens
            #latent_dim_total = seq_len * d_token
            print(f"Creando ClassifierHead con input_dim={d_token} y num_classes={num_classes}")
            self.classifier = ClassifierHead(input_dim=d_token,
                                            num_classes=num_classes)
        else:
            print("No se creó ClassifierHead (num_classes no proporcionado o <= 0).")

    def get_embedding(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)
        return self.VAE.get_embedding(x)

    def forward(self, x_num, x_cat):

        h, mu_z, std_z = self.VAE(x_num, x_cat)

        # recon_x_num, recon_x_cat = self.Reconstructor(h[:, 1:])
        recon_x_num, recon_x_cat = self.Reconstructor(h)

        class_logits = None
        if self.classifier is not None:
            # aplanamos todo el espacio latente: [B, seq_len * d_token]
            flat_mu = flat_mu = mu_z[:, 0, :]
            # ahora class_logits: [B, num_classes]
            class_logits = self.classifier(flat_mu)

        return recon_x_num, recon_x_cat, mu_z, std_z, class_logits # Devuelve 5 valores

class EncoderModel(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head, factor, bias = True):
        super(EncoderModel, self).__init__()
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        self.VAE_Encoder = Transformer(num_layers, d_token, n_head, d_token, factor)

    def load_weights(self, Pretrained_VAE):
        self.Tokenizer.load_state_dict(Pretrained_VAE.VAE.Tokenizer.state_dict())
        self.VAE_Encoder.load_state_dict(Pretrained_VAE.VAE.encoder_mu.state_dict())

    def forward(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)
        z = self.VAE_Encoder(x)

        return z

class DecoderModel(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head, factor, bias = True):
        super(DecoderModel, self).__init__()
        self.VAE_Decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.Detokenizer = Reconstructor(d_numerical, categories, d_token)
        
    def load_weights(self, Pretrained_VAE):
        self.VAE_Decoder.load_state_dict(Pretrained_VAE.VAE.decoder.state_dict())
        self.Detokenizer.load_state_dict(Pretrained_VAE.Reconstructor.state_dict())

    def forward(self, z):

        h = self.VAE_Decoder(z)
        x_hat_num, x_hat_cat = self.Detokenizer(h)

        return x_hat_num, x_hat_cat
