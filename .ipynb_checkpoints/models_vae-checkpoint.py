import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch.nn.init as nn_init
from torch import Tensor

import typing as ty
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

    def log(
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
        # Agrupa las pérdidas de reconstrucción para mayor claridad en TensorBoard.
        recon_train = train_mse + train_ce
        recon_val = val_mse + val_ce

        # Escribe las métricas escalares para la época actual.
        self.writer.add_scalars(
            "Recon", {"Train": recon_train, "Val": recon_val}, self.epoch
        )
        self.writer.add_scalars("KL", {"Train": train_kld, "Val": val_kld}, self.epoch)
        self.writer.add_scalars(
            "Class", {"Train": train_class, "Val": val_class}, self.epoch
        )
        self.writer.add_scalars(
            "Total", {"Train": train_total, "Val": val_total}, self.epoch
        )

        self.epoch += 1

    # Libera los recursos del writer al finalizar.
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
    beta: float = 1.0,
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
    # Pérdida de reconstrucción para las variables numéricas (Error Cuadrático Medio).
    mse_loss = F.mse_loss(recon_num, X_num)

    # Pérdida de reconstrucción para las variables categóricas (Entropía Cruzada).
    ce_loss = 0.0
    correct = 0
    total = 0
    n_cat_features = sum(1 for logits in recon_cat if logits is not None)

    for i, logits in enumerate(recon_cat):
        if logits is None:
            continue

        # Se acumula la pérdida de entropía cruzada para cada característica.
        ce_loss += F.cross_entropy(logits, X_cat[:, i])

        # Se calcula la precisión de la reconstrucción categórica.
        preds = logits.argmax(dim=1)
        correct += (preds == X_cat[:, i]).sum().item()
        total += X_cat.size(0)

    # Se normaliza la pérdida de CE por el número de características.
    ce_loss = ce_loss / max(n_cat_features, 1)
    if isinstance(ce_loss, float):
        ce_loss = torch.tensor(ce_loss, device=X_num.device)

    acc = correct / total if total > 0 else torch.tensor(0.0)

    # Pérdida de regularización (divergencia KL entre la distribución latente y una normal estándar).
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
        use_bias: bool = True,
    ):
        super().__init__()

        # Almacena las dimensiones principales.
        self.numeric_dim = numeric_dim
        self.token_dim = token_dim
        self.use_bias = use_bias

        # Define los embeddings y offsets para las características categóricas.
        if categories is None:
            bias_dim = numeric_dim
            self.categorical_offsets = None
            self.categorical_embeddings = None
        else:
            bias_dim = numeric_dim + len(categories)
            offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("categorical_offsets", offsets)
            total_categories = sum(categories)
            self.categorical_embeddings = nn.Embedding(total_categories, token_dim)
            nn.init.kaiming_uniform_(self.categorical_embeddings.weight, a=math.sqrt(5))

        # Define los pesos y sesgos para las características numéricas y el token [CLS].
        self.embedding_weights = nn.Parameter(torch.Tensor(numeric_dim + 1, token_dim))
        self.token_biases = (
            nn.Parameter(torch.Tensor(bias_dim, token_dim)) if use_bias else None
        )

        self._init_parameters()

    def _init_parameters(self):
        """Inicializa los parámetros del modelo usando Kaiming uniform."""
        nn.init.kaiming_uniform_(self.embedding_weights, a=math.sqrt(5))
        if self.token_biases is not None:
            nn.init.kaiming_uniform_(self.token_biases, a=math.sqrt(5))

    def forward(
        self,
        numeric_features: Optional[torch.Tensor],
        categorical_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Convierte características a embeddings.

        Args:
            numeric_features: Tensor(batch, num_features) o None
            categorical_features: Tensor(batch, num_categories) o None

        Returns:
            Tensor(batch, tokens, token_dim)
        """
        # Se requiere al menos un tipo de característica para proceder.
        if categorical_features is None and numeric_features is None:
            raise ValueError(
                "Ambas características (categóricas y numéricas) son None."
            )

        device = (
            numeric_features.device
            if numeric_features is not None
            else categorical_features.device
        )
        batch_size = (
            numeric_features.size(0)
            if numeric_features is not None
            else categorical_features.size(0)
        )

        # Prepara el token [CLS] y lo concatena con las características numéricas.
        cls_token = torch.ones(batch_size, 1, device=device)
        numeric_input = (
            torch.cat([cls_token, numeric_features], dim=1)
            if numeric_features is not None
            else cls_token
        )

        # Calcula los embeddings para los tokens numéricos y el [CLS].
        numeric_embeddings = self.embedding_weights[None] * numeric_input[:, :, None]

        # Si existen, calcula y concatena los embeddings categóricos.
        if self.categorical_embeddings is not None and categorical_features is not None:
            indices = (categorical_features + self.categorical_offsets[None]).long()
            categorical_embeddings = self.categorical_embeddings(indices)
            output = torch.cat([numeric_embeddings, categorical_embeddings], dim=1)
        else:
            output = numeric_embeddings

        # Aplica los sesgos a cada token de la secuencia.
        if self.token_biases is not None:
            biases = torch.cat(
                [
                    torch.zeros(1, self.token_biases.size(1), device=device),
                    self.token_biases,
                ],
                dim=0,
            )
            output = output + biases[None]

        return output


# ======================================================================================
# Las siguientes clases de modelos de PyTorch son una adaptación e implementación
# de las arquitecturas propuestas en el paper:
# "MIXED-TYPE TABULAR DATA SYNTHESIS WITH SCORE-BASED DIFFUSION IN LATENT SPACE"
# y trabajos relacionados con Transformers para datos tabulares.
# ======================================================================================


class MLP(nn.Module):
    """
    Implementación de un Perceptrón Multicapa (MLP) simple.

    Consiste en una capa de entrada, una capa oculta con activación ReLU,
    una capa de dropout y una capa de salida lineal.

    Args:
        input_dim (int): Dimensión del vector de entrada.
        hidden_dim (int): Dimensión de la capa oculta.
        output_dim (int): Dimensión del vector de salida.
        dropout (float, optional): Probabilidad de dropout. Por defecto es 0.5.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MLP, self).__init__()
        # Definición de las capas del modelo.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Paso hacia adelante del MLP."""
        # Aplica la primera capa lineal seguida de una activación ReLU.
        x = F.relu(self.fc1(x))
        # Aplica dropout para regularización.
        x = self.dropout(x)
        # La capa final produce los logits de salida.
        x = self.fc2(x)
        return x


class MultiheadAttention(nn.Module):
    """
    Implementación del mecanismo de Atención Multi-Cabeza (Multi-head Attention).

    Proyecta las entradas (Query, Key, Value) en múltiples sub-espacios de atención,
    calcula la atención en cada uno de ellos y concatena los resultados.

    Args:
        d (int): Dimensión del token de entrada y salida.
        n_heads (int): Número de cabezas de atención. `d` debe ser divisible por `n_heads`.
        dropout (float): Probabilidad de dropout para los pesos de atención.
        initialization (str, optional): Método de inicialización de pesos ('xavier' o 'kaiming').
    """

    def __init__(self, d, n_heads, dropout, initialization="kaiming"):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ["xavier", "kaiming"]

        super().__init__()
        # Define las proyecciones lineales para Query, Key y Value.
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        # Inicializa los pesos de las matrices de proyección.
        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == "xavier" and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        """Reorganiza el tensor para el cálculo paralelo en múltiples cabezas."""
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression=None, value_compression=None):
        """
        Paso hacia adelante de la capa de atención.

        Args:
            x_q (torch.Tensor): Tensor de entrada para Query.
            x_kv (torch.Tensor): Tensor de entrada para Key y Value (puede ser el mismo que x_q).
            key_compression (nn.Module, optional): Capa para comprimir Key.
            value_compression (nn.Module, optional): Capa para comprimir Value.

        Returns:
            torch.Tensor: Tensor de salida con la información contextualizada.
        """
        # Proyecta las entradas Q, K, V a través de las capas lineales.
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)

        # Reorganiza los tensores para el cómputo en paralelo de las cabezas.
        q = self._reshape(q)
        k = self._reshape(k)

        # Calcula los scores de atención y aplica escalado.
        attention_scores = q @ k.transpose(1, 2)
        attention_scores = attention_scores / math.sqrt(k.shape[-1])

        # Normaliza los scores para obtener los pesos de atención.
        attention_weights = F.softmax(attention_scores, dim=-1)
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)

        # Aplica los pesos de atención a los valores V.
        x = attention_weights @ self._reshape(v)

        # Recompone las cabezas a la dimensión original y proyecta a la salida.
        batch_size = len(x_q)
        x = (
            x.reshape(batch_size, self.n_heads, q.shape[1], v.shape[-1] // self.n_heads)
            .transpose(1, 2)
            .reshape(batch_size, q.shape[1], -1)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class Transformer(nn.Module):
    """
    Implementación de un bloque Transformer estándar (Encoder).

    Consiste en una pila de N capas, donde cada capa contiene un sub-bloque de
    atención multi-cabeza y un sub-bloque de red feed-forward. Incluye
    conexiones residuales y normalización de capa.

    Args:
        n_layers (int): Número de capas de Transformer a apilar.
        d_token (int): Dimensión de los tokens de entrada/salida.
        n_heads (int): Número de cabezas para la atención multi-cabeza.
        d_out (int): Dimensión de la capa de salida final (head).
        d_ffn_factor (int): Factor para determinar la dimensión oculta de la red feed-forward.
        attention_dropout (float): Dropout para la capa de atención.
        ffn_dropout (float): Dropout para la red feed-forward.
        residual_dropout (float): Dropout para las conexiones residuales.
        activation (str): Función de activación a usar.
        prenormalization (bool): Si es True, aplica LayerNorm antes de cada sub-bloque (Pre-LN).
        initialization (str): Método de inicialización de pesos.
    """

    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        activation="relu",
        prenormalization=True,
        initialization="kaiming",
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])

        # Construye cada capa del Transformer con sus componentes de atención y feed-forward.
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    "attention": MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    "linear0": nn.Linear(d_token, d_hidden),
                    "linear1": nn.Linear(d_hidden, d_token),
                    "norm1": make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer["norm0"] = make_normalization()
            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _start_residual(self, x, layer, norm_idx):
        """Prepara la entrada para la conexión residual (aplica Pre-LayerNorm si es necesario)."""
        x_residual = x
        if self.prenormalization:
            norm_key = f"norm{norm_idx}"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        """Finaliza la conexión residual (dropout, suma y Post-LayerNorm si es necesario)."""
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"norm{norm_idx}"](x)
        return x

    def forward(self, x):
        """Paso hacia adelante del bloque Transformer."""
        for layer in self.layers:
            # --- Bloque de Atención Multi-Cabeza con conexión residual ---
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer["attention"](x_residual, x_residual)
            x = self._end_residual(x, x_residual, layer, 0)

            # --- Bloque de Red Feed-Forward con conexión residual ---
            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer["linear0"](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer["linear1"](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        return x

class VAE(nn.Module):
    """Implementación de un Autoencoder Variacional (VAE) para datos tabulares.

    Esta clase define la arquitectura central del VAE, utilizando un Tokenizer para
    convertir los datos de entrada en una secuencia de tokens y una arquitectura

    basada en Transformers para el encoder y el decoder.

    Args:
        d_numerical (int): Número de características numéricas en los datos de entrada.
        categories (list[int]): Lista con la cardinalidad (número de valores únicos) de cada
                                característica categórica.
        num_layers (int): Número de capas a utilizar en los módulos Transformer.
        hid_dim (int): Dimensión de los tokens y del espacio latente (d_token).
        n_head (int, optional): Número de cabezas de atención para el Transformer. Por defecto es 1.
        factor (int, optional): Factor de expansión para las capas Feed-Forward del Transformer. Por defecto es 4.
        bias (bool, optional): Indica si el Tokenizer debe usar un término de sesgo. Por defecto es True.
    """
    def __init__(self, d_numerical, categories, num_layers, hid_dim, n_head=1, factor=4, bias=True):
        super(VAE, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.hid_dim = hid_dim
        d_token = hid_dim
        self.n_head = n_head

        # Módulo para convertir las características de entrada en una secuencia de embeddings.
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, use_bias=bias)

        # El encoder se divide en dos Transformers para predecir la media (mu) y la
        # varianza logarítmica (logvar) de la distribución latente.
        self.encoder_mu = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.encoder_logvar = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

        # El decoder es un Transformer que reconstruye los tokens a partir de una muestra del espacio latente.
        self.decoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

    def get_embedding(self, x):
        """Obtiene la media del espacio latente (mu) sin muestreo."""
        return self.encoder_mu(x, x).detach()

    def reparameterize(self, mu, logvar):
        """Aplica el truco de reparametrización para el muestreo del espacio latente.

        Este método permite que el gradiente fluya a través del proceso de muestreo
        durante el entrenamiento. z = mu + epsilon * std.

        Args:
            mu (torch.Tensor): La media de la distribución latente.
            logvar (torch.Tensor): La varianza logarítmica de la distribución latente.

        Returns:
            torch.Tensor: Una muestra del espacio latente.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num, x_cat):
        """Define el paso hacia adelante del VAE.

        Args:
            x_num (torch.Tensor): Tensor con las características numéricas de entrada.
            x_cat (torch.Tensor): Tensor con las características categóricas de entrada.

        Returns:
            tuple: Una tupla que contiene:
                - h (torch.Tensor): La salida del decoder (tokens reconstruidos).
                - mu_z (torch.Tensor): La media de la distribución latente.
                - std_z (torch.Tensor): La desviación estándar de la distribución latente.
        """
        # 1. Convertir los datos de entrada en una secuencia de tokens.
        x = self.Tokenizer(x_num, x_cat)

        # 2. Codificar los tokens para obtener los parámetros de la distribución latente.
        mu_z = self.encoder_mu(x)
        logvar_z = self.encoder_logvar(x) # Renombrado para claridad

        # 3. Muestrear una variable latente 'z' usando el truco de reparametrización.
        z = self.reparameterize(mu_z, logvar_z)

        # 4. Decodificar 'z' para reconstruir los tokens de características.
        #    Se omite el token [CLS] (en la posición 0) para la decodificación.
        h = self.decoder(z[:, 1:])
        
        return h, mu_z, logvar_z


class Reconstructor(nn.Module):
    """Módulo para reconstruir los datos tabulares a partir de los tokens decodificados.

    Este módulo actúa como un "de-tokenizer", convirtiendo la secuencia de embeddings
    de salida del decoder de vuelta al formato de datos original (numérico y categórico).

    Args:
        d_numerical (int): Número de características numéricas originales.
        categories (list[int]): Lista con la cardinalidad de cada característica categórica.
        d_token (int): Dimensión de los tokens de entrada.
    """
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        
        # Parámetro de peso para la reconstrucción de las variables numéricas.
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        
        # Lista de capas lineales, una para cada característica categórica, para predecir los logits.
        self.cat_recons = nn.ModuleList()
        for num_classes in categories:
            layer = nn.Linear(d_token, num_classes)
            nn.init.xavier_uniform_(layer.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(layer)

    def forward(self, h):
        """Define el paso hacia adelante del reconstructor.

        Args:
            h (torch.Tensor): Tensor de embeddings de salida del decoder.

        Returns:
            tuple: Una tupla que contiene:
                - recon_x_num (torch.Tensor): Las características numéricas reconstruidas.
                - recon_x_cat (list[torch.Tensor]): Una lista de tensores con los logits para cada
                  característica categórica reconstruida.
        """
        # Separa los tokens que corresponden a las variables numéricas y categóricas.
        h_num = h[:, :self.d_numerical]
        h_cat = h[:, self.d_numerical:]

        # Reconstruye las variables numéricas mediante una proyección.
        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        
        # Reconstruye cada variable categórica aplicando su capa lineal correspondiente.
        recon_x_cat = []
        for i, recon_layer in enumerate(self.cat_recons):
            recon_x_cat.append(recon_layer(h_cat[:, i]))

        return recon_x_num, recon_x_cat


class ClassifierHead(nn.Module):
    """Cabezal de clasificación para predecir una clase a partir de un vector latente.

    Implementa una arquitectura estándar para clasificación sobre embeddings:
    LayerNorm -> Dropout -> Capa Lineal. Esto estabiliza el entrenamiento y
    proporciona una regularización efectiva.

    Args:
        input_dim (int): Dimensión del vector de entrada (ej. la dimensión de un token).
        num_classes (int): Número de clases de salida.
        dropout (float, optional): Tasa de dropout a aplicar. Por defecto es 0.3.
    """
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        # Normalización de capa para estabilizar la entrada del clasificador.
        self.norm = nn.LayerNorm(input_dim)
        # Dropout para regularizar y prevenir el sobreajuste.
        self.dropout = nn.Dropout(dropout)
        # Capa lineal final que proyecta a los logits de las clases.
        self.fc = nn.Linear(input_dim, num_classes)

        # Inicialización de pesos recomendada para la capa de clasificación.
        nn.init.xavier_uniform_(self.fc.weight, gain=1.0)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paso hacia adelante del clasificador.

        Args:
            x (torch.Tensor): Tensor de entrada, típicamente el token [CLS] o
                              la media del espacio latente (`mu_z[:, 0, :]`).
                              Shape: [batch_size, input_dim].

        Returns:
            torch.Tensor: Logits de clasificación. Shape: [batch_size, num_classes].
        """
        # Aplica la secuencia de normalización, dropout y capa lineal.
        x = self.norm(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class ModelVAE(nn.Module):
    """Modelo VAE completo que integra el VAE, el Reconstructor y un cabezal de clasificación.

    Este es el modelo principal que se entrena de extremo a extremo, capaz de realizar
    reconstrucción (auto-supervisada) y clasificación (supervisada) de forma conjunta,
    lo que lo hace ideal para tareas de fine-tuning.

    Args:
        (Ver los argumentos de las clases VAE y ClassifierHead para una descripción completa).
        num_classes (int, optional): Si se proporciona un valor > 0, se crea un cabezal de
                                     clasificación para entrenamiento supervisado.
    """
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head=1, factor=4, bias=True, num_classes=None, droput_class=0.3):
        super(ModelVAE, self).__init__()

        # Módulo VAE principal para codificación y decodificación.
        self.VAE = VAE(d_numerical, categories, num_layers, d_token, n_head=n_head, factor=factor, bias=bias)
        # Módulo para reconstruir los datos tabulares desde los tokens decodificados.
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)

        # Creación opcional del cabezal de clasificación para fine-tuning.
        self.classifier = None
        if num_classes is not None and num_classes > 0:
            self.classifier = ClassifierHead(input_dim=d_token,
                                             num_classes=num_classes,
                                             dropout=droput_class)

    def get_embedding(self, x_num, x_cat):
        """Función de utilidad para obtener la representación latente de los datos."""
        x = self.Tokenizer(x_num, x_cat)
        return self.VAE.get_embedding(x)

    def forward(self, x_num, x_cat):
        """Paso hacia adelante del modelo completo.

        Args:
            x_num (torch.Tensor): Características numéricas de entrada.
            x_cat (torch.Tensor): Características categóricas de entrada.

        Returns:
            tuple: Una tupla con 5 elementos:
                - recon_x_num (torch.Tensor): Reconstrucción numérica.
                - recon_x_cat (list[torch.Tensor]): Logits de reconstrucción categórica.
                - mu_z (torch.Tensor): Media de la distribución latente.
                - logvar_z (torch.Tensor): Varianza logarítmica de la distribución latente.
                - class_logits (torch.Tensor or None): Logits de clasificación si el clasificador está activo.
        """
        # Obtiene la salida del VAE (tokens decodificados y parámetros latentes).
        h, mu_z, logvar_z = self.VAE(x_num, x_cat)

        # Reconstruye los datos tabulares a partir de los tokens decodificados.
        recon_x_num, recon_x_cat = self.Reconstructor(h)

        # Si el modelo incluye un clasificador, calcula los logits de clasificación.
        class_logits = None
        if self.classifier is not None:
            # Se utiliza la representación del token [CLS] (en la posición 0) como entrada.
            cls_token_representation = mu_z[:, 0, :]
            class_logits = self.classifier(cls_token_representation)

        return recon_x_num, recon_x_cat, mu_z, logvar_z, class_logits


class EncoderModel(nn.Module):
    """Modelo que encapsula únicamente la parte del Encoder de un VAE pre-entrenado.

    Diseñado para la etapa de inferencia, permite obtener la representación latente (z)
    de nuevos datos de manera eficiente, sin necesidad de ejecutar el decoder.
    """
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head, factor, bias=True):
        super(EncoderModel, self).__init__()
        # El encoder requiere un Tokenizer para procesar la entrada.
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        # El encoder en sí es un Transformer que predice la media (mu) del espacio latente.
        self.VAE_Encoder = Transformer(num_layers, d_token, n_head, d_token, factor)

    def load_weights(self, Pretrained_VAE: ModelVAE):
        """Carga los pesos del tokenizer y del encoder desde un VAE completo y pre-entrenado."""
        self.Tokenizer.load_state_dict(Pretrained_VAE.VAE.Tokenizer.state_dict())
        self.VAE_Encoder.load_state_dict(Pretrained_VAE.VAE.encoder_mu.state_dict())

    def forward(self, x_num, x_cat):
        """Paso hacia adelante: tokeniza la entrada y la codifica en el espacio latente."""
        x = self.Tokenizer(x_num, x_cat)
        z = self.VAE_Encoder(x)
        return z


class DecoderModel(nn.Module):
    """Modelo que encapsula únicamente la parte del Decoder de un VAE pre-entrenado.

    Diseñado para la etapa de inferencia, permite generar nuevos datos a partir de un
    vector latente (z) que puede ser muestreado o manipulado.
    """
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head, factor, bias=True):
        super(DecoderModel, self).__init__()
        # El decoder es un Transformer que convierte un vector latente en una secuencia de tokens.
        self.VAE_Decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        # El Detokenizer (Reconstructor) convierte los tokens de salida a datos tabulares.
        self.Detokenizer = Reconstructor(d_numerical, categories, d_token)
        
    def load_weights(self, Pretrained_VAE: ModelVAE):
        """Carga los pesos del decoder y del reconstructor desde un VAE completo y pre-entrenado."""
        self.VAE_Decoder.load_state_dict(Pretrained_VAE.VAE.decoder.state_dict())
        self.Detokenizer.load_state_dict(Pretrained_VAE.Reconstructor.state_dict())

    def forward(self, z):
        """Paso hacia adelante: decodifica un vector latente 'z' y lo reconstruye a datos tabulares."""
        # Pasa 'z' por el decoder para obtener la secuencia de tokens reconstruida.
        h = self.VAE_Decoder(z)
        # Convierte los tokens en los datos numéricos y categóricos finales.
        x_hat_num, x_hat_cat = self.Detokenizer(h)
        return x_hat_num, x_hat_cat