"""
Temporal Fusion Transformer Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attention_output), attention_weights

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for feature selection"""
    
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear = nn.Linear(input_size, output_size * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        linear_output = self.linear(x)
        linear_output = self.dropout(linear_output)
        
        # Split into two parts for gating
        linear_part, gate_part = linear_output.chunk(2, dim=-1)
        
        return linear_part * torch.sigmoid(gate_part)

class VariableSelectionNetwork(nn.Module):
    """Variable selection network for feature importance"""
    
    def __init__(self, input_size: int, num_variables: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.num_variables = num_variables
        self.variable_dim = input_size // num_variables
        
        # Feature processing
        self.variable_processors = nn.ModuleList([
            GatedLinearUnit(self.variable_dim, hidden_size, dropout)
            for _ in range(num_variables)
        ])
        
        # Variable selection weights
        self.variable_selection = nn.Sequential(
            nn.Linear(hidden_size * num_variables, num_variables),
            nn.Softmax(dim=-1)
        )
        
        # Output processing
        self.output_layer = GatedLinearUnit(hidden_size, hidden_size, dropout)
    
    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        
        # Reshape for variable processing
        x = x.view(batch_size, seq_len, self.num_variables, self.variable_dim)
        
        # Process each variable
        processed_variables = []
        for i, processor in enumerate(self.variable_processors):
            processed_var = processor(x[:, :, i, :])
            processed_variables.append(processed_var)
        
        # Stack processed variables
        processed_variables = torch.stack(processed_variables, dim=2)
        
        # Calculate variable selection weights
        flattened = processed_variables.view(batch_size, seq_len, -1)
        selection_weights = self.variable_selection(flattened)
        
        # Apply selection weights
        selection_weights = selection_weights.unsqueeze(-1)
        selected_variables = processed_variables * selection_weights
        
        # Sum across variables
        combined_output = selected_variables.sum(dim=2)
        
        # Final processing
        output = self.output_layer(combined_output)
        
        return output, selection_weights.squeeze(-1)

class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for financial forecasting"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_encoder_layers = config["num_encoder_layers"]
        self.num_decoder_layers = config["num_decoder_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.output_size = config.get("output_size", 4)  # Number of horizons
        self.quantiles = config.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
        
        # Input processing
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        
        # Variable selection
        num_variables = min(16, self.input_size // 4)  # Assume 4 features per variable
        self.variable_selection = VariableSelectionNetwork(
            self.input_size, num_variables, self.hidden_size, self.dropout_rate
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.hidden_size)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_decoder_layers)
        
        # Output heads
        self.point_forecast_head = nn.Linear(self.hidden_size, self.output_size)
        self.quantile_forecast_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, self.output_size) for _ in self.quantiles
        ])
        self.volatility_head = nn.Linear(self.hidden_size, self.output_size)
        self.direction_head = nn.Linear(self.hidden_size, self.output_size)
        
        # Attention pooling for final prediction
        self.attention_pooling = MultiHeadAttention(self.hidden_size, self.num_heads, self.dropout_rate)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x, target_sequence_length: int = 1):
        batch_size, seq_len, input_size = x.shape
        
        # Variable selection
        selected_features, variable_weights = self.variable_selection(x)
        
        # Input projection
        projected_input = self.input_projection(x)
        
        # Combine with variable selection
        encoder_input = selected_features + projected_input
        encoder_input = self.dropout(encoder_input)
        
        # Add positional encoding
        encoder_input = encoder_input.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        encoder_input = self.positional_encoding(encoder_input)
        encoder_input = encoder_input.transpose(0, 1)  # Back to (batch_size, seq_len, hidden_size)
        
        # Encoder
        encoder_output = self.encoder(encoder_input)
        
        # Decoder (for multi-step prediction)
        # Use last encoder output as initial decoder input
        decoder_input = encoder_output[:, -target_sequence_length:, :]
        
        # Self-attention in decoder
        decoder_output = self.decoder(decoder_input, encoder_output)
        
        # Attention pooling across sequence dimension
        pooled_output, attention_weights = self.attention_pooling(
            decoder_output, decoder_output, decoder_output
        )
        
        # Use last timestep for prediction
        final_representation = pooled_output[:, -1, :]
        
        # Generate predictions
        point_forecast = self.point_forecast_head(final_representation)
        
        quantile_forecasts = []
        for quantile_head in self.quantile_forecast_heads:
            quantile_pred = quantile_head(final_representation)
            quantile_forecasts.append(quantile_pred)
        quantile_forecasts = torch.stack(quantile_forecasts, dim=1)
        
        volatility_forecast = torch.exp(self.volatility_head(final_representation))  # Ensure positive
        direction_logits = self.direction_head(final_representation)
        
        return {
            "point_forecast": point_forecast,
            "quantile_forecasts": quantile_forecasts,
            "volatility_forecast": volatility_forecast,
            "direction_logits": direction_logits,
            "attention_weights": attention_weights,
            "variable_weights": variable_weights
        }
    
    def predict_step(self, x):
        """Single step prediction"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs
