import torch
import torch.nn as nn
from transformers import AutoModel

class HuBERTECG(nn.Module):
    def __init__(self, model_size='small'):
        super().__init__()
        self.model = AutoModel.from_pretrained(f"Edoardo-BS/hubert-ecg-{model_size}", trust_remote_code=True)
        
    def preprocess(self, ecg_data):
        """Preprocess the ECG data for the model."""
        # Ensure data is in the correct shape and format
        if len(ecg_data.shape) == 1:
            ecg_data = ecg_data.unsqueeze(0)
        return ecg_data
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

class HuBERTForECGClassification(nn.Module):
    def __init__(self, hubert_ecg, num_labels=2):
        super().__init__()
        self.hubert_ecg = hubert_ecg
        self.classifier = nn.Linear(hubert_ecg.model.config.hidden_size, num_labels)
        
    def preprocess(self, ecg_data):
        """Use the base model's preprocessing."""
        return self.hubert_ecg.preprocess(ecg_data)
    
    def forward(self, x):
        """Forward pass through the model with classification head."""
        outputs = self.hubert_ecg(x)
        # Use the last hidden state for classification
        last_hidden_state = outputs.last_hidden_state
        # Take the mean of the sequence dimension
        pooled_output = torch.mean(last_hidden_state, dim=1)
        # Pass through classifier
        logits = self.classifier(pooled_output)
        return logits 