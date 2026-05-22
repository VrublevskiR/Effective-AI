import torch
import torch.nn as nn
from torch.optim import AdamW

class InductionTransformer(nn.Module):
    """
    Минимальный каузальный трансформер для решения задачи повторения последовательности
    (формирование Induction Heads) 2 слоя, 2 головы.
    """
    def __init__(self, vocab_size: int = 4, d_model: int = 6, n_heads: int = 2, n_layers: int = 2, max_seq_len: int = 10):
        super().__init__()
        
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True
        )
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        device = input_ids.device
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=device)
        
        positions = torch.arange(seq_length, device=device).unsqueeze(0)
        
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        hidden_states = self.transformer_blocks(x, mask=causal_mask, is_causal=True)
        
        return self.lm_head(hidden_states)


def get_training_batch(batch_size: int = 128, half_len: int = 5, vocab_size: int = 4) -> tuple:
    """Генерирует батч с повторяющимися последовательностями (например: 3 1 0 2 1 | 3 1 0 2 1)"""
    
    first_half = torch.randint(0, vocab_size, (batch_size, half_len))
    
    full_sequence = torch.cat([first_half, first_half], dim=1)
    
    inputs = full_sequence[:, :-1]
    targets = full_sequence[:, 1:]
    return inputs, targets


def train_model():

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = InductionTransformer(vocab_size=4, d_model=6, n_heads=2, n_layers=2, max_seq_len=10).to(device)
    
    optimizer = AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Начинаем обучение Induction Heads...")
    model.train()
    epochs = 1500
    
    for epoch in range(1, epochs + 1):
        inputs, targets = get_training_batch(batch_size=128, half_len=5, vocab_size=4)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        
        loss = criterion(logits.view(-1, 4), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if epoch % 300 == 0:
            print(f"Шаг {epoch}/{epochs} | Loss: {loss.item():.4f}")
            
    return model


def generate_sequence(model: nn.Module, prompt: str):
    """Авторегрессионная генерация продолжения последовательности"""
    model.eval()
    device = next(model.parameters()).device
    
    char_to_id = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    id_to_char = {v: k for k, v in char_to_id.items()}
    
    input_ids = [char_to_id[char] for char in prompt.split()]
    context = torch.tensor([input_ids], device=device)
    
    print(f"\nКонтекст: {prompt} | Предсказание: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(5):  
            logits = model(context)
            next_token_id = logits[0, -1, :].argmax().item()
            
            print(f"{id_to_char[next_token_id]} ", end="", flush=True)
            
            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            context = torch.cat([context, next_token_tensor], dim=1)
    print("\n")


if __name__ == "__main__":
    trained_model = train_model()
    
    test_prompt = "A B A C D"
    generate_sequence(trained_model, test_prompt)