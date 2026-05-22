import torch
import torch.nn as nn
from typing import Optional

def calculate_smoothing_factors(
    calib_activations: torch.Tensor, 
    layer_weights: torch.Tensor, 
    alpha: float = 0.5, 
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Вычисляет коэффициенты сглаживания (SmoothQuant) для балансировки выбросов 
    между активациями и весами линейного слоя.
    """
    reduce_dims = tuple(range(calib_activations.ndim - 1))
    act_max = calib_activations.abs().amax(dim=reduce_dims)
    
    weight_max = layer_weights.abs().amax(dim=0)
    
    # s = max(X)^alpha / max(W)^(1-alpha)
    numerator = torch.pow(act_max, alpha)
    denominator = torch.pow(weight_max, 1.0 - alpha).clamp(min=eps)
    
    return numerator / denominator


class SmoothedLinear(nn.Module):
    """
    Обертка над nn.Linear, которая применяет SmoothQuant сглаживание.
    Впитывает коэффициенты в веса при инициализации и масштабирует входы при инференсе.
    """
    def __init__(self, base_layer: nn.Linear, smoothing_scales: torch.Tensor):
        super().__init__()
        
        import copy
        self.layer = copy.deepcopy(base_layer)
        
        
        with torch.no_grad():
            self.layer.weight.mul_(smoothing_scales)
            
        self.register_buffer('inv_scales', 1.0 / smoothing_scales)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Применяем обратное масштабирование к активациям: X' = X * diag(s^-1)
        smoothed_x = x * self.inv_scales
        return self.layer(smoothed_x)

if __name__ == "__main__":
    torch.manual_seed(42)

    in_features = 8
    out_features = 4
    
    # Исходная модель
    original_linear = nn.Linear(in_features, out_features, bias=False)

    X_calib = torch.randn(2, 5, in_features)
    X_calib[:, :, 2] *= 100.0 

    # Получаем исходные предсказания
    with torch.no_grad():
        y_baseline = original_linear(X_calib)

    # 1. Считаем факторы сглаживания
    scales = calculate_smoothing_factors(X_calib, original_linear.weight.data, alpha=0.5)

    # 2. Создаем сглаженный слой 
    smoothed_layer = SmoothedLinear(original_linear, scales)

    # 3. Делаем проход через новую модель
    with torch.no_grad():
        y_smoothed = smoothed_layer(X_calib)

    # 4. Сравниваем распределения
    print("--- Анализ активаций ---")
    print(f"Максимум в исходных активациях (X):  {X_calib.abs().max().item():.4f}")
    
    x_input_smoothed = X_calib * smoothed_layer.inv_scales
    print(f"Максимум в сглаженных активациях (X'): {x_input_smoothed.abs().max().item():.4f}\n")

    print("--- Проверка математической эквивалентности ---")
    max_error = torch.max(torch.abs(y_baseline - y_smoothed)).item()
    print(f"Максимальная разница результатов (ошибка): {max_error:.6e}")