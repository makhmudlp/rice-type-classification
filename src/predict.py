import torch
from RiceTypeClassificationBeginners.src.model import MyModel

def predict_single(model_path, input_features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyModel(input_dim=len(input_features)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = torch.tensor(input_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        prediction = model(x)
    
    return prediction.item()
