import torch
import os
from U_net_optimized import OptimizedUNet  

def save_model(model_path=None):
    # Initialize the model
    model = OptimizedUNet()
    
    # If you have trained weights, load them
    if model_path is not None:
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
    
    # Save the model
    save_path = os.path.join(os.getcwd(), 'saved_unet_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    # If you have trained weights, specify the path
    # save_model('path_to_your_weights.pth')
    
    # If you want to save the model with random initialization
    save_model()