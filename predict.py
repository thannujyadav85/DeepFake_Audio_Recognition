
# Deepfake Audio Detection - Standalone Inference Script


import sys
import os
import torch
import librosa
import numpy as np
import torch.nn as nn

# Replicate the exact model architecture from training
class DeepfakeAudioClassifier(nn.Module):
    def __init__(self):
        super(DeepfakeAudioClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def run_inference(audio_path, model_path='best_deepfake_detector.pth'):
    
    # Loads model weights, processes the input audio file, and outputs detection outcomes.
    # Note: Probabilistic threshold is calibrated to 0.5000 (equivalent to logit 0.0000).
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' could not be located.")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"Error: Model weight file '{model_path}' not found in the working directory.")
        sys.exit(1)

    # 1. Device Setup (Graceful CPU fallback if user lacks a GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Reconstruct Model Context
    model = DeepfakeAudioClassifier()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model state dict: {str(e)}")
        sys.exit(1)

    try:
        # 3. Audio Preprocessing Pipeline (3-second strict norm)
        sr, duration = 16000, 3
        target_length = sr * duration
        
        audio, _ = librosa.load(audio_path, sr=sr)
        
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        else:
            audio = audio[:target_length]

        # 4. Mel-Spectrogram Feature Transformation Matrix
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Scale range mapping to matches training inputs [-1, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        mel_spec_db = (mel_spec_db * 2) - 1

        # Format input tensor layout: (Batch=1, Channels=1, Freq=128, Time=94)
        X = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # 5. Model Execution
        with torch.no_grad():
            output = model(X).squeeze(1)
            raw_logit = output.item()
            probability = torch.sigmoid(output).item()

    except Exception as e:
        print(f"Fatal error while parsing audio stream data: {str(e)}")
        sys.exit(1)

    # 6. Evaluate Against Calibrated Decision Matrix Threshold
    # 0 = Genuine, 1 = Deepfake. Probability range is [0.0, 1.0]
    threshold = 0.5000
    
    if probability >= threshold:
        classification = "DEEPFAKE (AI-Generated Speech)"
        confidence = probability  # Confidence in it being fake
    else:
        classification = "GENUINE (Human Speech)"
        confidence = 1.0 - probability  # Confidence in it being real

    # Print clean terminal report layout
    print("\n" + "="*50)
    print("        DEEPFAKE AUDIO DETECTION SYSTEM ENGINE        ")
    print("="*50)
    print(f" Target File       : {os.path.basename(audio_path)}")
    print(f" Algorithmic Output: {classification}")
    print(f" Analysis Confidence : {confidence * 100:.2f}%")
    print(f" Raw Output Logit  : {raw_logit:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file.wav>")
        sys.exit(1)
        
    target_audio = sys.argv[1]
    run_inference(target_audio)