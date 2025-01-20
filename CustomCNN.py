from torch import nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        self.model = nn.Sequential(
            # Pierwsza warstwa splotowa
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Rozmiar zmniejsza się: 224x224 -> 112x112
            
            # Druga warstwa splotowa
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Rozmiar zmniejsza się: 112x112 -> 56x56
            
            # Trzecia warstwa splotowa
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Rozmiar zmniejsza się: 56x56 -> 28x28
            
            # Czwarta warstwa splotowa
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Rozmiar zmniejsza się: 28x28 -> 14x14
            
            # Piąta warstwa splotowa
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Rozmiar zmniejsza się: 14x14 -> 7x7
            
            # Spłaszczenie
            nn.Flatten(),
            
            # Warstwy w pełni połączone
            nn.Linear(256 * 7 * 7, 512),  # 256 kanałów, każdy o rozmiarze 7x7
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, num_classes)  # Liczba wyjściowych klas
        )

    def forward(self, x):
        return self.model(x)


class CustomCNN2(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN2, self).__init__()
        
        self.model = nn.Sequential(
            # Pierwsza warstwa splotowa
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Rozmiar zmniejsza się: 224x224 -> 112x112
            
            # Druga warstwa splotowa
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Rozmiar zmniejsza się: 112x112 -> 56x56
            
            # Trzecia warstwa splotowa
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Rozmiar zmniejsza się: 56x56 -> 28x28
            
            # Spłaszczenie
            nn.Flatten(),
            
            # Warstwy w pełni połączone
            nn.Linear(64*28*28, 512),  # 256 kanałów, każdy o rozmiarze 7x7
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, num_classes)  # Liczba wyjściowych klas
        )

    def forward(self, x):
        return self.model(x)
