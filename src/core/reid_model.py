# /top_eye/src/core/reid_model.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from collections import OrderedDict


class StrongReIDModel:
    """Сильная ReID модель с несколькими backbone"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.weights = [0.4, 0.3, 0.3]  # Веса для ансамбля

        # Инициализация моделей
        self._init_models()

        # Трансформы
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _init_models(self):
        """Инициализация ансамбля моделей"""
        # 1. ResNet50
        model1 = models.resnet50(pretrained=True)
        model1.fc = nn.Linear(2048, 512)
        self._load_weights_if_exists(model1, 'resnet50_reid.pth')

        # 2. EfficientNet
        try:
            from efficientnet_pytorch import EfficientNet
            model2 = EfficientNet.from_pretrained('efficientnet-b0')
            model2._fc = nn.Linear(1280, 512)
            self._load_weights_if_exists(model2, 'efficientnet_reid.pth')
        except:
            # Fallback to another model
            model2 = models.mobilenet_v3_large(pretrained=True)
            model2.classifier[3] = nn.Linear(1280, 512)

        # 3. OSNet (лучшая для ReID)
        model3 = self._create_osnet()

        # Переводим в eval mode
        for model in [model1, model2, model3]:
            model.to(self.device)
            model.eval()
            self.models.append(model)

    def _create_osnet(self):
        """Создание OSNet архитектуры"""

        class OSNet(nn.Module):
            def __init__(self):
                super(OSNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

                # Упрощенная архитектура
                self.layer1 = self._make_layer(64, 64, 2)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 2, stride=2)
                self.layer4 = self._make_layer(256, 512, 2, stride=2)

                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, 512)

            def _make_layer(self, in_channels, out_channels, blocks, stride=1):
                layers = []
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                        stride=stride, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))

                for _ in range(1, blocks):
                    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                            stride=1, padding=1, bias=False))
                    layers.append(nn.BatchNorm2d(out_channels))
                    layers.append(nn.ReLU(inplace=True))

                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)

                return x

        model = OSNet()
        self._load_weights_if_exists(model, 'osnet_reid.pth')
        return model

    def _load_weights_if_exists(self, model, weight_path):
        """Загрузка весов если есть"""
        full_path = f"models/reid/{weight_path}"
        try:
            if os.path.exists(full_path):
                state_dict = torch.load(full_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"✓ Загружены веса {weight_path}")
        except:
            print(f"⚠ Не удалось загрузить {weight_path}, используются предобученные")

    def extract_embedding(self, image):
        """Извлечение эмбеддинга из изображения"""
        if image is None or image.size == 0:
            return None

        try:
            # Препроцессинг
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image_rgb).unsqueeze(0).to(self.device)

            # Ансамбль предсказаний
            embeddings = []

            with torch.no_grad():
                for model in self.models:
                    emb = model(transformed)
                    emb = nn.functional.normalize(emb, p=2, dim=1)
                    embeddings.append(emb.cpu().numpy())

            # Взвешенное усреднение
            final_embedding = np.zeros_like(embeddings[0])
            for emb, weight in zip(embeddings, self.weights):
                final_embedding += emb * weight

            # Нормализация
            final_embedding = final_embedding.flatten()
            norm = np.linalg.norm(final_embedding)
            if norm > 0:
                final_embedding = final_embedding / norm

            return final_embedding

        except Exception as e:
            print(f"Ошибка извлечения эмбеддинга: {e}")
            return None

    def compute_similarity(self, emb1, emb2):
        """Вычисление схожести между эмбеддингами"""
        if emb1 is None or emb2 is None:
            return 0
        return 1 - cosine(emb1, emb2)