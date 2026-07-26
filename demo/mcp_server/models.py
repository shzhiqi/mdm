import torch.nn as nn
import clip
import torch

class PerceptionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.relu1 = nn.ReLU(True)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

class EmotionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout()

        self.fc2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout()

        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        embed = x

        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x, embed

class ReasonModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout()

        self.fc2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout()

        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.fc1(x)
        
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        embed = x

        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x, embed

class MultiTaskModelV3(nn.Module):
    # only emotion
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.backbone, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.backbone.to(torch.float32)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.guess_module = PerceptionModule().to(device)
        self.sharing_module = PerceptionModule().to(device)
        self.believable_module = PerceptionModule().to(device)

        self.happy_module = EmotionModule().to(device)
        self.layer_norm = nn.LayerNorm(1024).to(device)
        # self.impressive_module = EmotionModule().to(device)
        self.weight_happy = nn.Parameter(torch.tensor(1.))
        self.weight_embed = nn.Parameter(torch.tensor(1.))

    def forward(self, images, texts):
        # b, n_dim
        image_features = self.backbone.encode_image(images)

        # b_ n_dim
        text_features = self.backbone.encode_text(texts)

        # print('image_feaures', image_features.shape, 'text_features', text_features.shape)

        # Nx1024x1 * N*1*2 = Nx1024
        multi_feat = torch.cat((text_features, image_features), dim=1)

        pred_happy, emotional_embed = self.happy_module(multi_feat)

        # pred_impressive = self.impressive_module(multi_feat)
        # pred_sad = self.sad_module(multi_feat)

        enhanced_embed = self.layer_norm(self.weight_happy*emotional_embed + self.weight_embed * multi_feat)

        pred_guess = self.guess_module(enhanced_embed)
        pred_sharing = self.sharing_module(enhanced_embed)
        pred_believable = self.believable_module(enhanced_embed)

        return pred_guess, pred_sharing, pred_believable, pred_happy

class MultiTaskModelV3_2(nn.Module):
    # only emotion
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.backbone, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.backbone.to(torch.float32)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.guess_module = PerceptionModule().to(device)
        self.sharing_module = PerceptionModule().to(device)
        self.believable_module = PerceptionModule().to(device)

        self.emo_consist_module = EmotionModule().to(device)
        self.sem_consist_module = EmotionModule().to(device)

        self.layer_norm = nn.LayerNorm(1024).to(device)
        # self.impressive_module = EmotionModule().to(device)
        self.weight_emo_consist = nn.Parameter(torch.tensor(1.))
        self.weight_sem_consist = nn.Parameter(torch.tensor(1.))
        self.weight_embed = nn.Parameter(torch.tensor(1.))

    def forward(self, images, texts):
        # b, n_dim
        image_features = self.backbone.encode_image(images)

        # b_ n_dim
        text_features = self.backbone.encode_text(texts)

        # print('image_feaures', image_features.shape, 'text_features', text_features.shape)

        # Nx1024x1 * N*1*2 = Nx1024
        multi_feat = torch.cat((text_features, image_features), dim=1)

        pred_emo, emo_embed = self.emo_consist_module(multi_feat)

        pred_sem, sem_embed = self.emo_consist_module(multi_feat)

        # pred_impressive = self.impressive_module(multi_feat)
        # pred_sad = self.sad_module(multi_feat)

        enhanced_embed = self.layer_norm(
            self.weight_emo_consist * emo_embed + \
            self.weight_embed * multi_feat + \
            self.weight_sem_consist * sem_embed)

        pred_guess = self.guess_module(enhanced_embed)
        pred_sharing = self.sharing_module(enhanced_embed)
        pred_believable = self.believable_module(enhanced_embed)

        return pred_guess, pred_sharing, pred_believable, pred_emo, pred_sem


class MultiTaskModelV4(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.backbone, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.backbone.to(torch.float32)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.guess_module = PerceptionModule().to(device)
        self.sharing_module = PerceptionModule().to(device)
        self.believable_module = PerceptionModule().to(device)

        self.happy_module = EmotionModule().to(device)
        self.layer_norm = nn.LayerNorm(1024).to(device)
        self.reason_module = ReasonModule().to(device)
        # self.impressive_module = EmotionModule().to(device)
        self.weight_happy = nn.Parameter(torch.tensor(1.))
        self.weight_reason = nn.Parameter(torch.tensor(1.))
        self.weight_embed = nn.Parameter(torch.tensor(1.))

    def forward(self, images, texts):
        # b, n_dim
        image_features = self.backbone.encode_image(images)

        # b_ n_dim
        text_features = self.backbone.encode_text(texts)

        # print('image_feaures', image_features.shape, 'text_features', text_features.shape)

        # Nx1024x1 * N*1*2 = Nx1024
        multi_feat = torch.cat((text_features, image_features), dim=1)

        pred_happy, emotional_embed = self.happy_module(multi_feat)

        pred_reason, reason_embed = self.reason_module(multi_feat)
        # pred_impressive = self.impressive_module(multi_feat)
        # pred_sad = self.sad_module(multi_feat)

        enhanced_embed = self.layer_norm(self.weight_happy*emotional_embed + self.weight_embed * multi_feat + self.weight_reason*reason_embed)

        pred_guess = self.guess_module(enhanced_embed)
        pred_sharing = self.sharing_module(enhanced_embed)
        pred_believable = self.believable_module(enhanced_embed)

        return pred_guess, pred_sharing, pred_believable, pred_happy, pred_reason

class MultiTaskModelV4_MediaEval(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.backbone, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.backbone.to(torch.float32)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.guess_module = PerceptionModule().to(device)
        self.sharing_module = PerceptionModule().to(device)
        self.believable_module = PerceptionModule().to(device)

        self.happy_module = EmotionModule().to(device)
        self.layer_norm = nn.LayerNorm(1024).to(device)
        self.reason_module = ReasonModule().to(device)
        # self.impressive_module = EmotionModule().to(device)
        self.weight_happy = nn.Parameter(torch.tensor(1.))
        self.weight_reason = nn.Parameter(torch.tensor(1.))
        self.weight_embed = nn.Parameter(torch.tensor(1.))

    def forward(self, images, texts):
        # b, n_dim
        image_features = self.backbone.encode_image(images)

        # b_ n_dim
        text_features = self.backbone.encode_text(texts)

        # print('image_feaures', image_features.shape, 'text_features', text_features.shape)

        # Nx1024x1 * N*1*2 = Nx1024
        multi_feat = torch.cat((text_features, image_features), dim=1)

        pred_happy, emotional_embed = self.happy_module(multi_feat)

        pred_reason, reason_embed = self.reason_module(multi_feat)
        # pred_impressive = self.impressive_module(multi_feat)
        # pred_sad = self.sad_module(multi_feat)

        enhanced_embed = self.layer_norm(self.weight_happy*emotional_embed + self.weight_embed * multi_feat + self.weight_reason*reason_embed)

        pred_guess = self.guess_module(enhanced_embed)
        pred_sharing = self.sharing_module(enhanced_embed)
        pred_believable = self.believable_module(enhanced_embed)

        return pred_guess, pred_sharing, pred_believable, pred_happy, pred_reason