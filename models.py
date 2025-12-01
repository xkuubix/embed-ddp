# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ImagePatcher import ImagePatcher
from timm import create_model


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class GatedAttentionMIL(nn.Module):

    def __init__(
                self,
                num_classes=1,
                backbone='r18',
                pretrained=True,
                L=512,
                D=128,
                K=1,
                feature_dropout=0.1,
                attention_dropout=0.1,
                config=None
                ):

        super().__init__()
        self.L = L  
        self.D = D
        self.K = K
        assert K == 1, "only supports K=1"
        self.num_classes = 1
        print(f"Building GatedAttentionMIL with backbone={backbone}, pretrained={pretrained}")
        if backbone == 'r18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.feature_extractor = models.resnet18(weights=weights)
        elif backbone == 'r34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.feature_extractor = models.resnet34(weights=weights)
        elif backbone == 'r50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.feature_extractor = models.resnet50(weights=weights)
        elif backbone == 'convnext_base':
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.feature_extractor = models.convnext_base(weights=weights)
        elif backbone == 'convnext_large':
            weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
            self.feature_extractor = models.convnext_large(weights=weights)
        elif backbone == 'convnext_tiny':
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.feature_extractor = models.convnext_tiny(weights=weights)
        elif backbone == 'convnext_small':
            # Nguyen 2023 kaggle comp 1st place checkpoint
            pth = '/users/project1/pt01190/EUCAIM-PG-GUM/embed_ddp/checkpoints/best_convnext_fold_0.pth.tar'
            ckpt = torch.load(pth, map_location="cpu", weights_only=False)

            self.feature_extractor = create_model(
                "convnext_small",
                pretrained=False,
                num_classes=1,          # binary classifier
                drop_path_rate=0.2,
            )

            state = ckpt["state_dict"]       # this key exists in their file
            new_state = {}

            for k, v in state.items():
                k2 = k.replace("module.", "")     # strip DDP wrapper
                new_state[k2] = v

            self.feature_extractor.load_state_dict(new_state, strict=True)

        if backbone.startswith('r'):    
            self.num_features = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = Identity()
        elif backbone.startswith('convnext_small'):
            self.num_features = self.feature_extractor.head.fc.in_features
            self.feature_extractor.head.fc = Identity()
        elif backbone.startswith('convnext'):
            self.num_features = self.feature_extractor.classifier[2].in_features
            self.feature_extractor.classifier[2] = Identity()


        self.transform_features = nn.Linear(self.num_features, self.L)
        self.norm = nn.LayerNorm(self.L)

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Dropout(attention_dropout)
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
            nn.Dropout(attention_dropout)
        )
        self.attention_weights = nn.Linear(self.D, self.K)
        self.classifier = nn.Sequential(
                                        nn.Linear(self.L * self.K,
                                                  num_classes))
        self.feature_dropout = nn.Dropout(feature_dropout)

        self.patcher = ImagePatcher(patch_size=config['data']['patch_size'] if config else 128,
                                    overlap=config['data']['overlap'] if config else 0.25,
                                    empty_thresh=config['data']['empty_threshold'] if config else 0.75,
                                    bag_size=config['data']['bag_size'] if config else -1)

        self.reconstruct_attention = False

    def forward(self, x):
        device = x.device
        self.patcher.get_tiles(x.shape[2], x.shape[3])
        instances, instances_ids, _ = self.patcher.convert_img_to_bag(x.squeeze(0))
        instances = instances.unsqueeze(0)
        instances = self.norm_instances(instances)
        instances = instances.to(device)

        bs, num_instances, ch, w, h = instances.shape
        instances = instances.view(bs*num_instances, ch, w, h)
        H = self.feature_extractor(instances)
        H = self.feature_dropout(H)
        H = self.transform_features(H)
        H = self.norm(H)
        H = H.view(bs, num_instances, -1)
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(torch.mul(A_V, A_U))
        A = torch.transpose(A, 2, 1)
        A = F.softmax(A, dim=2)
        m = torch.matmul(A, H)
        Y = self.classifier(m)

        if self.reconstruct_attention:
            A = self.patcher.reconstruct_attention_map(A.unsqueeze(0), instances_ids, x.shape[2:])
            return Y, A
        return Y.squeeze(0)

    @staticmethod
    def norm_instances(instances):
        """
        Normalize instances
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=instances.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=instances.device).view(1, 1, 3, 1, 1)
        return (instances - mean) / std