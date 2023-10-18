from transformers import AutoTokenizer, T5EncoderModel
import torch
import torch.nn as nn


class Grid_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(10, 20, 5, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        feature_map = self.encoder(x)

        return feature_map

class prototype_fusion_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Grid_Encoder()
        self.encoder.state_dict(torch.load('auto_200.pt'))
        self.t5_encoder = T5EncoderModel.from_pretrained("t5-small")
        self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

        self.resize_layer1 = nn.Linear(484,512)
        self.resize_layer2 = nn.Linear(120, 20)
        self.fusion_layer = nn.Linear(512,484)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
        )

        self.model_freeze(self.encoder)
        self.model_freeze(self.t5_encoder)

    def forward(self, x, LARC_description):
        # ========== Encoder ==========
        feature_map = self.encoder(x)
        feature_map = feature_map.view(20, -1)
        resize_feature_map = self.resize_layer1(feature_map)
        # ========== T5 Encoder ==========
        input_ids = self.t5_tokenizer(LARC_description, return_tensors="pt", padding='max_length', max_length=100, truncation=True,).input_ids.to('cuda') # Batch size 1
        outputs = self.t5_encoder(input_ids=input_ids)
        LARC_representation = outputs.last_hidden_state.squeeze()

        # ========== concat & fusion layer ==========
        concat_feature = torch.cat((resize_feature_map, LARC_representation))
        fustion_feature = self.fusion_layer(concat_feature)

        # ========== CNN decoder ==========
        resize_fusion_feature1 = fustion_feature.view(-1, 22, 22)
        permute_resize_fusion_feature = resize_fusion_feature1.permute(1,2,0)
        resize_fusion_feature2 = self.resize_layer2(permute_resize_fusion_feature)
        final_resize = resize_fusion_feature2.permute(2,0,1)

        result = self.decoder(final_resize)

        return result
