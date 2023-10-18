import torch
import torch.nn as nn
from model import *

class Action_batch1_embedding(nn.Module):
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

        self.encoder_embedding = nn.Linear(20*22*22, 512)

        self.decompose_embedding = nn.Linear(512, 20*22*22)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 10, kernel_size = 5, stride = 1, padding=0),
            nn.ReLU(),
            )

        self.action_vector = nn.Parameter(torch.ones(512))
        self.proj = nn.Linear(10, 11)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def auto_encoder_pretrain(self, model_name, pretrain):
        self.autoencoder = globals()[model_name]().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pretrain))
        self.auto_encoder_freeze()

    def forward(self, x):
        feature_map = self.encoder(x)
        embedding_vector = self.encoder_embedding(feature_map.reshape(1, -1))
        new_latent_vector = embedding_vector + self.action_vector
        decompose_vector = self.decompose_embedding(new_latent_vector)
        output = self.decoder(decompose_vector)

        return output

class Action_origin(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_vector = nn.Parameter(torch.ones((20,22,22)))
        self.proj = nn.Linear(10, 11)
        self.norm_layer = nn.BatchNorm1d(22)
        self.relu = nn.ReLU()

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.proj.requires_grad_(False)

    def auto_encoder_pretrain(self, model_name, pretrain):
        self.autoencoder = globals()[model_name]().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pretrain))
        self.auto_encoder_freeze()

    def forward(self, x):
        feature_map = self.autoencoder.encoder(x)
        decoder_input = feature_map + self.action_vector
        decoder_input = self.norm_layer(decoder_input)
        decoder_input = self.relu(decoder_input)
        output = self.autoencoder.decoder(decoder_input)

        return output

class Action_origin_complexity(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_vector = nn.Parameter(torch.ones((20,22,22)))
        self.proj = nn.Linear(10, 11)
        self.relu = nn.ReLU()
        self.complexity_layer1 = nn.Linear(20*22*22, 10*22*22)
        self.complexity_layer2 = nn.Linear(10 * 22 * 22, 5 * 22 * 22)
        self.complexity_layer3 = nn.Linear(5 * 22 * 22, 10 * 22 * 22)
        self.complexity_layer4 = nn.Linear(10*22*22, 20*22*22)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.proj.requires_grad_(False)

    def auto_encoder_pretrain(self, model_name, pretrain):
        self.autoencoder = globals()[model_name]().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pretrain))
        self.auto_encoder_freeze()

    def improve_complexity(self, feature_map):
        new_feature = feature_map + self.action_vector
        new_feature = new_feature.reshape(-1)
        complexity_feature = self.complexity_layer1(new_feature)
        complexity_feature = self.relu(complexity_feature)
        complexity_feature = self.complexity_layer2(complexity_feature)
        complexity_feature = self.relu(complexity_feature)
        complexity_feature = self.complexity_layer3(complexity_feature)
        complexity_feature = self.relu(complexity_feature)
        complexity_feature = self.complexity_layer4(complexity_feature)
        complexity_feature = self.relu(complexity_feature)
        complexity_feature = complexity_feature.reshape(20, 22, 22)
        return complexity_feature

    def forward(self, x):
        feature_map = self.autoencoder.encoder(x)
        complexity_action_vector = self.improve_complexity(feature_map)
        output = self.autoencoder.decoder(complexity_action_vector)

        return output

class Action_batch1_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = None
        self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
        self.action_decoder1 = nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0)
        self.action_decoder2 = nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(1, 11)
        # ============================ version 2 ============================
        self.action_linear1 = nn.Linear(676,676)
        self.action_linear2 = nn.Linear(900,900)
        self.leaky_relu = nn.LeakyReLU()
        self.last_layer = nn.Linear(10, 1)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def auto_encoder_pretrain(self, model_name, pretrain):
        self.autoencoder = globals()[model_name]().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pretrain))
        self.auto_encoder_freeze()

    def forward(self, x):
        feature_map1 = self.autoencoder.encoder(x)
        new_feature_map1 = feature_map1 + self.action_vector
        output = self.action_decoder1(new_feature_map1)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        # output_size = output.shape
        # # output = output.view(output_size[0], -1)
        # output = output.view(output_size[0], output_size[0], -1)
        # output = self.action_linear1(output)
        # output = output.view(output_size)
        # output = self.relu(output)

        # output = self.leaky_relu(output)

        # ============================ layer ============================


        feature_map2 = self.autoencoder.decoder[0](feature_map1)
        new_feature_map2 = feature_map2 + output
        output = self.action_decoder2(new_feature_map2)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        # output_size = output.shape
        # # output = output.view(output_size[0], -1)
        # output = output.view(output_size[0], -1)
        # output = self.action_linear2(output)
        # output = output.view(output_size)
        # output = self.relu(output)
        # # output = self.leaky_relu(output)

        # ============================ layer ============================

        feature_map3 = self.autoencoder.decoder[2](feature_map2)
        new_feature_map3 = feature_map3 + output
        result = self.relu(new_feature_map3)
        # output = self.leaky_relu(new_feature_map3)
        # ============================ version 2 ============================
        # output_size = output.shape
        # output = output.view(output_size[0],  -1)
        # output = self.action_linear2(output)
        # output = output.view(output_size).squeeze()
        # output = self.relu(output)
        #
        # output = output.permute(1,2,0)
        # output = self.last_layer(output)
        # output = output.permute(2,0,1)
        # result = self.relu(output)
        # result = self.leaky_relu(output)

        return result

class Action_batch1_v1_c10(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = None
        self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
        self.action_decoder1 = nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0)
        self.action_decoder2 = nn.ConvTranspose2d(10, 10, kernel_size = 5, stride = 1, padding=0)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(10, 11)
        # ============================ version 2 ============================
        self.action_linear1 = nn.Linear(676,676)
        self.action_linear2 = nn.Linear(900,900)
        self.leaky_relu = nn.LeakyReLU()
        self.last_layer = nn.Linear(10, 1)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def auto_encoder_pretrain(self, model_name, pretrain):
        self.autoencoder = globals()[model_name]().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pretrain))
        self.auto_encoder_freeze()

    def forward(self, x):
        feature_map1 = self.autoencoder.encoder(x)
        new_feature_map1 = feature_map1 + self.action_vector
        output = self.action_decoder1(new_feature_map1)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        # output_size = output.shape
        # # output = output.view(output_size[0], -1)
        # output = output.view(output_size[0], output_size[0], -1)
        # output = self.action_linear1(output)
        # output = output.view(output_size)
        # output = self.relu(output)

        # output = self.leaky_relu(output)

        # ============================ layer ============================


        feature_map2 = self.autoencoder.decoder[0](feature_map1)
        new_feature_map2 = feature_map2 + output
        output = self.action_decoder2(new_feature_map2)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        # output_size = output.shape
        # # output = output.view(output_size[0], -1)
        # output = output.view(output_size[0], -1)
        # output = self.action_linear2(output)
        # output = output.view(output_size)
        # output = self.relu(output)
        # # output = self.leaky_relu(output)

        # ============================ layer ============================

        feature_map3 = self.autoencoder.decoder[2](feature_map2)
        new_feature_map3 = feature_map3 + output
        result = self.relu(new_feature_map3)
        # output = self.leaky_relu(new_feature_map3)
        # ============================ version 2 ============================
        # output_size = output.shape
        # output = output.view(output_size[0],  -1)
        # output = self.action_linear2(output)
        # output = output.view(output_size).squeeze()
        # output = self.relu(output)
        #
        # output = output.permute(1,2,0)
        # output = self.last_layer(output)
        # output = output.permute(2,0,1)
        # result = self.relu(output)
        # result = self.leaky_relu(output)

        return result

class Action_batch1_v1_c10_concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = None
        self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
        self.action_decoder1 = nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0)
        self.action_decoder2 = nn.ConvTranspose2d(10, 10, kernel_size = 5, stride = 1, padding=0)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(10, 11)
        # ============================ version 2 ============================
        self.action_linear1 = nn.Linear(676,676)
        self.action_linear2 = nn.Linear(900,900)
        self.leaky_relu = nn.LeakyReLU()
        self.last_layer = nn.Linear(10, 1)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def auto_encoder_pretrain(self, model_name, pretrain):
        self.autoencoder = globals()[model_name]().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pretrain))
        self.auto_encoder_freeze()

    def forward(self, x):
        feature_map1 = self.autoencoder.encoder(x)
        new_feature_map1 = feature_map1 + self.action_vector
        output = self.action_decoder1(new_feature_map1)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        output_size = output.shape
        # output = output.view(output_size[0], -1)
        output = output.view(output_size[0], output_size[0], -1)
        output = self.action_linear1(output)
        output = output.view(output_size)
        output = self.relu(output)

        # output = self.leaky_relu(output)

        # ============================ layer ============================


        feature_map2 = self.autoencoder.decoder[0](feature_map1)
        new_feature_map2 = feature_map2 + output
        output = self.action_decoder2(new_feature_map2)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        output_size = output.shape
        # output = output.view(output_size[0], -1)
        output = output.view(output_size[0], -1)
        output = self.action_linear2(output)
        output = output.view(output_size)
        output = self.relu(output)
        # output = self.leaky_relu(output)

        # ============================ layer ============================

        feature_map3 = self.autoencoder.decoder[2](feature_map2)
        new_feature_map3 = feature_map3 + output
        result = self.relu(new_feature_map3)
        # output = self.leaky_relu(new_feature_map3)
        # ============================ version 2 ============================
        output_size = output.shape
        output = output.view(output_size[0],  -1)
        output = self.action_linear2(output)
        output = output.view(output_size).squeeze()
        output = self.relu(output)

        output = output.permute(1,2,0)
        output = self.last_layer(output)
        output = output.permute(2,0,1)
        result = self.relu(output)
        # result = self.leaky_relu(output)

        return result



class Action_batch1_v1_concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = None
        self.action_vector = nn.Parameter(torch.ones((20,22,22)))
        self.action_decoder1 = nn.ConvTranspose2d(40, 10, kernel_size = 5, stride = 1, padding=0)
        self.action_decoder2 = nn.ConvTranspose2d(20, 1, kernel_size = 5, stride = 1, padding=0)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(1, 11)
        # ============================ version 2 ============================
        self.action_linear1 = nn.Linear(676,676)
        self.action_linear2 = nn.Linear(900,900)
        self.leaky_relu = nn.LeakyReLU()
        self.last_layer = nn.Linear(10, 1)

        # ============================ version 3 ============================
        self.skip_layer1 = nn.Linear(26*26, 26*26)
        self.skip_layer2 = nn.Linear(30*30, 30*30)

        self.fusion_layer1 = nn.Linear(20, 10)
        self.fusion_layer2 = nn.Linear(2, 1)
        self.elu = nn.ELU()

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def auto_encoder_pretrain(self, model_name, pretrain):
        self.autoencoder = globals()[model_name]().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pretrain))
        self.auto_encoder_freeze()

    def forward(self, x):
        feature_map1 = self.autoencoder.encoder(x)
        if len(feature_map1.shape) != 3:
            new_feature_map1 = torch.cat((feature_map1, self.action_vector.unsqueeze(0).repeat(feature_map1.shape[0], 1, 1, 1)), dim=1)
        else:
            new_feature_map1 = torch.cat((feature_map1, self.action_vector))
        output = self.action_decoder1(new_feature_map1)
        output = self.elu(output)

        # ============================ version 2 ============================
        # if len(feature_map1.shape) != 3:
        #     output_size = output.shape
        #     # output = output.view(output_size[0], -1)
        #     output = output.view(output_size[0], output_size[1], -1)
        #     output = self.action_linear1(output)
        #     output = output.view(output_size)
        #     output = self.relu(output)
        # else:
        #     output_size = output.shape
        #     # output = output.view(output_size[0], -1)
        #     output = output.view(output_size[0], output_size[0], -1)
        #     output = self.action_linear1(output)
        #     output = output.view(output_size)
        #     output = self.relu(output)

        # ============================ version 3 ============================
        output2 = self.skip_layer1(output.view(-1, 26*26))
        if len(feature_map1.shape) != 3:
            output2 = self.elu(output2).view(-1, 10, 26, 26)
            skip_concat_output = torch.cat((output, output2), dim=1).transpose(1, 3)
            new_output = self.fusion_layer1(skip_concat_output)

            output = new_output.transpose(1, 3)

        else:
            output2 = self.elu(output2).view(-1, 26, 26)

            skip_concat_output = torch.cat((output, output2)).transpose(0, 2)
            new_output = self.fusion_layer1(skip_concat_output)

            output = new_output.transpose(0, 2)


        # ============================ layer ============================


        feature_map2 = self.autoencoder.decoder[0](feature_map1)
        if len(feature_map1.shape) != 3:
            new_feature_map2 = torch.cat((feature_map2, output), dim=1)
        else:
            new_feature_map2 = torch.cat((feature_map2, output))
        output = self.action_decoder2(new_feature_map2)
        output = self.elu(output)

        # ============================ version 2 ============================
        # if len(feature_map1.shape) != 3:
        #     output_size = output.shape
        #     # output = output.view(output_size[0], -1)
        #     output = output.view(output_size[0], -1)
        #     output = self.action_linear2(output)
        #     output = output.view(output_size)
        #     output = self.relu(output)
        # else:
        #     output_size = output.shape
        #     # output = output.view(output_size[0], -1)
        #     output = output.view(output_size[0], -1)
        #     output = self.action_linear2(output)
        #     output = output.view(output_size)
        #     output = self.relu(output)
        #     # output = self.leaky_relu(output)

        # ============================ version 3 ============================
        output3 = self.skip_layer2(output.reshape(-1, 30 * 30))
        if len(feature_map1.shape) != 3:
            output3 = self.elu(output3).view(-1, 1, 30, 30)
            skip_concat_output = torch.cat((output, output3), dim=1).transpose(1, 3)
            new_output = self.fusion_layer2(skip_concat_output)

            output = new_output.transpose(1, 3)

        else:
            output3 = self.elu(output3).view(-1, 30, 30)

            skip_concat_output = torch.cat((output, output3)).transpose(0,2)
            new_output = self.fusion_layer2(skip_concat_output)

            output = new_output.transpose(0,2)



        # ============================ layer ============================

        # feature_map3 = self.autoencoder.decoder[2](feature_map2)
        # new_feature_map3 = feature_map3 + output
        # result = self.relu(new_feature_map3)
        #
        # output = self.leaky_relu(new_feature_map3)
        # ============================ version 2 ============================
        # if len(feature_map1.shape) != 3:
        #     output_size = output.shape
        #     output = output.view(output_size[0], -1)
        #     output = self.action_linear2(output)
        #     output = output.view(output_size).squeeze()
        #     # output = self.relu(output)
        #     #
        #     # output = output.permute(1, 2, 0)
        #     # output = self.last_layer(output)
        #     # output = output.permute(2, 0, 1)
        #     # result = self.relu(output)
        #     result = self.leaky_relu(output)
        # else:
        #     output_size = output.shape
        #     output = output.view(output_size[0],  -1)
        #     output = self.action_linear2(output)
        #     output = output.view(output_size).squeeze()
        #     output = self.relu(output)
        #
        #     output = output.permute(1,2,0)
        #     output = self.last_layer(output)
        #     output = output.permute(2,0,1)
        #     result = self.relu(output)
        #     # result = self.leaky_relu(output)

        result = output

        return result

class Action_batch1_v2(nn.Module):
    def __init__(self, pre_traind='auto_200.pt'):
        super().__init__()
        self.autoencoder = Autoencoder_batch1().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pre_traind))
        self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
        self.action_decoder1 = nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0)
        self.action_decoder2 = nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0)
        self.relu = nn.ReLU()
        # ============================ version 2 ============================
        self.action_linear1 = nn.Linear(676,676)
        self.action_linear2 = nn.Linear(900,900)
        self.leaky_relu = nn.LeakyReLU()
        self.last_layer = nn.Linear(10, 1)

        for param in self.autoencoder.parameters():
            param.requires_grad = False


    def forward(self, x):
        feature_map1 = self.autoencoder.encoder(x)
        new_feature_map1 = feature_map1 + self.action_vector
        output = self.action_decoder1(new_feature_map1)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        output_size = output.shape
        output = output.view(output_size[0], -1)
        output = self.action_linear1(output)
        output = output.view(output_size)
        output = self.relu(output)

        # output = self.leaky_relu(output)

        # ============================ layer ============================


        feature_map2 = self.autoencoder.decoder[0](feature_map1)
        new_feature_map2 = feature_map2 + output
        output = self.action_decoder2(new_feature_map2)
        output = self.relu(output)
        # output = self.leaky_relu(output)
        # ============================ version 2 ============================
        output_size = output.shape
        # output = output.view(output_size[0], -1)
        output = output.view(output_size[0], -1)
        output = self.action_linear2(output)
        output = output.view(output_size)
        output = self.relu(output)
        # output = self.leaky_relu(output)

        # ============================ layer ============================

        feature_map3 = self.autoencoder.decoder[2](feature_map2)
        new_feature_map3 = feature_map3 + output
        output = self.relu(new_feature_map3)
        # output = self.leaky_relu(new_feature_map3)
        # ============================ version 2 ============================
        output_size = output.shape
        output = output.view(output_size[0],  -1)
        output = self.action_linear2(output)
        output = output.view(output_size)#.squeeze()
        result = self.relu(output)

        # output = output.permute(1,2,0)
        # output = self.last_layer(output)
        # output = output.permute(2,0,1)
        # result = self.relu(output)
        # # result = self.leaky_relu(output)

        return result

class Action_batch1_v3(nn.Module):
    def __init__(self, pre_traind='auto_200.pt'):
        super().__init__()
        self.autoencoder = Autoencoder_batch1().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pre_traind))
        self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
        self.action_decoder1 = nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0)
        self.action_decoder2 = nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0)
        self.action_linear1 = nn.Linear(676,676)
        self.action_linear2 = nn.Linear(900,900)
        self.feature_linear1 = nn.Linear(676, 676)
        self.feature_linear2 = nn.Linear(900, 900)
        self.fusion_layer1 = nn.Linear(2*676, 676)
        self.fusion_layer2 = nn.Linear(2 * 900, 900)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()


        for param in self.autoencoder.parameters():
            param.requires_grad = False


    def forward(self, x):
        feature_map_origin = self.autoencoder.encoder(x)
        # new_feature_map1 = feature_map1# + self.action_vector
        output = self.action_decoder1(feature_map_origin)
        output = self.relu(output)

        output_size = output.shape
        output = output.view(output_size[0], -1)
        output = self.action_linear1(output)

        feature_map1 = self.autoencoder.decoder[0](feature_map_origin)
        feature1_size = feature_map1.shape
        feature1 = output.view(feature1_size[0], -1)
        feature1 = self.relu(feature1)

        fusion_feature1 = torch.cat((output,feature1),dim=1)
        output = self.fusion_layer1(fusion_feature1)
        output = self.relu(output)
        output = output.reshape(output_size)

        # ============================ layer1 ============================
        output = self.action_decoder2(output)
        output = self.relu(output)

        output_size = output.shape
        output = output.view(output_size[0], -1)
        output = self.action_linear2(output)

        feature2 = self.autoencoder.decoder[2](feature_map1)
        feature2_size = feature2.shape
        feature2 = output.view(feature2_size[0], -1)
        feature2 = self.relu(feature2)

        fusion_feature2 = torch.cat((output, feature2), dim=1)
        output = self.fusion_layer2(fusion_feature2)
        output = self.relu(output)
        result = output.reshape(output_size)

        return result

class Action_batch1_v4(nn.Module):
    def __init__(self, pre_traind='auto_200.pt'):
        super().__init__()
        self.autoencoder = Autoencoder_batch1().to('cuda')
        self.autoencoder.load_state_dict(torch.load(pre_traind))
        self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
        self.action_decoder1 = nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0)
        self.action_decoder2 = nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0)
        self.action_linear1 = nn.Linear(676,676)
        self.action_linear2 = nn.Linear(900,900)
        self.feature_linear1 = nn.Linear(676, 676)
        self.feature_linear2 = nn.Linear(900, 900)
        self.fusion_layer1 = nn.Linear(2*676, 676)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()


        for param in self.autoencoder.parameters():
            param.requires_grad = False


    def forward(self, x):
        feature_map_origin = self.autoencoder.encoder(x)
        # new_feature_map1 = feature_map1# + self.action_vector
        output = self.action_decoder1(feature_map_origin)
        output = self.relu(output)

        output_size = output.shape
        output = output.view(output_size[0], -1)
        output = self.action_linear1(output)

        feature_map1 = self.autoencoder.decoder[0](feature_map_origin)
        feature1_size = feature_map1.shape
        feature1 = output.view(feature1_size[0], -1)
        feature1 = self.relu(feature1)

        fusion_feature1 = torch.cat((output,feature1),dim=1)
        output = self.fusion_layer1(fusion_feature1)
        output = self.relu(output)
        output = output.reshape(output_size)

        # ============================ layer1 ============================
        output = self.action_decoder2(output)
        output = self.relu(output)

        output_size = output.shape
        output = output.view(output_size[0], -1)
        output = self.action_linear2(output)
        result = output.reshape(output_size)

        return result




class Autoencoder2(nn.Module):
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
        nn.ConvTranspose2d(20, 10, kernel_size = 5, stride = 1, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(10, 1, kernel_size = 5, stride = 1, padding=0),
        nn.ReLU(),
        )

    self.action_vector = nn.Parameter(torch.ones((20,22,22))*0.1)
    self.proj = nn.Linear(1,11)

  def forward(self, x):
    feature_map = self.encoder(x)
    decoder_input = feature_map + self.action_vector
    output = self.decoder(decoder_input)

    return output