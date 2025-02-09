import torch
from typing import Optional
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gmp
import numpy as np
values=[i for i in range(32)]
def generate_batch(len_set):
    counts = list(len_set)
    result = [val for count, val in zip(counts, values) for _ in range(count)]
    a=torch.tensor(result)
    return a
def create_src_lengths_mask(
        batch_size: int, src_lengths: Tensor, max_src_len: Optional[int] = None
):
    if max_src_len is None:
        max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
    return (src_indices < src_lengths).int().detach()


def masked_softmax(scores, src_lengths, src_length_masking=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if src_length_masking:
        bsz, max_src_len = scores.size()
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        src_mask = src_mask.to(device)
        scores = scores.masked_fill(src_mask == 0, -np.inf)
    return F.softmax(scores.float(), dim=-1).type_as(scores)
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout,device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x

class Encoder(nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size ,dropout,device):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.do(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved
        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout,device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout,device)
        self.ea = self_attention(hid_dim, n_heads, dropout,device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg

class Decoder(nn.Module):
    def __init__(self, embed_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout,device):
        super(Decoder, self).__init__()
        self.ft = nn.Linear(embed_dim, hid_dim)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout,device)
             for _ in range(n_layers)])
        self.do = nn.Dropout(dropout)
        self.device = device
    def forward(self, trg, src):
        trg = self.do(self.ft(trg))
        for layer in self.layers:
            trg = layer(trg, src)
        return trg

class TextCNN2(nn.Module):
    def __init__(self, embed_dim, hid_dim):
        super(TextCNN2, self).__init__()
        self.convs_protein = nn.Sequential(
            nn.Conv1d(embed_dim, 512, kernel_size=3, padding=3),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 256, kernel_size=3, padding=3),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, hid_dim, kernel_size=3, padding=3),
        )
    def forward(self, protein):
        protein = protein.permute([0, 2, 1])
        protein_features = self.convs_protein(protein)
        protein_features = protein_features.permute([0, 2, 1])
        return protein_features

class TextCNN(nn.Module):
    def __init__(self, embed_dim, hid_dim, kernels=[3, 5, 7], dropout_rate=0.5):
        super(TextCNN, self).__init__()
        padding1 = (kernels[0] - 1) // 2
        padding2 = (kernels[1] - 1) // 2
        padding3 = (kernels[2] - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv = nn.Sequential(
            nn.Linear(hid_dim*len(kernels), hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )
    def forward(self, protein):
        protein = protein.permute([0, 2, 1])
        features1 = self.conv1(protein)
        features2 = self.conv2(protein)
        features3 = self.conv3(protein)
        features = torch.cat((features1, features2, features3), 1)
        features = features.max(dim=-1)[0]
        return self.conv(features)

class PerMolSyner(torch.nn.Module):
    def __init__(self, device,hid_dim=128,n_layers=3, kernel_size=9,n_heads=8, pf_dim=256,n_output=2,
                num_features_xd=78, num_features_xt=954, output_dim=128, dropout=0.2):

        super(PerMolSyner, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.drug1_conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*4, num_features_xd*2)
        self.drug1_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)
        self.drug2_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug2_conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.drug2_conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 4, num_features_xd*2)
        self.drug2_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.decoder=nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_features_xt)
        )
        self.map1 = nn.Linear(128, 128)
        self.map2 = nn.Linear(128, 128)
        self.fc=nn.Linear(100, output_dim)
        self.fc1 = nn.Linear(4*32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)
        self.enc_prot = Encoder(output_dim, hid_dim, n_layers, kernel_size, dropout, device)
        self.dec_smi = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                               PositionwiseFeedforward, dropout, device)
        self.enc_smi = Encoder(output_dim, hid_dim, n_layers, kernel_size, dropout, device)
        self.dec_prot = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                                PositionwiseFeedforward, dropout, device)
    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell,x1_sm,lenx1 = data1.x, data1.edge_index, data1.batch, data1.cell,data1.smile2vec_feature,data1.lens
        new_batch1=generate_batch(lenx1)
        x2, edge_index2, batch2, x2_sm,lenx2= data2.x, data2.edge_index, data2.batch,data2.smile2vec_feature,data2.lens
        new_batch2 = generate_batch(lenx2)
        #######################   Drug-Drug Interactive Representation Learning #############################
        drug1=x1_sm
        drug2=x2_sm
        drug1_squeeze = drug1.unsqueeze(0)
        drug2_squeeze = drug2.unsqueeze(0)
        drug1_squeeze=self.fc(drug1_squeeze)
        drug2_squeeze = self.fc(drug2_squeeze)
        out_enc_prot = self.enc_prot(drug1_squeeze)
        out_dec_smi = self.dec_smi(drug2_squeeze, out_enc_prot)
        out_enc_smi = self.enc_smi(drug2_squeeze)
        out_dec_prot = self.dec_prot(drug1_squeeze, out_enc_smi)
        smi_direction=out_dec_smi.squeeze(0)
        prot_direction = out_dec_prot.squeeze(0)
        interaction_d_d = gmp(prot_direction, new_batch1)#drug A -drug B direction
        interaction_d_d1 = gmp(smi_direction, new_batch2)#drug B -drug A direction
        ######################  Independent Feature Representation Learning #########################
        ############################ Drug A feature extraction #####################################
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)
        x1 = self.relu(self.drug1_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.drug1_fc_g2(x1)
        x1 = self.dropout(x1)
        ############################ Drug B feature extraction #####################################
        x2 = self.drug1_conv1(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = self.drug1_conv2(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = self.drug1_conv3(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2)
        x2 = self.relu(self.drug1_fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.drug1_fc_g2(x2)
        x2 = self.dropout(x2)
        ############################ cell line feature extraction #####################################
        cell_vector = F.normalize(cell, 2, 1)
        original_cell_vector = cell_vector
        cell_vector = self.reduction(cell_vector)
        decoder_cell_vector=self.decoder(cell_vector)
        ############################ Adapative feature fusion #####################################
        x1_f1 = self.map1(x1)
        x1_f2 = self.map2(x1)
        x2_f1 = self.map1(x2)
        x2_f2 = self.map2(x2)
        interaction_d_d_f1 = self.map1(interaction_d_d)
        cell_vector_f1 = self.map1(cell_vector)
        cell_vector_f2 = self.map2(cell_vector)
        interaction_d_d1_f2 = self.map2(interaction_d_d1)
        xc_f1 = x1_f1+x2_f1+interaction_d_d_f1+cell_vector_f1
        xc_f2 = x1_f2 + x2_f2 + interaction_d_d1_f2 + cell_vector_f2
        # Inference on two directions
        xc = self.fc1(xc_f1)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        xc = self.fc1(xc_f2)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out1 = self.out(xc)
        return out,original_cell_vector,decoder_cell_vector,xc_f1,xc_f2,out1
