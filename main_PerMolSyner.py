import random
import torch.nn.functional as F
import torch.nn as nn
from model import PerMolSyner
from utils_test import *
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def IFA(A, B, temperature=0.1):
    batch_size_times_20 = A.shape[0]
    feature_dim = A.shape[1]
    if batch_size_times_20 != B.shape[0]:
        raise ValueError("A and B must have the same number of rows (batch_size * 20).")
    if feature_dim != B.shape[1]:
        raise ValueError("A and B must have the same feature dimension.")
    A_norm = F.normalize(A, p=2, dim=1)
    B_norm = F.normalize(B, p=2, dim=1)
    logits = torch.matmul(A_norm, B_norm.transpose(0, 1)) / temperature
    labels = torch.arange(batch_size_times_20, device=A.device)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    logits_t = logits.transpose(0, 1)
    logits_max_t, _ = torch.max(logits_t, dim=1, keepdim=True)
    logits_t = logits_t - logits_max_t.detach()
    combined_logits = torch.cat([logits, logits_t], dim=0)
    combined_labels = torch.cat([labels, labels], dim=0)
    loss = F.cross_entropy(combined_logits, combined_labels)
    return loss
def pretrain_finetune(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    t=0.4
    t1=0.4
    model.train()
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output,decoder,origin,f1,f2,output1 = model(data1, data2)
        loss1 = loss_fn(output, y)
        loss2=loss_fn1(decoder,origin)
        loss3=IFA(f1,f2,0.7)
        loss4=PMP(output,output1)
        loss_pretrain=(1-t1)*((1-t)*loss3+t *loss4)+t1*loss2
        loss=loss1+loss_pretrain
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))


def inference_comb(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output,sasdf,fffff,ssss,sssssss,ssssssss = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2
modeling = PerMolSyner

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 1000
datafile = 'samples'
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')
drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1_smile2vec')
drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2_smile2vec')
lenth = len(drug1_data)
pot = int(lenth/5)
random_num = random.sample(range(0, lenth), lenth)
for i in range(5):
    test_num = random_num[pot*i:pot*(i+1)]
    train_num = random_num[:pot*i] + random_num[pot*(i+1):]
    drug1_data_train = drug1_data[train_num]
    drug1_data_test = drug1_data[test_num]
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TEST_BATCH_SIZE, shuffle=None)
    drug2_data_test = drug2_data[test_num]
    drug2_data_train = drug2_data[train_num]
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    model = modeling(device).to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn1=nn.MSELoss()
    PMP=nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        pretrain_finetune(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        T, S, Y = inference_comb(model, device, drug1_loader_test, drug2_loader_test)
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        recall = recall_score(T, Y)
        AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]
        if best_auc < AUC:
            best_auc = AUC