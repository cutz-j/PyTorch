import glob
import pathlib
import re
import torch
from torch import nn, optim
from torch.utils.data import (Dataset, 
                              DataLoader,
                              TensorDataset)
import tqdm
from statistics import mean

remove_marks_regex = re.compile("[,\.\(\)\[\]\*:;]|<.*?>")
shift_marks_regex = re.compile("([?!])")

def text2ids(text, vocab_dict):
    # !? 이외의 기호 삭제
    text = remove_marks_regex.sub("", text)
    # !?와 단어 사이에 공백 삽입
    text = shift_marks_regex.sub(r" \1 ", text)
    tokens = text.split()
    return [vocab_dict.get(token, 0) for token in tokens]

def list2tensor(token_idxes, max_len=100, padding=True):
    if len(token_idxes) > max_len:
        token_idxes = token_idxes[:max_len]
    n_tokens = len(token_idxes)
    if padding:
        token_idxes = token_idxes \
            + [0] * (max_len - len(token_idxes))
    return torch.tensor(token_idxes, dtype=torch.int64), n_tokens

## Dataset
class IMDBDataset(Dataset):
    def __init__(self, dir_path, train=True,
                 max_len=100, padding=True):
        self.max_len = max_len
        self.padding = padding
        
        path = pathlib.Path(dir_path)
        vocab_path = path.joinpath("imdb.vocab")
        
        # 용어집 파일을 읽어서 행 단위로 분할
        self.vocab_array = vocab_path.open() \
                            .read().strip().splitlines()
        # 단어가 키이고 값이 ID인 dict 만들기
        self.vocab_dict = dict((w, i+1) \
            for (i, w) in enumerate(self.vocab_array))
        
        if train:
            target_path = path.joinpath("train")
        else:
            target_path = path.joinpath("test")
        pos_files = sorted(glob.glob(
            str(target_path.joinpath("pos/*.txt"))))
        neg_files = sorted(glob.glob(
            str(target_path.joinpath("neg/*.txt"))))
        # pos는 1, neg는 0인 label을 붙여서
        # (file_path, label)의 튜플 리스트 작성
        self.labeled_files = \
            list(zip([0]*len(neg_files), neg_files )) + \
            list(zip([1]*len(pos_files), pos_files))
    
    @property
    def vocab_size(self):
        return len(self.vocab_array)
    
    def __len__(self):
        return len(self.labeled_files)
    
    def __getitem__(self, idx):
        label, f = self.labeled_files[idx]
        # 파일의 텍스트 데이터를 읽어서 소문자로 변환
        data = open(f).read().lower()
        # 텍스트 데이터를 ID 리스트로 변환
        data = text2ids(data, self.vocab_dict)
        # ID 리스트를 Tensor로 변환
        data, n_tokens = list2tensor(data, self.max_len, self.padding)
        return data, label, n_tokens

        
train_data = IMDBDataset("d:/dataset/aclImdb/")
test_data = IMDBDataset("d:/dataset/aclImdb/", train=False)
train_loader = DataLoader(train_data, batch_size=32,
                          shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=32,
                        shuffle=False, num_workers=0)

## NN building
class SequenceTaggingNet(nn.Module):
    def __init__(self, num_embeddings,
                 embedding_dim=50, 
                 hidden_size=50,
                 num_layers=1,
                 dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                            padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)


        
        
    def forward(self, x, h0=None, l=None):
        # ID를 Embedding으로 다차원 벡터로 변환
        # x는 (batch_size, step_size) 
        # -> (batch_size, step_size, embedding_dim)
        x = self.emb(x)
        # 초기 상태 h0와 함께 RNN에 x를 전달
        # x는(batch_size, step_size, embedding_dim)
        # -> (batch_size, step_size, hidden_dim)
        x, h = self.lstm(x, h0)
        # 마지막 단계만 추출
        # xは(batch_size, step_size, hidden_dim)
        # -> (batch_size, 1)
        if l is not None:
            # 입력의 원래 길이가 있으면 그것을 이용
            x = x[list(range(len(x))), l-1, :]
        else:
            # 없으면 단순히 마지막 것을 이용
            x = x[:, -1, :]
        # 추출한 마지막 단계를 선형 계층에 넣는다
        x = self.linear(x)
        # 불필요한 차원을 삭제
        # (batch_size, 1) -> (batch_size, )
        x = x.squeeze()
        return x
    
def eval_net(net, data_loader, device="cpu"):
    net.eval()
    ys = []
    ypreds = []
    for x, y, l in data_loader:
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)
        with torch.no_grad():
            y_pred = net(x, l=l)
            y_pred = (y_pred > 0).long()
            ys.append(y)
            ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()

# num_embeddings에는 0을 포함해서 train_data.vocab_size+1를 넣는다
net = SequenceTaggingNet(train_data.vocab_size+1, 
num_layers=2)
net.to("cuda:0")
opt = optim.Adam(net.parameters())
loss_f = nn.BCEWithLogitsLoss()

for epoch in range(10):
    losses = []
    net.train()
    for x, y, l in tqdm.tqdm(train_loader):
        x = x.to("cuda:0")
        y = y.to("cuda:0")
        l = l.to("cuda:0")
        y_pred = net(x, l=l)
        loss = loss_f(y_pred, y.float())
        net.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    train_acc = eval_net(net, train_loader, "cuda:0")
    val_acc = eval_net(net, test_loader, "cuda:0")
    print(epoch, mean(losses), train_acc, val_acc)








            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        