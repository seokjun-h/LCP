import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import math
from torch.utils.data import TensorDataset, DataLoader

# 시드 고정
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
set_seed(42)

data = pd.read_csv('/home/aibig25/hong_sj/trb/num.csv')
data = data.fillna(0)

unique_ids = data['sequence_ID'].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=41, random_state=42)
train_data = data[data['sequence_ID'].isin(train_ids)]
test_data = data[data['sequence_ID'].isin(test_ids)]

independent_vars = data.columns.difference(['center_x', 'center_y','center_x_ma','center_y_ma', 'ID', 'frame'])
dependent_vars = ['center_y_ma']

scaler = MinMaxScaler()

train_data[independent_vars] = scaler.fit_transform(train_data[independent_vars])
test_data[independent_vars] = scaler.transform(test_data[independent_vars])

X_train = train_data[independent_vars]
y_train = train_data[dependent_vars]

X_test = test_data[independent_vars]
y_test = test_data[dependent_vars]

# 입력 및 예측 시퀀스 길이 정의
input_sequence_length = 120
output_sequence_length = 90

def create_sequences(data, input_sequence_length, output_sequence_length):
    X = []
    y = []

    for i in range(len(data) - input_sequence_length - output_sequence_length + 1):
        X.append(data.iloc[i:(i + input_sequence_length)][independent_vars].values)
        y.append(data.iloc[(i + input_sequence_length):(i + input_sequence_length + output_sequence_length)][dependent_vars].values)
    
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, input_sequence_length, output_sequence_length)
X_test, y_test = create_sequences(test_data, input_sequence_length, output_sequence_length)

# 데이터셋을 텐서로 변환
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_dim):
        super(TrajectoryTransformer, self).__init__()
        self.model_dim = model_dim
        
        self.encoder = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.tgt_linear = nn.Linear(1, model_dim)  # tgt 차원 변환용 Linear 계층
        
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=model_dim * 4,
            dropout=0.1
        )
        
        self.decoder = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        src = self.encoder(src)
        src = src * math.sqrt(self.model_dim)
        src = self.pos_encoder(src.permute(1, 0, 2))

        tgt = tgt.squeeze(-1)  # 마지막 차원 제거
        original_shape = tgt.shape
        tgt = tgt.reshape(-1, 1)  # view 대신 reshape 사용 (batch_size * seq_length, 1)
        tgt = self.tgt_linear(tgt)  # 차원 조정
        tgt = tgt.view(original_shape[0], original_shape[1], -1)  # 원래 차원으로 복원
        tgt = tgt * math.sqrt(self.model_dim)
        tgt = self.pos_encoder(tgt.permute(1, 0, 2))

        output = self.transformer(src, tgt)
        output = self.decoder(output.permute(1, 0, 2))

        return output


# 위치 인코딩 추가
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    


# 모델 초기화
input_dim = len(independent_vars)
output_dim = len(dependent_vars)
model_dim = 512
num_heads = 4
num_encoder_layers = 3
num_decoder_layers = 3

model = TrajectoryTransformer(input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_dim)

# 옵티마이저와 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


device = torch.device("cuda")
model.to(device)

def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()  # 모델을 훈련 모드로 설정
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()

            # 모델 출력
            output = model(src, tgt)

            # 손실 계산을 위한 타겟 데이터 조정
            min_length = min(output.size(1), tgt.size(1) - 1)
            adjusted_tgt = tgt[:, 1:min_length+1, :]  # output 길이에 맞게 tgt 조정

            # 손실 계산
            output = output[:, :min_length, :]  # output도 동일한 길이로 조정
            loss = criterion(output, adjusted_tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}')


def evaluate_model(model, test_loader):
    model.eval()  # 모델을 평가 모드로 설정
    total_rmse = 0
    total_mape = 0
    total_count = 0
    
    with torch.no_grad():  # 기울기 계산을 중지하여 메모리 사용량과 계산 속도를 개선
        for src, tgt in test_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            output = model(src, tgt[:, :-1, :])  # 마지막 타임 스텝을 제외하고 입력
            
            # 실제 값과 예측 값을 정렬
            tgt_actual = tgt[:, 1:, :]  # 첫 타임 스텝을 제외한 실제 값
            min_length = min(output.size(1), tgt_actual.size(1))
            output = output[:, :min_length, :]
            tgt_actual = tgt_actual[:, :min_length, :]
            
            # RMSE 계산
            rmse = torch.sqrt(torch.mean((output - tgt_actual) ** 2))
            total_rmse += rmse * output.size(0)  # 배치별 가중치를 더하기
            
            # MAPE 계산
            mape = torch.mean(torch.abs((tgt_actual - output) / tgt_actual)) * 100
            total_mape += mape * output.size(0)  # 배치별 가중치를 더하기
            
            total_count += output.size(0)
    
    average_rmse = total_rmse / total_count
    average_mape = total_mape / total_count
    print(f'Final Test RMSE: {average_rmse:.4f}')
    print(f'Final Test MAPE: {average_mape:.4f}')

train_model(model, train_loader, optimizer, criterion, epochs=100)
evaluate_model(model, test_loader)