import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from AutoEncoder_Basic import encoder, decoder

def main():
    # 이미지 저장 폴더 생성 함수
    def create_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 이미지 저장 경로 설정
    create_dir('./AE_img/Encoder')
    create_dir('./AE_img/Decoder')
    create_dir('./AE_img/Origin')

    def to_img(x):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 1, 28, 28)
        return x

    img_transform = transforms.Compose([transforms.ToTensor()])

    # Hyper Parameter 설정
    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3

    #  맨 처음 한번만 다운로드 하기
    # dataset = MNIST('./data', transform=img_transform, download=True)
    
    # 데이터셋 및 DataLoader 설정
    dataset = MNIST('./data', transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 설정
    encoder_model = encoder(3).to(device).train()
    decoder_model = decoder(3).to(device).train()

    # Optimizer 설정
    criterion = nn.MSELoss()
    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    decoder_optimizer = torch.optim.Adam(decoder_model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 학습 루프
    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device)

            # Forward
            latent_z = encoder_model(img)
            output = decoder_model(latent_z)

            # Backward
            loss = criterion(output, img)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        if epoch % 10 == 0:
            with torch.no_grad():
                # Decoder output 저장
                output_img = output.view(output.size(0), 1, 28, 28)
                save_image(output_img.cpu(), f'./AE_img/Decoder/output_image_{epoch}.png')
                
                # Original image 저장
                origin_img = img.view(img.size(0), 1, 28, 28)
                save_image(origin_img.cpu(), f'./AE_img/Origin/origin_image_{epoch}.png')
                
                # Latent space 저장
                latent_img = latent_z.view(latent_z.size(0), 1, 1, -1)
                save_image(latent_img.cpu(), f'./AE_img/Encoder/encoder_image_{epoch}.png')

    # 모델 저장
    torch.save(encoder_model.state_dict(), './encoder.pth')
    torch.save(decoder_model.state_dict(), './decoder.pth')

if __name__ == '__main__':
    main()