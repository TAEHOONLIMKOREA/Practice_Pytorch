import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm
from AutoEncoder_BasicRGB import encoder, decoder

def main():
    # 이미지 저장 폴더 생성 함수
    def create_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 이미지 저장 경로 설정
    create_dir('./AE_img/Encoder')
    create_dir('./AE_img/Decoder')
    create_dir('./AE_img/Origin')

    # 이미지 변환 설정
    img_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 이미지 크기 조정
        transforms.ToTensor(),        # 이미지를 Tensor로 변환
    ])

    # Hyper Parameter 설정
    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3

    # 로컬 이미지 폴더 경로 설정
    image_folder_path = 'data/M160'

    # 현재 작업 경로 가져오기
    current_path = os.getcwd()
    
    # 다른 경로와 결합
    new_path = os.path.join(current_path, image_folder_path)
    
    # ImageFolder를 사용하여 데이터셋 로드
    dataset = ImageFolder(root=image_folder_path, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 설정
    encoder_model = encoder(4).to(device).train()
    decoder_model = decoder(4).to(device).train()

    # Optimizer 설정
    criterion = nn.MSELoss()
    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    decoder_optimizer = torch.optim.Adam(decoder_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 학습 루프
    for epoch in range(num_epochs):
        for data in tqdm(dataloader, desc="Processing:"):
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
                output_img = output.view(output.size(0), 3, 64, 64)
                save_image(output_img.cpu(), f'./AE_img/Decoder/output_image_{epoch}.png')
                
                # Original image 저장
                origin_img = img.view(img.size(0), 3, 64, 64)
                save_image(origin_img.cpu(), f'./AE_img/Origin/origin_image_{epoch}.png')
                
                # Latent space 저장
                latent_img = latent_z.view(latent_z.size(0), 1, 1, -1)
                save_image(latent_img.cpu(), f'./AE_img/Encoder/encoder_image_{epoch}.png')


    # 모델 저장
    torch.save(encoder_model.state_dict(), './encoder.pth')
    torch.save(decoder_model.state_dict(), './decoder.pth')

if __name__ == '__main__':
    main()