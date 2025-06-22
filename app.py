import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import time

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), 1)
        img = self.model(gen_input)
        return img.view(img.size(0), 1, 28, 28)

@st.cache_resource(show_spinner=False)
def train_model():
    st.info("Training the model for the first time ...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    latent_dim = 100
    batch_size = 128
    epochs = 20

    generator = Generator()
    discriminator = nn.Sequential(
        nn.Linear(28 * 28 + 10, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
    adversarial_loss = nn.BCELoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_set = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            progress = (epoch * len(dataloader) + i) / (epochs * len(dataloader))
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(dataloader)}")

            real = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)
            imgs = imgs.view(imgs.size(0), -1).to(device)
            labels = labels.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_input = torch.cat((imgs, nn.functional.one_hot(labels, 10).float()), 1)
            real_loss = adversarial_loss(discriminator(real_input), real)
            noise = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_labels = torch.randint(0, 10, (imgs.size(0),), device=device)
            fake_imgs = generator(noise, gen_labels)
            fake_input = torch.cat((fake_imgs.detach().view(fake_imgs.size(0), -1), nn.functional.one_hot(gen_labels, 10).float()), 1)
            fake_loss = adversarial_loss(discriminator(fake_input), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            validity = discriminator(torch.cat((fake_imgs.view(fake_imgs.size(0), -1), nn.functional.one_hot(gen_labels, 10).float()), 1))
            g_loss = adversarial_loss(validity, real)
            g_loss.backward()
            optimizer_G.step()

            status_text.text(f"Epoch {epoch + 1}/{epochs}, complete | D Loss: {d_loss.item():.4f}, | G Loss: {g_loss.item():.4f}")

    progress_bar.progress(1.0)
    status_text.text("Training complete!")
    time.sleep(1)
    status_text.empty()

    return generator.eval()

def generate_digits(model, digit, num_images=5):
    device = next(model.parameters()).device
    with torch.no_grad():
        z = torch.randn(num_images, 100, device=device)
        labels = torch.full((num_images,), digit, dtype=torch.long, device=device)
        gen_images = model(z, labels)
        imgs = []
        for img in gen_images:
            img = img.squeeze().cpu().numpy()
            img = ((img * 0.5) + 0.5) * 255
            imgs.append(Image.fromarray(img.astype('uint8'), mode='L'))
        return imgs

st.set_page_config(page_title="Handwritten Digit Generator", layout="wide")
st.title("Handwritten Digit Generator")

st.sidebar.header("About")
st.sidebar.info("This app generates handwritten digits using a trained GAN model. Select a digit to generate images.")
col1, col2 = st.columns([1, 3])
digit = col1.selectbox("Select Digit", list(range(10)), index=5)
generate_btn = col1.button("Generate Images")

if generate_btn:
    model = train_model()
    st.subheader(f"Generated Images for Digit {digit}")
    images = generate_digits(model, digit)
    cols = st.columns(5)
    for i, img in enumerate(images):
        with cols[i % 5]:
            st.image(img, caption=f"Image {i + 1}")
    st.success("Images generated successfully!")
else:
    st.warning("Click the button to generate images.")