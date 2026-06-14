import torch
import torch.nn as nn

# --- BLOQUE BASE CONVOLUCIONAL ---
class DoubleConv(nn.Module):
    """
    Bloque estándar de U-Net: (Conv2d -> BatchNorm -> ReLU) x 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# --- ARQUITECTURA PRINCIPAL U-NET ---
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        # ENCODER (Ruta de contracción)
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK (Capa más profunda)
        self.bottleneck = DoubleConv(256, 512)

        # DECODER (Ruta de expansión con Skip Connections)
        self.upConv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Al concatenar el upConv (256) con el skip connection (256), la entrada de DoubleConv es 512
        self.up4 = DoubleConv(512, 256)

        self.upConv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Al concatenar el upConv (128) con el skip connection (128), la entrada de DoubleConv es 256
        self.up3 = DoubleConv(256, 128)

        self.upConv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Al concatenar el upConv (64) con el skip connection (64), la entrada de DoubleConv es 128
        self.up2 = DoubleConv(128, 64)

        self.upConv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # Al concatenar el upConv (32) con el skip connection (32), la entrada de DoubleConv es 64
        self.up1 = DoubleConv(64, 32)

        # CAPA DE SALIDA
        # out_channels = 1 para clasificación binaria
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Bajada (Encoder) ---
        x1 = self.down1(x)
        x2 = self.pool1(x1)

        x3 = self.down2(x2)
        x4 = self.pool2(x3)

        x5 = self.down3(x4)
        x6 = self.pool3(x5)

        x7 = self.down4(x6)
        x8 = self.pool4(x7)

        # --- Cuello de Botella ---
        x9 = self.bottleneck(x8)

        # --- Subida (Decoder) + Skip Connections ---
        # Subimos la resolución y concatenamos con el mapa de características del encoder
        x = self.upConv4(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.up4(x)

        x = self.upConv3(x)
        x = torch.cat([x, x5], dim=1)
        x = self.up3(x)

        x = self.upConv2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)

        x = self.upConv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)

        # Salida final (Probabilidad bruta / Logits)
        return self.out(x)
