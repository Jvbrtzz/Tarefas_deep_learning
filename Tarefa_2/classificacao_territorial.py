"""Treina e compara duas CNNs para classificar imagens territoriais."""

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DADOS = Path(__file__).resolve().parent / "data"
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
LOTE = 64
SEMENTE = 42


def carregar_dados(pasta):
    # O arquivo do Kaggle pode ter um nível de pastas repetido.
    def particao(nome):
        caminho = pasta / nome
        return caminho / nome if (caminho / nome).is_dir() else caminho

    if not all(particao(nome).is_dir() for nome in ("seg_train", "seg_test")):
        import kagglehub

        pasta = Path(kagglehub.dataset_download(
            "puneet6060/intel-image-classification", output_dir=str(pasta),
        ))

    transformar = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()])
    treino = datasets.ImageFolder(particao("seg_train"), transform=transformar)
    teste = datasets.ImageFolder(particao("seg_test"), transform=transformar)
    if treino.classes != CLASSES or teste.classes != CLASSES:
        raise ValueError("As pastas de treino e teste devem conter as seis classes do dataset.")
    return treino, teste


def montar_cnn(regularizada):
    camadas = []
    for entrada, saida in [(3, 32), (32, 64)]:
        camadas.append(nn.Conv2d(entrada, saida, kernel_size=3, padding=1))
        if regularizada:
            camadas.append(nn.BatchNorm2d(saida))
        camadas.extend([nn.ReLU(), nn.MaxPool2d(2)])
    camadas.extend([nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten()])
    if regularizada:
        camadas.append(nn.Dropout(0.30))
    camadas.append(nn.Linear(64 * 4 * 4, len(CLASSES)))
    return nn.Sequential(*camadas)


def treinar(modelo, lotes, epocas, dispositivo):
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    erro = nn.CrossEntropyLoss()
    modelo.train()
    for epoca in range(1, epocas + 1):
        perda_total = 0.0
        for imagens, classes in lotes:
            otimizador.zero_grad()
            pontuacoes = modelo(imagens.to(dispositivo))
            perda = erro(pontuacoes, classes.to(dispositivo))
            perda.backward()
            otimizador.step()
            perda_total += perda.item() * len(classes)
        print(f"Época {epoca}/{epocas} | Perda: {perda_total / len(lotes.dataset):.3f}")


@torch.no_grad()
def avaliar(modelo, lotes, dispositivo):
    modelo.eval()
    classes_reais, previsoes = [], []
    for imagens, classes in lotes:
        pontuacoes = modelo(imagens.to(dispositivo))
        previsoes.extend(pontuacoes.argmax(dim=1).cpu().tolist())
        classes_reais.extend(classes.tolist())
    return (
        accuracy_score(classes_reais, previsoes),
        f1_score(classes_reais, previsoes, average="macro", zero_division=0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--dados", type=Path, default=DADOS)
    args = parser.parse_args()
    if args.epocas < 1:
        parser.error("--epocas deve ser maior que zero.")

    treino, teste = carregar_dados(args.dados)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo} | Treino: {len(treino)} | Teste: {len(teste)}")
    lotes_teste = DataLoader(teste, batch_size=LOTE)

    for nome, regularizada in [("CNN simples", False), ("CNN regularizada", True)]:
        print(f"\n{nome}")
        torch.manual_seed(SEMENTE)
        lotes_treino = DataLoader(
            treino, batch_size=LOTE, shuffle=True,
            generator=torch.Generator().manual_seed(SEMENTE),
        )
        modelo = montar_cnn(regularizada).to(dispositivo)
        treinar(modelo, lotes_treino, args.epocas, dispositivo)
        acuracia, f1 = avaliar(modelo, lotes_teste, dispositivo)
        print(f"Teste | Acurácia: {acuracia:.2%} | F1 macro: {f1:.4f}")


if __name__ == "__main__":
    main()
