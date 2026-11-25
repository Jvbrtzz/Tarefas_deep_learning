"""Compara duas CNNs para classificar imagens territoriais com TensorFlow."""

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers

DADOS = Path(__file__).resolve().parent / "data"
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
TAMANHO = (150, 150)
LOTE = 64
SEMENTE = 42


def particao(pasta, nome):
    # A extração do Kaggle pode repetir o nome da pasta.
    caminho = pasta / nome
    return caminho / nome if (caminho / nome).is_dir() else caminho


def carregar_imagens(pasta, embaralhar):
    return keras.utils.image_dataset_from_directory(
        str(pasta),
        class_names=CLASSES,
        image_size=TAMANHO,
        batch_size=LOTE,
        shuffle=embaralhar,
        seed=SEMENTE,
    )


def montar_cnn(regularizada):
    camadas = [keras.Input(shape=(*TAMANHO, 3)), layers.Rescaling(1 / 255)]
    for filtros in (32, 64):
        camadas.append(layers.Conv2D(filtros, 3, padding="same"))
        if regularizada:
            camadas.append(layers.BatchNormalization())
        camadas.extend([layers.ReLU(), layers.MaxPooling2D()])
    # Após os dois max poolings, 37 × 37 vira 4 × 4 com médias de 9 × 9.
    camadas.extend([layers.AveragePooling2D(pool_size=9), layers.Flatten()])
    if regularizada:
        camadas.append(layers.Dropout(0.30))
    camadas.append(layers.Dense(len(CLASSES)))

    modelo = keras.Sequential(camadas)
    modelo.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return modelo


def avaliar(modelo, teste):
    reais, previstos = [], []
    for imagens, rotulos in teste:
        reais.extend(rotulos.numpy())
        previstos.extend(modelo(imagens, training=False).numpy().argmax(axis=1))
    acuracia = accuracy_score(reais, previstos)
    f1 = f1_score(reais, previstos, labels=range(len(CLASSES)), average="macro", zero_division=0)
    print(f"Teste | Acurácia: {acuracia:.2%} | F1 macro: {f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--dados", type=Path, default=DADOS)
    args = parser.parse_args()
    if args.epocas < 1:
        parser.error("--epocas deve ser maior que zero.")

    pasta = args.dados
    if not all(particao(pasta, nome).is_dir() for nome in ("seg_train", "seg_test")):
        import kagglehub

        pasta = Path(kagglehub.dataset_download(
            "puneet6060/intel-image-classification", output_dir=str(pasta),
        ))

    teste = carregar_imagens(particao(pasta, "seg_test"), embaralhar=False)
    for nome, regularizada in [("CNN simples", False), ("CNN regularizada", True)]:
        print(f"\n{nome}")
        keras.utils.set_random_seed(SEMENTE)
        treino = carregar_imagens(particao(pasta, "seg_train"), embaralhar=True)
        modelo = montar_cnn(regularizada)
        modelo.fit(treino, epochs=args.epocas, verbose=2)
        avaliar(modelo, teste)


if __name__ == "__main__":
    main()
