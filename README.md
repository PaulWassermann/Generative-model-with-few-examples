# Generative-model-with-few-examples

Ce dépôt Github contient le travail de Léo Bonal, Eliott Brucy, Gino Frazzoli et Paul Wassermann, réalisé dans le cadre de leur projet Digital.e. 
Une grande partie du code provient de Thierry Artières et Stéphane Ayache.

Le dépôt contient trois notebooks Python qui permettent de relancer le code avec les hyper-paramètres que nous avons sélectionnés et du code packagé sous le nom de 
few_shot_auto_encoder.

# Package few_shot_auto_encoder

Pour installer le package sur un intepréteur Python en mode "édition", il faut lancer dans un terminal de commande depuis 
le dossier parent du package few_shot_auto_encoder:

`pip -e install few_shot_auto_encoder`

Le mode "édition" permet de faire des modifications au code source du package (dans le dossier `src`) sans avoir besoin de le réinstaller sur l'interpréteur Python.

(peut être fait sur Google Colab à condition de monter le drive depuis un notebook et d'importer le package Google Drive)

# Exemple de code

Pour créer un auto-encodeur:

```python
import torch
import torch.nn as nn
from few_shot_auto_encoder import AutoEncoder
from few_shot_auto_encoder.datasets import load_mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, nb_classes = load_mnist(batch_size=64, shuffle=True, num_workers=2)

encoder = AutoEncoder(input_size=28,
                      latent_size=10, 
                      kernel_size=7, 
                      n_filters=[32], 
                      kernel_sizes=[7],
                      reconstruction_loss=nn.MSELoss(),
                      spatial_transformer=None).to(device)
```
