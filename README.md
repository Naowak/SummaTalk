# SummaTalk
Hack1Robot Edition 2023

## Installation

```
pip install -r requirements.txt
```

Ce paquet utilise `llama_cpp_python`, vous pouvez trouver son dépôt et le processus d'installation [ici](https://github.com/abetlen/llama-cpp-python).

Il utilise également `langchain`. Vous pouvez trouver un tutoriel sur comment charger un modèle quantifié avec langchain et llama_cpp_python [ici](https://python.langchain.com/docs/integrations/llms/llamacpp).

N'oubliez pas de télécharger les poids du modèle et de les placer dans le répertoire `./models/` (à la racine du projet), ainsi que les .mp3 et de les placer dans le répertoire `./data/`.

## Ressources

### Données

Créez un dossier data/ dans la racine du projet :

```
mkdir data
```

Vous pouvez ensuite téléchager les données ici : [Lien Google Drive](https://drive.google.com/drive/folders/1e_BTqp4fPxMOQ98GxGBYxkAN7inUWcKw?usp=sharing).

### Modèles

Créez un dossier models/ dans la racine du projet :

```
mkdir models
```

Si vous souhaitez utiliser Llama.cpp, vous devez télécharger un modèle sur Huggingface, je vous conseille de télécharger la version Q_4_K_M : [Mistral-7B-Instruct-v0.2](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main).

Sinon, vous pouvez utiliser Plafrim ;)