# Vignemale 

Ce projet a été réalisé pour le Data Battle 2025 organisé par IA Pau.

Il s'agit d'une plateforme utilisant des agents IA afin d'évaluer des étudiants en droit des brevets.

# Utilisation

Pour la démo, nous utilisons des services IA externes (Groq et OpenRouter). Cela est principalement dû au fait que nous n'avons que 10 minutes de présentation et que nos ordinateurs sont peu puissants.  

La solution peut facilement être adaptée pour fonctionner uniquement avec Ollama, à condition de disposer d'un GPU avec environ 14 Go de VRAM (comme une Nvidia T4 disponible sur Google Colab).  

### Instructions pour lancer la démo :  
1. Renseignez les champs manquants dans le fichier Docker, à savoir :  
   - Clés API Groq  
   - Clés API OpenRouter  
   - Clé API Qdrant + URL de la base de données  

2. Ensuite, créez l'image Docker :


```
docker build -t data-battle .
```

puis vous pouvez la lancer 

```
docker run -t data-battle
```

# Information

Les scripts pour créer la base de données sont dans le répertoire data.











