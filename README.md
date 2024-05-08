# Information Retrieval for Maths Exercises

The project aims to explore information retrieval systems for math data. The developed model can be used either as a search engine (looking for "exercises with circles and volumes" for example) or a recommender system (recommending other similar exercises similar). Models are trained and tested using the publicly available MATH dataset (mathematical expressions being represented using Latex syntax).

## Notebooks and code
Below is a short description of each notebook:
1. MATH dataset preprocessing (train/test splits, etc.)
2. Pretrained SBERT model analysis
3. Training SBERT model using Contrastive Loss and gradient accumulation. 
    * `custom_sentence_transformers.py` defines a slightly modified `SentenceTransformers` object that supports gradient accumulation for faster training on GPU. Note that gradient accumulation for embedding models only makes sense for loss that are pairwise computed like the Contrastive Loss ([see issue](https://github.com/UKPLab/sentence-transformers/pull/1822)).
4. Examples of usage of the trained search engine and recommender system
5. Index MATH dataset in Opensearch vector database to take advantage of the fast and optimized KNN search functionality.
    * After installing Docker, execute `docker-compose up` from the `./notebook/` directory. It contains the `docker-compose.yml` file to deploy an opensearch node and dashboard locally.
    * Additional instructions for Windows: to solve this ([this issue](https://stackoverflow.com/questions/42111566/elasticsearch-in-windows-docker-image-vm-max-map-count)) perform the following commands:
        * `wsl -d docker-desktop`
        * `sysctl -w vm.max_map_count=262144`
6. Compare the performance of KNN search using the fine tuned SBERT embeddings to the classical approach using TF-IDF
