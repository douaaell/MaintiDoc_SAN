"""Stockage vectoriel Qdrant pour MaintiDoc (étape 1 : industrialiser le stockage).

Deux collections séparées :
  - maintidoc_text   : passages texte et tableaux (BGE-M3, 1024 dimensions)
  - maintidoc_visual : images et schémas       (CLIP ViT-B/32, 512 dimensions)

Chaque point garde ses métadonnées : id de l'élément, page, type, document,
version (sha) et type d'équipement.
"""
from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

from PIL import Image
from qdrant_client import QdrantClient, models

TEXT_COLLECTION = 'maintidoc_text'
VISUAL_COLLECTION = 'maintidoc_visual'
TEXT_DIM = 1024    # BAAI/bge-m3
VISUAL_DIM = 512   # sentence-transformers/clip-ViT-B-32
DEFAULT_EQUIPMENT_TYPE = 'SAN'


def get_client() -> QdrantClient:
    """Serveur Qdrant (Docker) par défaut ; QDRANT_URL=local pour un mode sans Docker."""
    url = os.getenv('QDRANT_URL', 'http://localhost:6333')
    if url == 'local':
        return QdrantClient(path=str(Path(__file__).resolve().parent / 'qdrant_local'))
    return QdrantClient(url=url, timeout=30)


def ensure_collections(client: QdrantClient) -> None:
    for name, dim in [(TEXT_COLLECTION, TEXT_DIM), (VISUAL_COLLECTION, VISUAL_DIM)]:
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )


def _doc_filter(document: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key='document', match=models.MatchValue(value=document))])


def _point_id(sha: str, element_id: str) -> str:
    # Identifiant déterministe : réindexer la même version écrase les mêmes points.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{sha}:{element_id}'))


def indexed_sha(client: QdrantClient, document: str) -> str | None:
    """Version (sha) du document déjà indexé, ou None s'il ne l'est pas."""
    points, _ = client.scroll(TEXT_COLLECTION, scroll_filter=_doc_filter(document),
                              limit=1, with_payload=['sha'])
    return points[0].payload.get('sha') if points else None


def delete_document(client: QdrantClient, document: str) -> None:
    for name in (TEXT_COLLECTION, VISUAL_COLLECTION):
        client.delete(collection_name=name,
                      points_selector=models.FilterSelector(filter=_doc_filter(document)))


def index_document(client, document, text_records, images, text_emb, clip,
                   equipment_type: str = DEFAULT_EQUIPMENT_TYPE) -> dict:
    """Indexe un document. Ne fait rien si cette version est déjà indexée.

    Si le PDF a changé (sha différent), l'ancienne version est supprimée puis réindexée.
    Le texte est écrit en dernier : sa présence signifie que l'indexation est complète.
    """
    if not text_records and not images:
        raise ValueError('Aucun élément à indexer.')
    sha = (text_records or images)[0]['sha']

    current = indexed_sha(client, document)
    if current == sha:
        return {'status': 'skipped',
                'message': f'« {document} » (version {sha}) est déjà indexé dans Qdrant : rien à recalculer.'}

    delete_document(client, document)  # nettoie une ancienne version ou une indexation partielle

    # 1) Index visuel
    if images and clip is not None:
        pil_images = [Image.open(io.BytesIO(x['bytes'])).convert('RGB') for x in images]
        vectors = clip.encode(pil_images, normalize_embeddings=True, show_progress_bar=False)
        client.upsert(collection_name=VISUAL_COLLECTION, points=[
            models.PointStruct(
                id=_point_id(sha, x['id']), vector=v.tolist(),
                payload={'id': x['id'], 'page': x['page'], 'type': 'image', 'document': document,
                         'sha': sha, 'equipment_type': equipment_type,
                         'width': x['width'], 'height': x['height']})
            for x, v in zip(images, vectors)])

    # 2) Index textuel (texte et tableaux)
    if text_records:
        vectors = text_emb.get_text_embedding_batch([r['text'] for r in text_records])
        client.upsert(collection_name=TEXT_COLLECTION, points=[
            models.PointStruct(
                id=_point_id(sha, r['id']), vector=list(v),
                payload={'id': r['id'], 'page': r['page'], 'type': r['type'], 'document': document,
                         'sha': sha, 'equipment_type': equipment_type, 'text': r['text']})
            for r, v in zip(text_records, vectors)])

    verb = 'réindexé (nouvelle version)' if current else 'indexé'
    return {'status': 'reindexed' if current else 'indexed',
            'message': f'« {document} » {verb} dans Qdrant : {len(text_records)} éléments texte, {len(images)} images.'}


def _to_hits(points) -> list[dict]:
    # Même format que les anciennes fonctions text_search / image_search.
    return [{'id': p.payload['id'], 'page': int(p.payload['page']), 'rank': rank,
             'score': float(p.score), 'type': p.payload['type']}
            for rank, p in enumerate(points, 1)]


def search_text(client, text_emb, query: str, top_k: int = 8, document: str | None = None) -> list[dict]:
    result = client.query_points(
        collection_name=TEXT_COLLECTION, query=text_emb.get_query_embedding(query), limit=top_k,
        query_filter=_doc_filter(document) if document else None, with_payload=True)
    return _to_hits(result.points)


def search_visual(client, clip, query: str, top_k: int = 6, document: str | None = None) -> list[dict]:
    vector = clip.encode([query], normalize_embeddings=True)[0].tolist()
    result = client.query_points(
        collection_name=VISUAL_COLLECTION, query=vector, limit=top_k,
        query_filter=_doc_filter(document) if document else None, with_payload=True)
    return _to_hits(result.points)


def stats(client: QdrantClient) -> dict:
    return {name: client.count(collection_name=name, exact=True).count
            for name in (TEXT_COLLECTION, VISUAL_COLLECTION)}