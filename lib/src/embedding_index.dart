import 'dart:convert';
import 'dart:typed_data';

import 'utils.dart';

/// A single hit from [EmbeddingIndex.search]: the stored entry's id, its
/// similarity to the query, and the payload stored alongside it (if any).
final class SearchResult {
  /// Creates a search hit.
  const SearchResult(this.id, this.score, {this.payload});

  /// The id the vector was stored under.
  final String id;

  /// Cosine similarity to the query, in `[-1.0, 1.0]`.
  final double score;

  /// The document stored with the vector via `add(..., payload: ...)`, or
  /// `null` when the entry was added without one.
  final String? payload;

  @override
  String toString() => 'SearchResult($id, ${score.toStringAsFixed(4)})';
}

/// One stored entry: the vector (a `Float32List`, or an `Int8List` when the
/// index is quantized) and its optional payload.
typedef _Entry = ({Object vector, String? payload});

/// An in-memory index of embeddings for local semantic search / retrieval.
///
/// Store vectors by id with [add], then query the nearest ones with [search].
/// Attach an optional [String] payload to each entry — e.g. the source passage
/// behind the vector — so [search] hands back the document directly instead of
/// just its id, with no parallel id→text map to keep in sync.
///
/// Set `quantized: true` to keep vectors int8-quantized (~4x less memory, at a
/// small accuracy cost). Persist the whole index (payloads included) with
/// [toBytes] and restore it with [EmbeddingIndex.fromBytes].
///
/// This is a pure data structure — it never touches the native model, so it is
/// fully testable with hand-made vectors.
class EmbeddingIndex {
  /// Creates an empty index. When [quantized], stored vectors are int8.
  EmbeddingIndex({bool quantized = false}) : _quantized = quantized;

  final bool _quantized;

  // id -> stored entry (vector + optional payload). Insertion order is
  // preserved, which keeps [toBytes] output stable.
  final _store = <String, _Entry>{};
  int? _dimension;

  /// Number of stored vectors.
  int get length => _store.length;

  /// Whether the index is empty.
  bool get isEmpty => _store.isEmpty;

  /// Whether vectors are stored int8-quantized.
  bool get isQuantized => _quantized;

  /// Dimension of the stored vectors, or `null` while the index is empty.
  int? get dimension => _dimension;

  /// The ids currently stored, in insertion order.
  Iterable<String> get ids => _store.keys;

  /// Whether [id] is present.
  bool contains(String id) => _store.containsKey(id);

  /// Stores [vector] under [id], replacing any existing entry for that id.
  ///
  /// Optionally attach a [payload] (e.g. the source text) that [search] will
  /// return with the hit. Throws [ArgumentError] if [vector]'s length differs
  /// from the vectors already in the index.
  void add(String id, Float32List vector, {String? payload}) {
    _checkDimension(vector.length);
    _dimension = vector.length;
    _store[id] = (
      vector: _quantized
          ? Model2VecUtils.quantizeToInt8(vector)
          : Float32List.fromList(vector),
      payload: payload,
    );
  }

  /// Stores every entry in [entries]. See [add].
  void addAll(Map<String, Float32List> entries) {
    entries.forEach(add);
  }

  /// Removes the entry for [id]. Returns `true` if it was present.
  bool remove(String id) {
    final removed = _store.remove(id) != null;
    if (_store.isEmpty) {
      _dimension = null;
    }
    return removed;
  }

  /// Removes all entries.
  void clear() {
    _store.clear();
    _dimension = null;
  }

  /// Returns the [topK] entries most similar to [query], most similar first.
  ///
  /// Throws [ArgumentError] if [query]'s length differs from the index's
  /// [dimension]. Returns an empty list for an empty index.
  List<SearchResult> search(Float32List query, {int topK = 5}) {
    final scored = _scoreAll(query)..sort((a, b) => b.score.compareTo(a.score));
    if (topK <= 0) {
      return [];
    }
    if (topK >= scored.length) {
      return scored;
    }
    return scored.sublist(0, topK);
  }

  /// Returns every entry whose similarity to [query] is `>= threshold`, most
  /// similar first.
  List<SearchResult> searchWithThreshold(
    Float32List query, {
    required double threshold,
  }) {
    final scored = _scoreAll(query).where((r) => r.score >= threshold).toList()
      ..sort((a, b) => b.score.compareTo(a.score));
    return scored;
  }

  List<SearchResult> _scoreAll(Float32List query) {
    if (_store.isEmpty) {
      return [];
    }
    if (query.length != _dimension) {
      throw ArgumentError(
        'query length ${query.length} != index dimension $_dimension',
      );
    }
    final results = <SearchResult>[];
    for (final entry in _store.entries) {
      final vector = _asFloat32(entry.value.vector);
      results.add(
        SearchResult(
          entry.key,
          Model2VecUtils.cosineSimilarity(query, vector),
          payload: entry.value.payload,
        ),
      );
    }
    return results;
  }

  Float32List _asFloat32(Object stored) => _quantized
      ? Model2VecUtils.dequantizeInt8(stored as Int8List)
      : stored as Float32List;

  void _checkDimension(int length) {
    if (_dimension != null && length != _dimension) {
      throw ArgumentError(
        'vector length $length != index dimension $_dimension',
      );
    }
  }

  // --- persistence ---------------------------------------------------------

  static const _magic = 'M2VI';
  static const _version = 2;

  // Per-entry payload length sentinel meaning "no payload" (distinct from a
  // zero-length, i.e. empty-string, payload).
  static const _noPayload = 0xFFFFFFFF;

  /// Serializes the whole index — vectors and payloads — to a compact binary
  /// blob. Restore it with [EmbeddingIndex.fromBytes].
  Uint8List toBytes() {
    final dim = _dimension ?? 0;
    final elemBytes = _quantized ? 1 : 4;

    final idByteList = _store.keys.map(utf8.encode).toList(growable: false);
    // Null payload is encoded via the _noPayload sentinel (no trailing bytes).
    final payloadByteList = _store.values
        .map((e) => e.payload == null ? null : utf8.encode(e.payload!))
        .toList(growable: false);

    var size = 4 + 1 + 1 + 4 + 4; // magic + version + flags + dim + count
    for (var i = 0; i < idByteList.length; i++) {
      // u32 idLen + id + vector + u32 payloadLen + payload
      size += 4 + idByteList[i].length + dim * elemBytes;
      size += 4 + (payloadByteList[i]?.length ?? 0);
    }

    final bytes = Uint8List(size);
    final data = ByteData.sublistView(bytes);
    var o = 0;

    bytes.setRange(0, 4, ascii.encode(_magic));
    o = 4;
    data.setUint8(o, _version);
    o += 1;
    data.setUint8(o, _quantized ? 1 : 0);
    o += 1;
    data.setUint32(o, dim, Endian.little);
    o += 4;
    data.setUint32(o, _store.length, Endian.little);
    o += 4;

    var e = 0;
    for (final stored in _store.values) {
      final idBytes = idByteList[e];
      data.setUint32(o, idBytes.length, Endian.little);
      o += 4;
      bytes.setRange(o, o + idBytes.length, idBytes);
      o += idBytes.length;
      if (_quantized) {
        final q = stored.vector as Int8List;
        for (var i = 0; i < dim; i++) {
          data.setInt8(o, q[i]);
          o += 1;
        }
      } else {
        final f = stored.vector as Float32List;
        for (var i = 0; i < dim; i++) {
          data.setFloat32(o, f[i], Endian.little);
          o += 4;
        }
      }
      final payloadBytes = payloadByteList[e++];
      if (payloadBytes == null) {
        data.setUint32(o, _noPayload, Endian.little);
        o += 4;
      } else {
        data.setUint32(o, payloadBytes.length, Endian.little);
        o += 4;
        bytes.setRange(o, o + payloadBytes.length, payloadBytes);
        o += payloadBytes.length;
      }
    }
    return bytes;
  }

  /// Reconstructs an index from a blob produced by [toBytes].
  ///
  /// Throws [ArgumentError] if the blob is not a recognized index.
  static EmbeddingIndex fromBytes(Uint8List bytes) {
    if (bytes.length < 14 || !_hasMagic(bytes)) {
      throw ArgumentError('not a valid EmbeddingIndex blob');
    }
    try {
      return _decode(bytes);
    } on FormatException catch (e) {
      throw ArgumentError('corrupt EmbeddingIndex blob: $e');
      // A short/corrupt buffer surfaces as a RangeError from the typed reads;
      // convert it to the documented ArgumentError for callers.
      // ignore: avoid_catching_errors
    } on RangeError catch (e) {
      throw ArgumentError('corrupt EmbeddingIndex blob: $e');
    }
  }

  static bool _hasMagic(Uint8List bytes) {
    final magic = ascii.encode(_magic);
    for (var i = 0; i < magic.length; i++) {
      if (bytes[i] != magic[i]) {
        return false;
      }
    }
    return true;
  }

  static EmbeddingIndex _decode(Uint8List bytes) {
    final data = ByteData.sublistView(bytes);
    var o = 4; // past the magic

    final version = data.getUint8(o);
    o += 1;
    // v1 has no payload section; v2 adds a payload per entry. Both still read.
    if (version != 1 && version != 2) {
      throw ArgumentError('unsupported EmbeddingIndex version $version');
    }
    final hasPayloads = version >= 2;
    final quantized = data.getUint8(o) == 1;
    o += 1;
    final dim = data.getUint32(o, Endian.little);
    o += 4;
    final count = data.getUint32(o, Endian.little);
    o += 4;

    final index = EmbeddingIndex(quantized: quantized);
    for (var e = 0; e < count; e++) {
      final idLen = data.getUint32(o, Endian.little);
      o += 4;
      final id = utf8.decode(bytes.sublist(o, o + idLen));
      o += idLen;
      final Object vector;
      if (quantized) {
        final q = Int8List(dim);
        for (var i = 0; i < dim; i++) {
          q[i] = data.getInt8(o);
          o += 1;
        }
        vector = q;
      } else {
        final f = Float32List(dim);
        for (var i = 0; i < dim; i++) {
          f[i] = data.getFloat32(o, Endian.little);
          o += 4;
        }
        vector = f;
      }
      String? payload;
      if (hasPayloads) {
        final payloadLen = data.getUint32(o, Endian.little);
        o += 4;
        if (payloadLen != _noPayload) {
          payload = utf8.decode(bytes.sublist(o, o + payloadLen));
          o += payloadLen;
        }
      }
      index._store[id] = (vector: vector, payload: payload);
    }
    index._dimension = count > 0 ? dim : null;
    return index;
  }
}
