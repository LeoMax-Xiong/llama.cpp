// Qwen / GPT-2 风格 byte-level BPE 分词器（纯 JS，与 llama.cpp 对齐）
// 预处理正则取自 llama.cpp src/llama-vocab.cpp 的 LLAMA_VOCAB_PRE_TYPE_QWEN2
(function (global) {
  'use strict';

  // GPT-2 标准：字节(0..255) <-> 可见 Unicode 字符
  var BYTE_TO_CHAR = (function () {
    var bs = [], i;
    for (i = 33; i <= 126; i++) bs.push(i);
    for (i = 161; i <= 172; i++) bs.push(i);
    for (i = 174; i <= 255; i++) bs.push(i);
    var cs = bs.slice();
    var n = 0;
    for (var b = 0; b < 256; b++) {
      if (bs.indexOf(b) === -1) { bs.push(b); cs.push(256 + n); n++; }
    }
    var map = {};
    for (i = 0; i < bs.length; i++) map[bs[i]] = String.fromCodePoint(cs[i]);
    return map;
  })();
  var CHAR_TO_BYTE = (function () {
    var m = {};
    for (var b in BYTE_TO_CHAR) m[BYTE_TO_CHAR[b]] = parseInt(b, 10);
    return m;
  })();

  var PRE_RE = /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

  function Tokenizer(tokens, merges) {
    this.tokenToId = new Map();
    for (var i = 0; i < tokens.length; i++) {
      if (!this.tokenToId.has(tokens[i])) this.tokenToId.set(tokens[i], i);
    }
    this.ranks = new Map();
    for (var j = 0; j < merges.length; j++) {
      var ln = merges[j];
      if (ln) this.ranks.set(ln, j);
    }
    this.enc = new TextEncoder();
    this.dec = new TextDecoder('utf-8');
  }

  Tokenizer.prototype._toByteStr = function (bytes) {
    var s = '';
    for (var i = 0; i < bytes.length; i++) s += BYTE_TO_CHAR[bytes[i]];
    return s;
  };

  // 对单个 pre-token（byte-char 串）做 BPE 合并
  Tokenizer.prototype._bpe = function (token) {
    var word = token.split('');
    while (word.length > 1) {
      var bestRank = Infinity, bestIdx = -1;
      for (var i = 0; i < word.length - 1; i++) {
        var r = this.ranks.get(word[i] + ' ' + word[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
      }
      if (bestIdx < 0) break;
      word.splice(bestIdx, 2, word[bestIdx] + word[bestIdx + 1]);
    }
    return word;
  };

  // 分词：返回 [{ id, token, nbytes, text }]
  //   id     - 词表中的 token id
  //   token  - 词表中的 token 原文（byte-level 形式）
  //   nbytes - 该 token 占用的字节数
  //   text   - 该 token 对应的原文片段
  Tokenizer.prototype.encode = function (text) {
    var allBytes = this.enc.encode(text);
    var out = [];
    var bytePos = 0;
    var m;
    PRE_RE.lastIndex = 0;
    while ((m = PRE_RE.exec(text)) !== null) {
      if (m[0] === '') { PRE_RE.lastIndex++; continue; }
      var pieceBytes = this.enc.encode(m[0]);
      var toks = this._bpe(this._toByteStr(pieceBytes));
      for (var i = 0; i < toks.length; i++) {
        var t = toks[i];
        var chars = t.split('');
        var nb = 0;
        for (var k = 0; k < chars.length; k++) {
          if (CHAR_TO_BYTE[chars[k]] !== undefined) nb++;
        }
        var id = this.tokenToId.get(t);
        out.push({
          id: id === undefined ? -1 : id,
          token: t,
          nbytes: nb,
          text: this.dec.decode(allBytes.subarray(bytePos, bytePos + nb))
        });
        bytePos += nb;
      }
    }
    return out;
  };

  global.GGUFTokenizer = Tokenizer;
})(typeof window !== 'undefined' ? window : globalThis);
