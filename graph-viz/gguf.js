// 最小 GGUF 头部解析器（纯前端，无依赖）
// 只解析文件头部：magic / version / kv 元数据 / tensor info，不读取权重数据。
// GGUF 布局：header -> kv 段 -> tensor info 段 -> (对齐填充) -> tensor data
(function (global) {
  'use strict';

  // ggml 张量类型（用于显示）
  var GGML_TYPE = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
    8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K',
    14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS',
    19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS',
    24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16'
  };

  // GGUF 元数据值类型
  var GGUF_TYPE = {
    0: 'UINT8', 1: 'INT8', 2: 'UINT16', 3: 'INT16', 4: 'UINT32', 5: 'INT32',
    6: 'FLOAT32', 7: 'BOOL', 8: 'STRING', 9: 'ARRAY', 10: 'UINT64', 11: 'INT64', 12: 'FLOAT64'
  };

  var ARRAY_SAMPLE = 64; // 数组类型最多保留多少个样本值

  function Reader(buf) {
    this.dv = new DataView(buf);
    this.u8 = new Uint8Array(buf);
    this.pos = 0;
    this.decoder = new TextDecoder('utf-8');
  }
  Reader.prototype.need = function (n) {
    if (this.pos + n > this.dv.byteLength) throw new Error('NEED_MORE');
  };
  Reader.prototype.u8v = function () { this.need(1); var v = this.dv.getUint8(this.pos); this.pos += 1; return v; };
  Reader.prototype.i8v = function () { this.need(1); var v = this.dv.getInt8(this.pos); this.pos += 1; return v; };
  Reader.prototype.u16 = function () { this.need(2); var v = this.dv.getUint16(this.pos, true); this.pos += 2; return v; };
  Reader.prototype.i16 = function () { this.need(2); var v = this.dv.getInt16(this.pos, true); this.pos += 2; return v; };
  Reader.prototype.u32 = function () { this.need(4); var v = this.dv.getUint32(this.pos, true); this.pos += 4; return v; };
  Reader.prototype.i32 = function () { this.need(4); var v = this.dv.getInt32(this.pos, true); this.pos += 4; return v; };
  Reader.prototype.f32 = function () { this.need(4); var v = this.dv.getFloat32(this.pos, true); this.pos += 4; return v; };
  Reader.prototype.u64 = function () { this.need(8); var v = Number(this.dv.getBigUint64(this.pos, true)); this.pos += 8; return v; };
  Reader.prototype.i64 = function () { this.need(8); var v = Number(this.dv.getBigInt64(this.pos, true)); this.pos += 8; return v; };
  Reader.prototype.f64 = function () { this.need(8); var v = this.dv.getFloat64(this.pos, true); this.pos += 8; return v; };
  Reader.prototype.str = function () {
    var len = this.u64();
    this.need(len);
    var bytes = this.u8.subarray(this.pos, this.pos + len);
    this.pos += len;
    return this.decoder.decode(bytes);
  };

  // 读取一个值；type 为 GGUF 值类型
  function readValue(r, type) {
    switch (type) {
      case 0: return r.u8v();
      case 1: return r.i8v();
      case 2: return r.u16();
      case 3: return r.i16();
      case 4: return r.u32();
      case 5: return r.i32();
      case 6: return r.f32();
      case 7: return r.u8v() !== 0;
      case 8: return r.str();
      case 10: return r.u64();
      case 11: return r.i64();
      case 12: return r.f64();
      case 9: {
        var et = r.u32();
        var n = r.u64();
        var keep = Math.min(n, ARRAY_SAMPLE);
        var sample = [];
        for (var i = 0; i < n; i++) {
          if (i < keep) sample.push(readValue(r, et));
          else skipValue(r, et); // 其余元素必须跳过以保持偏移正确
        }
        return { __array: true, elemType: GGUF_TYPE[et] || String(et), count: n, sample: sample };
      }
    }
    throw new Error('未知的 GGUF 值类型: ' + type);
  }

  // 跳过值而不解码（用于大数组的剩余元素）
  function skipValue(r, type) {
    switch (type) {
      case 0: case 1: case 7: r.pos += 1; break;
      case 2: case 3: r.pos += 2; break;
      case 4: case 5: case 6: r.pos += 4; break;
      case 10: case 11: case 12: r.pos += 8; break;
      case 8: { var len = r.u64(); r.need(len); r.pos += len; break; }
      case 9: {
        var et = r.u32();
        var n = r.u64();
        for (var i = 0; i < n; i++) skipValue(r, et);
        break;
      }
      default: throw new Error('未知的 GGUF 值类型: ' + type);
    }
  }

  // 解析入口：buf 为 ArrayBuffer（至少覆盖到 tensor info 段末尾）
  function parseGGUF(buf) {
    var r = new Reader(buf);
    var magic = String.fromCharCode(r.u8v(), r.u8v(), r.u8v(), r.u8v());
    if (magic !== 'GGUF') throw new Error('不是 GGUF 文件（magic = "' + magic + '"）');
    var version = r.u32();
    var nTensors = r.u64();
    var nKv = r.u64();

    var metadata = [];
    for (var i = 0; i < nKv; i++) {
      var key = r.str();
      var type = r.u32();
      metadata.push({ key: key, type: GGUF_TYPE[type] || ('TYPE_' + type), value: readValue(r, type) });
    }

    // metadata（kv）段结束的位置，即 tensor info 段起点
    var kvEnd = r.pos;

    var tensors = [];
    for (var j = 0; j < nTensors; j++) {
      var name = r.str();
      var nDims = r.u32();
      var dims = [];
      for (var d = 0; d < nDims; d++) dims.push(r.u64());
      var ttype = r.u32();
      tensors.push({
        name: name,
        dims: dims,
        type: GGML_TYPE[ttype] || ('TYPE_' + ttype),
        typeId: ttype,
        offset: r.u64()
      });
    }

    var alignment = 32;
    for (var m = 0; m < metadata.length; m++) {
      if (metadata[m].key === 'general.alignment') alignment = metadata[m].value;
    }

    return {
      magic: magic,
      version: version,
      nTensors: nTensors,
      nKv: nKv,
      metadata: metadata,
      tensors: tensors,
      alignment: alignment,
      headerSize: 24,   // magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
      kvEnd: kvEnd,     // kv 段结束偏移 = tensor info 段起点
      headerBytes: r.pos // 头部（含 kv 段与 tensor info 段）消耗的字节数
    };
  }

  global.parseGGUF = parseGGUF;
  global.GGML_TYPE_NAMES = GGML_TYPE;
})(typeof window !== 'undefined' ? window : globalThis);
