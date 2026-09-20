// Qwen3-0.6B 计算流程可视化
// 纯静态演示：用一个自包含的状态机，逐步展示
//   1) 模型加载  2) 构建计算图  3) 推理计算  4) 采样与解码
// 参数取自 Qwen3-0.6B 的 config.json / GGUF metadata，其它数值为示意数据。
(function () {
  'use strict';

  // ---------------------------------------------------------------- 模型参数
  // 默认值为 Qwen3-0.6B；用户加载 GGUF 后由 setModelFromGGUF() 覆盖各字段。
  var MODEL = {
    source: 'default',                 // 'default' | 'loaded'
    name: 'Qwen3-0.6B',
    fileName: 'qwen3-0.6b.gguf',
    fileSize: 1509347584,
    magic: 'GGUF',
    version: 3,
    nTensors: 311,
    nKv: 37,
    headerBytes: null,                 // 加载后填充（kv 段 + tensor info 段结束偏移）
    kvEnd: null,                       // 加载后填充（kv 段结束偏移 = tensor info 段起点）
    alignment: 32,
    meta: null,                        // 加载后：完整 metadata 数组
    tensors: null,                     // 加载后：完整 tensor 数组
    arch: 'qwen3',
    layers: 28,
    hidden: 1024,
    heads: 16,
    kvHeads: 8,
    headDim: 128,
    ffn: 3072,
    vocab: 151936,
    ropeTheta: 1000000,
    ctxLen: 40960,
    eos: 151645,
    fileType: 'F16'
  };

  function isLoaded() { return MODEL.source === 'loaded'; }

  // KV cache 大小：n_ctx（演示假设值）* n_layer * n_kv_heads * head_dim * 2(K,V) * 2字节(f16)
  var KV_DEMO_CTX = 4096;
  function kvMiB() {
    return Math.round(KV_DEMO_CTX * MODEL.layers * MODEL.kvHeads * MODEL.headDim * 2 * 2 / 1024 / 1024);
  }

  // ---------------------------------------------------------------- 工具函数
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function node(label, op, cls) {
    return '<span class="node ' + (cls || '') + '">' + esc(label) +
           (op ? '<span class="op">' + esc(op) + '</span>' : '') + '</span>';
  }
  function arrow() { return '<span class="arrow">-&gt;</span>'; }
  function row(nodes) { return '<div class="rowline">' + nodes.join(arrow()) + '</div>'; }
  function fmtBytes(n) {
    if (n == null) return '-';
    if (n >= 1024 * 1024 * 1024) return (n / 1024 / 1024 / 1024).toFixed(2) + ' GiB';
    if (n >= 1024 * 1024) return (n / 1024 / 1024).toFixed(2) + ' MiB';
    if (n >= 1024) return (n / 1024).toFixed(1) + ' KiB';
    return n + ' B';
  }
  function metaGet(meta, key) {
    if (!meta) return undefined;
    for (var i = 0; i < meta.length; i++) if (meta[i].key === key) return meta[i].value;
    return undefined;
  }
  function fmtMetaValue(v) {
    if (v && v.__array) {
      var shown = Math.min(6, v.count);
      var s = v.sample.slice(0, shown).map(function (x) {
        return typeof x === 'string' ? '"' + x + '"' : String(x);
      }).join(', ');
      return 'ARRAY(' + v.elemType + ') x ' + v.count + ' [' + s + (v.count > shown ? ', ...' : '') + ']';
    }
    if (typeof v === 'string') return '"' + v + '"';
    return String(v);
  }
  function fmtMetaShort(v) {
    if (v && v.__array) return v.elemType + '[' + v.count + ']';
    if (typeof v === 'string') return v;
    return String(v);
  }
  // 按张量名推断用途，用于着色
  function tensorCls(name) {
    if (/token_embd/.test(name)) return 'weight';
    if (/attn_(q|k|v|output)/.test(name) || /attn_qkv/.test(name)) return 'attn';
    if (/ffn_/.test(name)) return 'mlp';
    if (/norm/.test(name)) return 'norm';
    if (/^output\./.test(name)) return 'out';
    return '';
  }
  // GGUF general.file_type -> 名称
  var FILE_TYPE = {
    0: 'ALL_F32', 1: 'MOSTLY_F16', 2: 'MOSTLY_Q4_0', 3: 'MOSTLY_Q4_1', 4: 'MOSTLY_Q5_0',
    5: 'MOSTLY_Q5_1', 6: 'MOSTLY_Q8_0', 7: 'MOSTLY_Q2_K', 8: 'MOSTLY_Q3_K_S', 9: 'MOSTLY_Q3_K_M',
    10: 'MOSTLY_Q3_K_L', 11: 'MOSTLY_Q4_K_S', 12: 'MOSTLY_Q4_K_M', 13: 'MOSTLY_Q5_K_S',
    14: 'MOSTLY_Q5_K_M', 15: 'MOSTLY_Q6_K'
  };

  // 用解析出的 GGUF 信息更新 MODEL（全页面的单一数据源）
  function setModelFromGGUF(info, file) {
    var m = info.metadata;
    var arch = metaGet(m, 'general.architecture') || MODEL.arch;
    MODEL.source = 'loaded';
    MODEL.fileName = file.name;
    MODEL.fileSize = file.size;
    MODEL.magic = info.magic;
    MODEL.version = info.version;
    MODEL.nTensors = info.nTensors;
    MODEL.nKv = info.nKv;
    MODEL.headerBytes = info.headerBytes;
    MODEL.kvEnd = info.kvEnd;
    MODEL.alignment = info.alignment;
    MODEL.meta = m;
    MODEL.tensors = info.tensors;
    MODEL.arch = arch;
    MODEL.layers  = metaGet(m, arch + '.block_count')             || MODEL.layers;
    MODEL.hidden  = metaGet(m, arch + '.embedding_length')        || MODEL.hidden;
    MODEL.heads   = metaGet(m, arch + '.attention.head_count')    || MODEL.heads;
    MODEL.kvHeads = metaGet(m, arch + '.attention.head_count_kv') || MODEL.heads;
    MODEL.headDim = metaGet(m, arch + '.attention.key_length')    || Math.round(MODEL.hidden / MODEL.heads);
    MODEL.ffn     = metaGet(m, arch + '.feed_forward_length')     || MODEL.ffn;
    MODEL.ropeTheta = metaGet(m, arch + '.rope.freq_base')        || MODEL.ropeTheta;
    MODEL.ctxLen  = metaGet(m, arch + '.context_length')          || MODEL.ctxLen;
    var toks = metaGet(m, 'tokenizer.ggml.tokens');
    if (toks && toks.__array) MODEL.vocab = toks.count;
    var eos = metaGet(m, 'tokenizer.ggml.eos_token_id');
    if (eos != null) MODEL.eos = eos;
    var ft = metaGet(m, 'general.file_type');
    if (ft != null) MODEL.fileType = FILE_TYPE[ft] || ('TYPE_' + ft);
    var nm = metaGet(m, 'general.name');
    MODEL.name = nm ? String(nm).trim() : file.name;
  }

  // ---------------------------------------------------------- GGUF 元数据字段解释
  // '[arch]' 匹配任意架构前缀（qwen3. / llama. / qwen2. 等）
  var META_DESC = {
    // general
    'general.architecture': '模型架构名称（如 qwen3、llama）',
    'general.name': '模型完整名称',
    'general.basename': '模型基础名（不含版本/尺寸）',
    'general.size_label': '参数量标注（如 0.6B、8B）',
    'general.version': '模型版本号',
    'general.finetune': '微调目标（如 instruct、chat）',
    'general.type': '文件用途类型（model / lora / vocab）',
    'general.organization': '模型组织/发布者',
    'general.author': '模型作者',
    'general.license': '许可证名称',
    'general.license.name': '许可证名称',
    'general.license.link': '许可证链接',
    'general.url': '模型主页地址',
    'general.repo_url': '模型仓库地址',
    'general.description': '模型描述',
    'general.tags': '标签数组',
    'general.languages': '支持的语言',
    'general.file_type': '权重主要量化类型（如 F16、Q4_K_M）',
    'general.quantization_version': '量化方案版本号',
    'general.alignment': '数据对齐字节数（data 段填充到该倍数）',
    'general.parameter_count': '参数总量',
    'general.sampling.top_k': '推荐采样参数：top-k',
    'general.sampling.top_p': '推荐采样参数：top-p',
    'general.sampling.temp': '推荐采样参数：温度 temperature',
    'general.base_model.count': '基座模型条目数',
    'general.base_model.N.name': '基座模型名称',
    'general.base_model.N.organization': '基座模型所属组织',
    'general.base_model.N.repo_url': '基座模型仓库地址',
    'general.base_model.N.version': '基座模型版本',
    // [arch] 结构参数
    '[arch].context_length': '最大上下文长度（token 数）',
    '[arch].block_count': 'Transformer 层数（n_layer）',
    '[arch].embedding_length': '隐藏层维度（n_embd）',
    '[arch].feed_forward_length': '前馈网络中间维度（n_ff）',
    '[arch].vocab_size': '词表大小',
    '[arch].attention.head_count': '注意力头数（n_head）',
    '[arch].attention.head_count_kv': 'KV 头数（GQA，多个 Q 头共享）',
    '[arch].attention.key_length': '每个 K 头维度（head_dim）',
    '[arch].attention.value_length': '每个 V 头维度',
    '[arch].attention.layer_norm_epsilon': 'LayerNorm 的 epsilon',
    '[arch].attention.layer_norm_rms_epsilon': 'RMSNorm 的 epsilon（数值稳定项）',
    '[arch].attention.clamp_kqv': 'Q/K/V 数值裁剪范围',
    '[arch].attention.alibi_bias_max': 'ALiBi 最大偏置',
    '[arch].attention.max_alibi_bias': 'ALiBi 最大偏置',
    '[arch].attention.sliding_window': '滑动窗口大小',
    '[arch].attention.sliding_window_pattern': '滑窗与全注意力的交替模式',
    '[arch].rope.dimension_count': 'RoPE 旋转维数',
    '[arch].rope.freq_base': 'RoPE 位置编码基础频率',
    '[arch].rope.scaling.type': 'RoPE 缩放类型（none / linear / yarn）',
    '[arch].rope.scaling.factor': 'RoPE 缩放因子（用于扩展上下文）',
    '[arch].rope.scaling.original_context_length': 'RoPE 缩放前的原始上下文长度',
    '[arch].rope.scale_linear': 'RoPE 线性缩放因子（旧键）',
    '[arch].expert_count': 'MoE 专家总数',
    '[arch].expert_used_count': '每个 token 激活的专家数',
    '[arch].expert_shared_count': '共享专家数',
    '[arch].expert_feed_forward_length': '专家前馈中间维度',
    '[arch].expert_ffn_length': '专家前馈中间维度',
    '[arch].use_parallel_residual': '是否使用并行残差',
    '[arch].tensor_data_layout': '权重张量布局约定',
    '[arch].pooling_type': '池化类型（embedding 模型）',
    '[arch].ssm.conv_kernel': 'SSM 卷积核大小（Mamba）',
    '[arch].ssm.inner_size': 'SSM 内部状态维度',
    '[arch].ssm.state_size': 'SSM 循环状态大小',
    '[arch].ssm.time_step_rank': 'SSM 时间步秩',
    // tokenizer
    'tokenizer.ggml.model': '分词器模型类型（llama / gpt2 / rwkv）',
    'tokenizer.ggml.pre': '分词预处理规则名（如 qwen2、llama-bpe）',
    'tokenizer.ggml.tokens': '词表：token id 到文本的映射',
    'tokenizer.ggml.scores': '每个 token 的分数（SentencePiece）',
    'tokenizer.ggml.token_type': '每个 token 的类型（普通/控制/未知等）',
    'tokenizer.ggml.merges': 'BPE 合并规则表',
    'tokenizer.ggml.added_tokens': '训练后追加的 token',
    'tokenizer.ggml.bos_token_id': '起始符（BOS）token id',
    'tokenizer.ggml.eos_token_id': '结束符（EOS）token id',
    'tokenizer.ggml.unknown_token_id': '未知符（UNK）token id',
    'tokenizer.ggml.separator_token_id': '分隔符 token id',
    'tokenizer.ggml.padding_token_id': '填充符（PAD）token id',
    'tokenizer.ggml.add_bos_token': '是否自动添加 BOS',
    'tokenizer.ggml.add_eos_token': '是否自动添加 EOS',
    'tokenizer.ggml.add_space_prefix': '是否给 token 片段添加空格前缀',
    'tokenizer.ggml.remove_extra_whitespaces': '是否移除多余空白',
    'tokenizer.ggml.precompiled_charsmap': '预编译字符映射（SentencePiece）',
    'tokenizer.chat_template': '对话模板（Jinja 字符串）',
    'tokenizer.huggingface.json': '内嵌的 HF tokenizer.json'
  };

  // 按字段名查中文解释；未收录返回 '-'
  function metaDesc(key) {
    if (META_DESC[key] !== undefined) return META_DESC[key];
    var bm = /^general\.base_model\.\d+\.(.+)$/.exec(key);
    if (bm) return META_DESC['general.base_model.N.' + bm[1]] || '-';
    if (/^(general|tokenizer)\./.test(key)) return '-';
    var i = key.indexOf('.');
    if (i > 0) {
      var v = META_DESC['[arch]' + key.slice(i)];
      if (v !== undefined) return v;
    }
    return '-';
  }

  // ---------------------------------------------------------- GGUF 文件结构图
  function alignUp(n, a) { return Math.ceil(n / a) * a; }
  // "GGUF" -> 小端 u32 值 0x46554747
  function magicLE(s) {
    var v = 0;
    for (var i = 0; i < s.length; i++) v += s.charCodeAt(i) * Math.pow(256, i);
    return '0x' + v.toString(16).toUpperCase();
  }
  function gfield(name, type, value, note, color) {
    return '<div class="gguf-field ' + (color || '') + '">' +
           '<div class="fn">' + esc(name) + '</div>' +
           '<div class="ft">' + esc(type) + '</div>' +
           '<div class="fv">' + esc(value) + '</div>' +
           '<div class="gnote">' + (note ? esc(note) : '') + '</div></div>';
  }
  // 按字节顺序绘制 GGUF 布局；数值随 MODEL 联动，未加载时显示占位
  function renderGGUFLayout() {
    var HDR = 24;
    var kvBytes = MODEL.kvEnd != null ? MODEL.kvEnd - HDR : null;
    var tiBytes = (MODEL.kvEnd != null && MODEL.headerBytes != null) ? MODEL.headerBytes - MODEL.kvEnd : null;
    var dataOff = MODEL.headerBytes != null ? alignUp(MODEL.headerBytes, MODEL.alignment) : null;
    var padBytes = dataOff != null ? dataOff - MODEL.headerBytes : null;

    function size(v) { return v == null ? '加载后显示' : fmtBytes(v) + ' = ' + v.toLocaleString() + ' B'; }

    var kvSample = (isLoaded() && MODEL.meta && MODEL.meta.length)
      ? '首项 ' + MODEL.meta[0].key + ' = ' + fmtMetaShort(MODEL.meta[0].value)
      : '每项 = key(string) + value_type(u32) + value(变长)';
    var tiSample = (isLoaded() && MODEL.tensors && MODEL.tensors.length)
      ? '首项 ' + MODEL.tensors[0].name + ' [' + MODEL.tensors[0].dims.join(', ') + '] ' + MODEL.tensors[0].type
      : '每项 = name + n_dimensions + dimensions + type + offset';

    return '<div class="gguf-diagram">' +
      '<div class="gguf-diagram-head">GGUF 文件结构（依据 ggml <code>docs/gguf.md</code> 规范绘制，自上而下为字节顺序）' +
        (isLoaded() ? ' · <span style="color:var(--green)">实测</span>' : ' · <span style="color:var(--yellow)">示例</span>') + '</div>' +

      '<div class="gguf-seg hdr">' +
        '<div class="gguf-seg-head"><span>gguf_header_t（文件头）</span><span class="gguf-seg-size">' + HDR + ' B（固定）</span></div>' +
        '<div class="gguf-fields hdr">' +
          gfield('magic', 'u32 · 4 B', magicLE(MODEL.magic),
            '字节为 47 47 55 46（"GGUF"），按小端 u32 读出即 ' + magicLE(MODEL.magic), 'r-blue') +
          gfield('version', 'u32 · 4 B', MODEL.version, '', 'r-purple') +
          gfield('tensor_count', 'u64 · 8 B', MODEL.nTensors, '', 'r-green') +
          gfield('metadata_kv_count', 'u64 · 8 B', MODEL.nKv, '', 'r-yellow') +
        '</div>' +
      '</div>' +

      '<div class="gguf-seg kv">' +
        '<div class="gguf-seg-head"><span>gguf_metadata_kv_t[metadata_kv_count]（元数据段）</span><span class="gguf-seg-size">' + size(kvBytes) + '</span></div>' +
        '<div class="gguf-fields cols3">' +
          gfield('key', 'gguf_string_t', 'u64 len + bytes') +
          gfield('value_type', 'u32 · 4 B', '0..12') +
          gfield('value', '变长', '按 value_type') +
        '</div>' +
        '<div class="gguf-note">共 ' + MODEL.nKv + ' 项 · ' + esc(kvSample) + '</div>' +
      '</div>' +

      '<div class="gguf-seg ti">' +
        '<div class="gguf-seg-head"><span>tensor_info[tensor_count]（张量描述表）</span><span class="gguf-seg-size">' + size(tiBytes) + '</span></div>' +
        '<div class="gguf-fields cols5">' +
          gfield('name', 'gguf_string_t', 'u64 len + bytes') +
          gfield('n_dimensions', 'u32 · 4 B', '维数') +
          gfield('dimensions[n]', 'u64 × n', '各维大小') +
          gfield('type', 'ggml_type · u32', '如 F16') +
          gfield('offset', 'u64 · 8 B', '相对 data') +
        '</div>' +
        '<div class="gguf-note">共 ' + MODEL.nTensors + ' 项 · ' + esc(tiSample) + '</div>' +
      '</div>' +

      '<div class="gguf-seg pad">' +
        '<div class="gguf-seg-head"><span>padding（对齐填充）</span><span class="gguf-seg-size">' + size(padBytes) + '</span></div>' +
        '<div class="gguf-note">以 0x00 填充，使张量数据起点对齐到 general.alignment = ' + MODEL.alignment + '</div>' +
      '</div>' +

      '<div class="gguf-seg data">' +
        '<div class="gguf-seg-head"><span>tensor_data（权重数据）</span><span class="gguf-seg-size">' +
          (MODEL.fileSize != null && dataOff != null
            ? fmtBytes(MODEL.fileSize - dataOff) + ' = ' + (MODEL.fileSize - dataOff).toLocaleString() + ' B'
            : '加载后显示') +
        '</span></div>' +
        '<div class="gguf-note">' +
          (dataOff != null ? '起始偏移 ' + dataOff.toLocaleString() + ' B；' : '') +
          '各权重按各自的 offset 排列于此，可 mmap 直接映射</div>' +
      '</div>' +

    '</div>';
  }

  // ---------------------------------------------------------------- 日志面板
  var Log = {
    root: null,
    lines: [],
    init: function (root) { this.root = root; },
    setStageLogs: function (lines) {
      this.root.innerHTML = lines.map(function (l) {
        return '<span class="' + l.cls + '">' + esc(l.text) + '</span>';
      }).join('\n');
      this.root.scrollTop = this.root.scrollHeight;
    }
  };

  // ---------------------------------------------------------------- 左侧说明
  function concepts(items) {
    if (typeof items === 'function') items = items();
    return items.map(function (it) {
      return '<div class="concept-card"><b>' + esc(it[0]) + '</b><br>' + it[1] + '</div>';
    }).join('');
  }

  // ================================================================ 阶段 1：模型加载
  var LOAD_STEPS = [
    {
      title: '打开 GGUF 文件',
      desc: 'llama.cpp 使用 GGUF 格式存放模型。加载的第一步是打开文件、校验文件头（magic "GGUF"、版本号），并读出张量数量与元数据条目数。',
      concepts: [
        ['GGUF 文件头', '魔数 <code>GGUF</code>(4字节) + 版本 + 张量数 + 元数据 kv 数'],
        ['张量(Tensor)', '模型的权重矩阵，如 <code>blk.0.attn_q.weight</code>'],
        ['元数据(kv)', '架构、层数、词表大小等超参数，以 key-value 形式存储']
      ],
      logs: function () {
        var out = [{ cls: 'l-load', text: 'llama_model_loader: loaded meta data with ' + MODEL.nKv + ' key-value pairs and ' + MODEL.nTensors + ' tensors' }];
        if (isLoaded()) {
          for (var i = 0; i < Math.min(5, MODEL.meta.length); i++) {
            var m = MODEL.meta[i];
            out.push({ cls: 'l-info', text: 'llama_model_loader: - kv ' + i + ': ' + m.key + ' = ' + fmtMetaShort(m.value) });
          }
        } else {
          out.push({ cls: 'l-time', text: '（当前为示例数据，点击左侧【加载模型】可解析真实 GGUF 文件）' });
        }
        return out;
      },
      render: function (box) {
        var badge = isLoaded()
          ? ' <span style="color:var(--green);font-size:11px">实测</span>'
          : ' <span style="color:var(--yellow);font-size:11px">示例</span>';
        box.innerHTML =
          '<div class="file-card">' +
            '<div class="file-icon">GGUF</div>' +
            '<div><div class="fname">' + esc(MODEL.fileName) + badge + '</div>' +
            '<div class="fmeta">' + fmtBytes(MODEL.fileSize) + '  -  header: magic=' + esc(MODEL.magic) +
              ', version=' + MODEL.version + ', n_tensors=' + MODEL.nTensors + ', n_kv=' + MODEL.nKv + '</div>' +
            (MODEL.headerBytes != null ? '<div class="fmeta">头部（kv 段 + tensor info 段）占用 ' + fmtBytes(MODEL.headerBytes) + '</div>' : '') +
          '</div></div>' +
          '<div style="margin-top:16px" class="rowline">' +
            node('magic', esc(MODEL.magic), 'input') + arrow() +
            node('version', 'u32 = ' + MODEL.version, 'input') + arrow() +
            node('n_tensors', 'u64 = ' + MODEL.nTensors, 'input') + arrow() +
            node('n_kv', 'u64 = ' + MODEL.nKv, 'input') +
          '</div>' +
          '<p class="step-desc" style="margin-top:14px">文件头校验通过，准备读取元数据与张量表。</p>' +
          (isLoaded() ? '' : '<p class="step-desc">点击左侧【加载模型】按钮选择一个 .gguf 文件，即可用真实数据替换以上示例。</p>') +
          renderGGUFLayout();
      }
    },
    {
      title: '解析元数据（超参数）',
      desc: '从文件头之后读出 key-value 元数据。这些超参数决定了模型结构，也决定了后面计算图如何构建。',
      concepts: function () {
        return [
          ['架构', '<code>' + esc(MODEL.arch) + '</code>：仅解码器（decoder-only）Transformer，用 GQA 注意力'],
          ['GQA', MODEL.heads + ' 个 query 头共享 ' + MODEL.kvHeads + ' 个 key/value 头，KV cache 更小'],
          ['head_dim', MODEL.headDim + '：每个注意力头的维度，' + MODEL.heads + ' x ' + MODEL.headDim + ' = ' + (MODEL.heads * MODEL.headDim) + ' 为 Q 投影输出']
        ];
      },
      logs: function () {
        return [
          { cls: 'l-load', text: 'print_info: arch                  = ' + MODEL.arch },
          { cls: 'l-load', text: 'print_info: n_layer               = ' + MODEL.layers },
          { cls: 'l-load', text: 'print_info: n_embd                = ' + MODEL.hidden },
          { cls: 'l-load', text: 'print_info: n_head                = ' + MODEL.heads },
          { cls: 'l-load', text: 'print_info: n_head_kv             = ' + MODEL.kvHeads },
          { cls: 'l-load', text: 'print_info: n_embd_head_k         = ' + MODEL.headDim },
          { cls: 'l-load', text: 'print_info: n_ff                  = ' + MODEL.ffn },
          { cls: 'l-load', text: 'print_info: n_vocab               = ' + MODEL.vocab },
          { cls: 'l-load', text: 'print_info: rope_freq_base        = ' + MODEL.ropeTheta }
        ];
      },
      render: function (box) {
        if (isLoaded()) {
          var rows = MODEL.meta.map(function (m, i) {
            return '<tr><td class="idx">' + (i + 1) + '</td>' +
                   '<td class="key">' + esc(m.key) + '</td>' +
                   '<td class="desc">' + esc(metaDesc(m.key)) + '</td>' +
                   '<td class="typ">' + esc(m.type) + '</td>' +
                   '<td class="val">' + esc(fmtMetaValue(m.value)) + '</td></tr>';
          }).join('');
          box.innerHTML =
            '<div style="font-family:var(--mono);font-size:11.5px;color:var(--text-dim);margin-bottom:8px">来自所选文件的 ' +
              MODEL.nKv + ' 条元数据 <span style="color:var(--green)">实测</span></div>' +
            '<table class="insp-table"><tbody>' + rows + '</tbody></table>';
        } else {
          var rows2 = [
            ['general.architecture', MODEL.arch],
            ['general.name', MODEL.name],
            [MODEL.arch + '.context_length', MODEL.ctxLen],
            [MODEL.arch + '.embedding_length', MODEL.hidden],
            [MODEL.arch + '.block_count', MODEL.layers],
            [MODEL.arch + '.feed_forward_length', MODEL.ffn],
            [MODEL.arch + '.attention.head_count', MODEL.heads],
            [MODEL.arch + '.attention.head_count_kv', MODEL.kvHeads],
            [MODEL.arch + '.attention.key_length', MODEL.headDim],
            [MODEL.arch + '.rope.freq_base', MODEL.ropeTheta],
            [MODEL.arch + '.attention.layer_norm_rms_epsilon', '1e-06'],
            ['tokenizer.ggml.model', 'gpt2 (BPE)'],
            ['tokenizer.ggml.tokens', MODEL.vocab]
          ];
          box.innerHTML = '<table class="insp-table">' + rows2.map(function (r, i) {
            return '<tr><td class="idx">' + (i + 1) + '</td>' +
                   '<td class="key">' + esc(r[0]) + '</td>' +
                   '<td class="desc">' + esc(metaDesc(r[0])) + '</td>' +
                   '<td class="val">' + esc(r[1]) + '</td></tr>';
          }).join('') + '</table>' +
          '<p class="step-desc" style="margin-top:14px">当前为示例数据 <span style="color:var(--yellow)">示例</span>；' +
          '点击左侧【加载模型】后，此处将展示所选文件的完整元数据 <span style="color:var(--green)">实测</span>。</p>';
        }
      }
    },
    {
      title: '创建后端并上传权重',
      desc: '根据 -ngl 参数创建后端（CPU / Metal GPU），把模型的全部张量分配到各后端缓冲，并把权重数据从磁盘读入设备内存。',
      concepts: function () {
        return [
          ['后端(backend)', '执行计算的目标设备：CPU、Metal、CUDA、Vulkan 等'],
          ['-ngl 99', '把全部 ' + MODEL.layers + ' 层放到 GPU；-ngl 0 则全部在 CPU'],
          ['权重张量', 'token_embd、每层的 attn_q/k/v/o、ffn_gate/up/down 等']
        ];
      },
      logs: function () {
        return [
          { cls: 'l-load', text: 'llama_model_load: model size    = ' + fmtBytes(MODEL.fileSize) },
          { cls: 'l-load', text: 'load_tensors: offloading ' + MODEL.layers + ' repeating layers to GPU' },
          { cls: 'l-load', text: 'load_tensors: offloaded ' + (MODEL.layers + 1) + '/' + (MODEL.layers + 1) + ' layers to GPU' },
          { cls: 'l-load', text: 'llama_model_load: model buffer size = ' + fmtBytes(MODEL.fileSize) }
        ];
      },
      render: function (box) {
        var list;
        if (isLoaded()) {
          list = MODEL.tensors.slice(0, 12).map(function (t) {
            return [t.name, '[' + t.dims.join(', ') + '] ' + t.type, tensorCls(t.name)];
          });
          if (MODEL.nTensors > 12) list.push(['... 共 ' + MODEL.nTensors + ' 个张量 ...', '', '']);
        } else {
          list = [
            ['token_embd.weight', '[' + MODEL.vocab + ', ' + MODEL.hidden + ']', 'weight'],
            ['blk.0.attn_norm.weight', '[' + MODEL.hidden + ']', 'norm'],
            ['blk.0.attn_q.weight', '[' + (MODEL.heads * MODEL.headDim) + ', ' + MODEL.hidden + ']', 'attn'],
            ['blk.0.attn_k.weight', '[' + (MODEL.kvHeads * MODEL.headDim) + ', ' + MODEL.hidden + ']', 'attn'],
            ['blk.0.attn_v.weight', '[' + (MODEL.kvHeads * MODEL.headDim) + ', ' + MODEL.hidden + ']', 'attn'],
            ['blk.0.ffn_gate.weight', '[' + MODEL.ffn + ', ' + MODEL.hidden + ']', 'mlp'],
            ['...  (第 1..' + (MODEL.layers - 1) + ' 层同构)  ...', '', ''],
            ['output_norm.weight', '[' + MODEL.hidden + ']', 'norm'],
            ['output.weight', '(tied, 复用 token_embd)', 'out']
          ];
        }
        box.innerHTML =
          '<div class="rowline" style="margin-bottom:14px">' +
            node('CPU', 'backend', 'input') + arrow() +
            node('Metal / GPU', 'backend', 'attn') +
          '</div>' +
          '<div class="layer-stack">' + list.map(function (t) {
            if (!t[1]) return '<div class="layer-fold">' + esc(t[0]) + '</div>';
            return '<div class="rowline">' + node(t[0], '', t[2]) +
                   '<span class="arrow" style="margin-left:auto">' + esc(t[1]) + '</span></div>';
          }).join('') + '</div>' +
          (isLoaded()
            ? '<p class="step-desc" style="margin-top:14px">张量列表取自所选文件 <span style="color:var(--green)">实测</span>（仅展示前 12 个）。</p>'
            : '');
      }
    },
    {
      title: '创建 llama_context（分配 KV cache）',
      desc: '上下文（llama_context）持有推理所需的运行时状态：KV cache、输出缓冲、后端调度器。其中 KV cache 用来缓存历史 token 的 K/V，避免重复计算。',
      concepts: function () {
        return [
          ['KV cache', '缓存每层每步的 Key/Value 张量，是自回归生成的关键'],
          ['GQA 的好处', '只需 ' + MODEL.kvHeads + ' 个 KV 头而非 ' + MODEL.heads + ' 个，KV cache 更小'],
          ['后端调度器', '<code>ggml_backend_sched</code>：把计算图切分到多个后端执行']
        ];
      },
      logs: function () {
        return [
          { cls: 'l-load', text: 'llama_context: n_ctx         = ' + KV_DEMO_CTX },
          { cls: 'l-load', text: 'llama_context: n_batch       = 2048' },
          { cls: 'l-load', text: 'llama_context: flash_attn    = auto' },
          { cls: 'l-load', text: 'llama_kv_cache: buffer size  = ' + kvMiB() + ' MiB' },
          { cls: 'l-load', text: 'sched_reserve: reserving ... graph nodes' }
        ];
      },
      render: function (box) {
        box.innerHTML =
          '<div class="concept-card" style="font-family:var(--mono)">' +
            'KV = n_ctx(' + KV_DEMO_CTX + ') x n_layer(' + MODEL.layers + ') x n_kv_heads(' + MODEL.kvHeads +
            ') x head_dim(' + MODEL.headDim + ') x 2(K,V) x 2B = <b>' + kvMiB() + ' MiB</b>' +
          '</div>' +
          '<div class="mem-bar" style="margin-top:14px">' +
            '<div class="mem-seg w"  style="flex:60">weights ' + fmtBytes(MODEL.fileSize) + '</div>' +
            '<div class="mem-seg kv" style="flex:20">KV cache ' + kvMiB() + ' MiB</div>' +
            '<div class="mem-seg out" style="flex:12">output / compute</div>' +
          '</div>' +
          '<div class="rowline" style="margin-top:16px">' +
            node('llama_model', '权重', 'weight') + arrow() +
            node('llama_context', '运行时状态', 'input') + arrow() +
            node('llama_sampler', '采样器链', 'out') +
          '</div>' +
          '<p class="step-desc" style="margin-top:14px">模型加载完成，可以开始推理。</p>' +
          '<p class="step-desc">KV cache 依模型结构参数估算 <span style="color:var(--cyan)">计算值</span>；n_ctx=' +
            KV_DEMO_CTX + ' 为演示假设，非模型文件携带的信息。</p>';
      }
    }
  ];

  // ================================================================ 阶段 2：构建计算图
  function layerBody() {
    var attn = row([
      node('attn_norm', 'RMSNorm', 'norm'),
      node('QKV proj', 'MUL_MAT', 'attn'),
      node('RoPE', 'position', 'attn'),
      node('KV cache', 'write', 'cache'),
      node('attention', 'GQA+softmax', 'attn'),
      node('Wo', 'MUL_MAT', 'attn'),
      node('+ residual', 'ADD', 'out')
    ]);
    var ffn = row([
      node('ffn_norm', 'RMSNorm', 'norm'),
      node('gate / up', 'MUL_MAT', 'mlp'),
      node('SiLU', 'swiglu', 'mlp'),
      node('down', 'MUL_MAT', 'mlp'),
      node('+ residual', 'ADD', 'out')
    ]);
    return '<div class="layer-body">' + attn + ffn + '</div>';
  }

  function layerBlock(i, open) {
    return '<div class="layer-block' + (open ? ' open' : '') + '">' +
      '<div class="layer-head">' +
        '<span class="idx">blk.' + i + '.</span>' +
        '<span class="name">Transformer 层</span>' +
        '<span class="meta">attn + ffn</span>' +
      '</div>' + layerBody() + '</div>';
  }

  var GRAPH_STEPS = [
    {
      title: '准备输入张量',
      desc: '推理从 token 序列开始。文本先被分词（tokenize）成 token id，再通过 token_embd 查表得到每个 token 的向量表示（嵌入）。',
      concepts: [
        ['tokenize', '"你是谁" -> token id 列表（BPE 分词）'],
        ['embeddings', 'token_embd.weight 是 [vocab, hidden] 的大表，查表即取行'],
        ['batch', '一次送入的 token 序列，形状 [n_tokens, n_embd]']
      ],
      logs: [
        { cls: 'l-graph', text: 'graph: build inp_tokens   = [3, 1] (I32)   tokens = [你是谁 的 id]' },
        { cls: 'l-graph', text: 'graph: build inp_pos      = [3, 1] (I32)   positions = [0,1,2]' },
        { cls: 'l-graph', text: 'graph: build token_embd   = MUL_MAT   [151936,1024] x [3,1]' }
      ],
      render: function (box) {
        box.innerHTML =
          '<div class="output-box"><span class="prompt">输入文本：</span>你是谁</div>' +
          '<div class="rowline" style="margin-top:16px">' +
            node('"你是谁"', 'text', 'input') + arrow() +
            node('tokenize', 'BPE', 'input') + arrow() +
            node('[id0, id1, id2]', 'token ids', 'input') + arrow() +
            node('inp_tokens', 'I32', 'weight') +
          '</div>' +
          '<div class="rowline" style="margin-top:12px">' +
            node('token_embd', 'MUL_MAT lookup', 'weight') + arrow() +
            node('embeddings', '[3, 1024]', 'attn') +
          '</div>';
      }
    },
    {
      title: '展开一层：注意力 + 前馈',
      desc: '每一层 Transformer 由两部分组成：注意力子层（混合 token 之间的信息）和前馈子层（对每个 token 做非线性变换），两者都带残差连接。',
      concepts: [
        ['注意力', '按 Query/Key 相似度加权聚合 Value，实现 token 间信息交换'],
        ['RMSNorm', '归一化，稳定数值；Qwen 用 RMSNorm 而非 LayerNorm'],
        ['SwiGLU', '门控前馈：gate 经 SiLU 门控 up，再投影回 hidden'],
        ['残差连接', 'x = x + sublayer(x)，保证梯度与信息直通']
      ],
      logs: [
        { cls: 'l-graph', text: 'graph: build inp_attn_norm  = RMS_NORM' },
        { cls: 'l-graph', text: 'graph: build attn_qkv       = MUL_MAT  (Q:2048, K:1024, V:1024)' },
        { cls: 'l-graph', text: 'graph: build rope           = ROPE     (按位置旋转 Q/K)' },
        { cls: 'l-graph', text: 'graph: build kv cache write = SET_ROWS (写入本层 KV)' },
        { cls: 'l-graph', text: 'graph: build attn           = FLASH_ATTN_EXT (GQA, 16/8 头)' },
        { cls: 'l-graph', text: 'graph: build attn_out       = MUL_MAT  (Wo)' },
        { cls: 'l-graph', text: 'graph: build ffn            = RMS_NORM -> gate/up -> SiLU -> down' },
        { cls: 'l-graph', text: 'graph: build l_out          = ADD      (两处残差相加)' }
      ],
      render: function (box) {
        box.innerHTML = '<div class="layer-block open">' +
          '<div class="layer-head"><span class="idx">blk.0.</span>' +
          '<span class="name">Transformer 层</span>' +
          '<span class="meta">' + MODEL.layers + ' 层中的第 0 层</span></div>' +
          layerBody() + '</div>' +
          '<p class="step-desc" style="margin-top:14px">上方为注意力子层，下方为前馈子层；两条 <code>+ residual</code> 为残差连接。</p>';
      }
    },
    {
      title: '堆叠 28 层',
      desc: '同样的结构重复 28 次，上一层的输出作为下一层的输入。这一结构决定了模型有 28 次信息混合与变换的机会。',
      concepts: [
        ['层堆叠', '0 -> 1 -> ... -> 27，逐层抽象，低层学语法，高层学语义'],
        ['张量复用', '同一份计算图模板，权重按 blk.N.* 索引切换'],
        ['graph 节点数', '28 层约 2200+ 个 ggml 节点，规模不小']
      ],
      logs: [
        { cls: 'l-graph', text: 'graph: layers 0..27 (repeating)  total nodes = 2240' },
        { cls: 'l-graph', text: 'graph: layer 0..27 share the same topology' }
      ],
      render: function (box) {
        var html = '<div class="layer-stack">' + layerBlock(0, true);
        for (var i = 1; i <= 3; i++) html += layerBlock(i, false);
        html += '<div class="layer-fold">...  第 4 ~ 26 层省略（结构相同）  ...</div>';
        html += layerBlock(27, false);
        html += '</div>';
        box.innerHTML = html;
        // 点击展开/折叠
        Array.prototype.forEach.call(box.querySelectorAll('.layer-head'), function (head) {
          head.addEventListener('click', function () {
            head.parentNode.classList.toggle('open');
          });
        });
      }
    },
    {
      title: '输出头：归一化 + lm_head',
      desc: '最后一层的输出经过 output_norm 归一化，再由 lm_head 投影到词表维度，得到每个 token 的分数（logits）。',
      concepts: [
        ['logits', '形状 [n_tokens, vocab]，每个位置对 15 万个 token 的分数'],
        ['权重共享', 'Qwen3-0.6B 的 lm_head 与 token_embd 共享权重（tie）'],
        ['下一步', 'logits 交给采样器，选出一个 token 作为输出']
      ],
      logs: [
        { cls: 'l-graph', text: 'graph: build output_norm   = RMS_NORM' },
        { cls: 'l-graph', text: 'graph: build result_output = MUL_MAT  [1024] x [151936,1024]' },
        { cls: 'l-graph', text: 'graph: build -> logits     = [3, 151936] (F32)' }
      ],
      render: function (box) {
        box.innerHTML =
          row([
            node('l_out (layer 27)', '[3, 1024]', 'attn'),
            node('output_norm', 'RMSNorm', 'norm'),
            node('lm_head', 'MUL_MAT', 'weight'),
            node('logits', '[3, 151936]', 'out')
          ]) +
          '<p class="step-desc" style="margin-top:16px">注意 logits 是 3 个位置各自的分数；生成时只关心最后一个位置。</p>' +
          '<div class="concept-card">计算图构建完成：一个从输入 token 到 logits 的有向无环图（DAG），共约 2240 个节点。</div>';
      }
    }
  ];

  // ================================================================ 阶段 3：推理计算
  var INFER_STEPS = [
    {
      title: 'llama_decode：拆分 batch',
      desc: '调用 llama_decode() 提交一批 token。内部先由 batch allocator 把逻辑 batch 拆分成物理 ubatch，以便分批或跨序列处理。',
      concepts: [
        ['逻辑 batch -> ubatch', '受 n_ubatch 限制，大 batch 会被切成多段'],
        ['prefill', '一次喂入整段 prompt，可并行计算所有位置'],
        ['token 位置', '每个 token 需要位置索引，用于 RoPE 与 KV 写入']
      ],
      logs: [
        { cls: 'l-comp', text: 'llama_decode: batch 3 tokens (1 sequence)' },
        { cls: 'l-comp', text: 'balloc: split -> 1 ubatch of 3 tokens' }
      ],
      render: function (box) {
        box.innerHTML =
          '<div class="rowline">' + node('llama_decode(batch)', 'API', 'input') + arrow() +
          node('balloc->init', 'planner', 'weight') + arrow() +
          node('ubatch[0]', '3 tokens', 'attn') + '</div>' +
          '<div class="concept-card" style="margin-top:16px">prompt 处理（prefill）阶段：3 个 token 可一次并行算完。</div>';
      }
    },
    {
      title: '分配 KV cache 槽位',
      desc: 'memory->init_batch 为这批 token 在 KV cache 中找到空闲槽位并绑定。若上下文已满，会返回 1（不是错误），提示需要滑动窗口或增大上下文。',
      concepts: [
        ['槽位(slot)', 'KV cache 按位置分配，每个 token 占一个位置'],
        ['复用', '多序列/多会话各自占用不同区间'],
        ['失败返回 1', '找不到槽位时返回 1，调用方可做上下文管理']
      ],
      logs: [
        { cls: 'l-comp', text: 'llama_kv_cache: init_batch: n_tokens = 3, n_ctx = ' + KV_DEMO_CTX },
        { cls: 'l-comp', text: 'llama_kv_cache: find_slot: assigned positions [0, 1, 2]' }
      ],
      render: function (box) {
        var cells = '';
        for (var i = 0; i < 48; i++) {
          cells += '<div class="kv-cell' + (i < 3 ? ' filled' : '') + '"></div>';
        }
        box.innerHTML =
          '<div style="font-family:var(--mono);font-size:11.5px;color:var(--text-dim);margin-bottom:8px">KV cache 位置（前 48 个示意）</div>' +
          '<div class="kv-grid">' + cells + '</div>' +
          '<div class="rowline" style="margin-top:14px">' +
            node('pos 0', 'token id0', 'input') + arrow() +
            node('pos 1', 'token id1', 'input') + arrow() +
            node('pos 2', 'token id2', 'input') +
          '</div>';
      }
    },
    {
      title: 'process_ubatch：构图 or 复用',
      desc: '处理每个 ubatch：先算图参数，若图拓扑与上次相同就复用旧图（省下构图与分配开销），否则重建图并把输入写入输入张量。',
      concepts: [
        ['图复用', '生成阶段每步拓扑不变，可跳过 build_graph，显著提速'],
        ['set_inputs', '把 token id / 位置等写入图的输入张量'],
        ['graph_params', 'ubatch + memory 上下文 + 图类型 共同决定拓扑']
      ],
      logs: [
        { cls: 'l-comp', text: 'process_ubatch: graph_params (n_tokens=3, n_seqs=1)' },
        { cls: 'l-comp', text: 'process_ubatch: can_reuse = false (first run) -> build_graph' },
        { cls: 'l-comp', text: 'process_ubatch: set_inputs -> inp_tokens, inp_pos, KQ_mask' }
      ],
      render: function (box) {
        box.innerHTML =
          '<div class="rowline">' +
            node('ubatch', '3 tokens', 'attn') + arrow() +
            node('graph_params', '决策', 'weight') + arrow() +
            node('can_reuse?', '是/否', 'norm') +
          '</div>' +
          '<div class="rowline" style="margin-top:14px">' +
            node('build_graph', '重建', 'mlp') + arrow() +
            node('alloc_graph', '分配缓冲', 'mlp') + arrow() +
            node('set_inputs', '填充输入', 'out') +
          '</div>' +
          '<div class="concept-card" style="margin-top:16px">复用分支会跳过前两步，直接到 set_inputs。生成第 2 个 token 起通常走复用。</div>';
      }
    },
    {
      title: '后端调度：切分图（split）',
      desc: '把整张图按节点所在的后端切分成若干段（split）：连续同后端的节点聚成一段，设备切换处切开。这实现了"部分层放 GPU、部分放 CPU"的混合执行。',
      concepts: [
        ['split', '一段可在单一后端连续执行的子图，含区间与输入表'],
        ['混合 offload', '-ngl 20 会把前 20 层放 GPU、其余留 CPU'],
        ['输入拷贝', '段之间数据依赖靠拷贝衔接，已用异步流水线掩盖']
      ],
      logs: [
        { cls: 'l-comp', text: 'sched_reserve: splitting graph into splits' },
        { cls: 'l-comp', text: 'split 0: CPU   nodes [0,2)    (token_embd)' },
        { cls: 'l-comp', text: 'split 1: GPU   nodes [2,1500) (blk.0 .. blk.19)' },
        { cls: 'l-comp', text: 'split 2: CPU   nodes [1500,2240) (blk.20 .. output)' }
      ],
      render: function (box) {
        box.innerHTML =
          '<div style="font-family:var(--mono);font-size:11.5px;color:var(--text-dim);margin-bottom:8px">' +
          '示例：-ngl 20（前 20 层 offload 到 GPU）</div>' +
          '<div class="split-row">' +
            '<div class="split-seg cpu" style="flex:1"><span class="dev">CPU</span><span class="rg">embd [0,2)</span></div>' +
            '<div class="split-seg gpu" style="flex:9"><span class="dev">GPU</span><span class="rg">blk.0 .. blk.19</span></div>' +
            '<div class="split-seg cpu" style="flex:4"><span class="dev">CPU</span><span class="rg">blk.20 .. head</span></div>' +
          '</div>' +
          '<p class="step-desc" style="margin-top:14px">每段独立提交给对应后端；段与段之间用输入张量拷贝衔接。</p>';
      }
    },
    {
      title: '执行计算图',
      desc: '逐段提交：compute_splits 把每段异步发给对应后端。CPU 后端先做计算规划（ggml_graph_plan），再多线程执行（ggml_graph_compute）。完成后 logits 异步拷回内存。',
      concepts: [
        ['异步执行', 'async 提交后不等完成，用同步点或事件等待'],
        ['多线程', 'CPU 后端按 n_threads 把节点任务分给线程池'],
        ['logits 拷回', '结果从设备缓冲拷回主机，供采样器读取']
      ],
      logs: [
        { cls: 'l-comp', text: 'compute_splits: split 0 (CPU)  graph_compute' },
        { cls: 'l-time', text: 'compute_splits: split 0 done (embd)' },
        { cls: 'l-comp', text: 'compute_splits: split 1 (GPU)  graph_compute' },
        { cls: 'l-time', text: 'compute_splits: split 1 done (20 layers)' },
        { cls: 'l-comp', text: 'compute_splits: split 2 (CPU)  graph_compute' },
        { cls: 'l-info', text: 'llama_decode: copy logits to host (3 x 151936)' },
        { cls: 'l-time', text: 'llama_perf: eval time = 128.40 ms / 3 tokens' }
      ],
      render: function (box) {
        var threads = '';
        for (var i = 0; i < 8; i++) threads += '<div class="thread" data-i="' + i + '">T' + i + '</div>';
        box.innerHTML =
          '<div class="split-row">' +
            '<div class="split-seg cpu" data-seg="0" style="flex:1"><span class="dev">CPU</span></div>' +
            '<div class="split-seg gpu" data-seg="1" style="flex:9"><span class="dev">GPU</span></div>' +
            '<div class="split-seg cpu" data-seg="2" style="flex:4"><span class="dev">CPU</span></div>' +
          '</div>' +
          '<div style="margin-top:16px;font-family:var(--mono);font-size:11.5px;color:var(--text-dim)">CPU 线程池</div>' +
          '<div class="threads" style="margin-top:8px">' + threads + '</div>' +
          '<div class="rowline" style="margin-top:16px">' +
            node('device buffer', 'logits', 'attn') + arrow() +
            node('host memory', 'async copy', 'out') +
          '</div>';
        // 动画：依次高亮 split 与线程
        var segs = box.querySelectorAll('.split-seg');
        var ths = box.querySelectorAll('.thread');
        var k = 0;
        var timer = setInterval(function () {
          Array.prototype.forEach.call(segs, function (s) { s.classList.remove('hl'); });
          Array.prototype.forEach.call(ths, function (t) { t.classList.remove('busy'); });
          if (k >= segs.length) { clearInterval(timer); return; }
          segs[k].classList.add('hl');
          Array.prototype.forEach.call(ths, function (t) { t.classList.add('busy'); });
          setTimeout(function () {
            Array.prototype.forEach.call(ths, function (t) { t.classList.remove('busy'); });
          }, 500);
          k++;
        }, 900);
        box.dataset.timer = 'set';
      }
    }
  ];

  // ================================================================ 阶段 4：采样与解码
  // 示意候选：【token 文本, 概率】
  var CANDS = [
    ['我', 0.382], ['你', 0.176], ['您', 0.118], ['你好', 0.074],
    ['请问', 0.052], ['您好', 0.041], ['是', 0.028], ['这', 0.019]
  ];
  var TOP_K = 5;
  var TOP_P = 0.90;

  function histHTML(cands, opts) {
    opts = opts || {};
    var max = Math.max.apply(null, cands.map(function (c) { return c[1]; }));
    return '<div class="hist">' + cands.map(function (c, i) {
      var h = Math.round((c[1] / max) * 100);
      var cls = 'bar-wrap';
      if (opts.cut && i >= opts.keep) cls += ' cut';
      if (opts.sel === i) cls += ' sel';
      return '<div class="' + cls + '">' +
        '<div class="val">' + (c[1] * 100).toFixed(1) + '%</div>' +
        '<div class="bar" style="height:' + h + '%"></div>' +
        '<div class="label">' + esc(c[0]) + '</div>' +
      '</div>';
    }).join('') + '</div>';
  }

  var SAMPLE_STEPS = [
    {
      title: '取最后一个位置的 logits',
      desc: '生成时只关心序列最后一个位置：它代表"下一个 token 应该是什么"。从这个位置取出长度为 151936 的分数向量。',
      concepts: [
        ['为什么是最后一个', '自回归：前面的 token 已确定，只需预测下一个'],
        ['logits', '未归一化的分数，可为任意实数'],
        ['温度前', '此时还没做 softmax，不能直接当概率用']
      ],
      logs: [
        { cls: 'l-samp', text: 'llama_get_logits_ith: idx = -1 (最后一个位置)' },
        { cls: 'l-samp', text: 'logits shape = [151936], dtype = F32' }
      ],
      render: function (box) {
        box.innerHTML =
          row([
            node('logits', '[3, 151936]', 'weight'),
            node('取最后一行', 'idx=-1', 'norm'),
            node('分数向量', '[151936]', 'input')
          ]) +
          '<div class="concept-card" style="margin-top:16px">15 万个候选，每个都有一个分数；分数越大表示模型越"倾向"该 token。</div>';
      }
    },
    {
      title: 'softmax -> 概率分布',
      desc: '对分数做 softmax，得到总和为 1 的概率分布。这时才能按概率排序、过滤和采样。下面展示概率最高的 8 个候选（示意数据）。',
      concepts: [
        ['softmax', 'softmax(x_i) = exp(x_i) / sum(exp(x_j))'],
        ['长尾', '15 万候选里绝大多数概率接近 0'],
        ['采样目标', '从高概率区挑一个，兼顾准确与多样']
      ],
      logs: [
        { cls: 'l-samp', text: 'sampler chain apply: softmax -> probs[151936]' },
        { cls: 'l-samp', text: 'top candidates: 我 0.382 | 你 0.176 | 您 0.118 | 你好 0.074 ...' }
      ],
      render: function (box) {
        box.innerHTML = histHTML(CANDS) +
          '<p class="step-desc" style="margin-top:14px">横轴为候选 token，高度为概率（仅展示前 8 个）。</p>';
      }
    },
    {
      title: 'top-k 过滤',
      desc: '只保留概率最高的 k 个候选，其余直接丢弃。它砍掉长尾，防止模型偶尔抽到完全无关的低概率 token。',
      concepts: [
        ['top-k', '保留前 k 个，k=1 等价于贪心'],
        ['k 越大越随机', 'k 太小易重复，太大易跑偏'],
        ['与 top-p 配合', '通常先 top-k 粗过滤，再 top-p 精修']
      ],
      logs: [
        { cls: 'l-samp', text: 'top_k: k = ' + TOP_K + ' -> keep 5, cut 3' }
      ],
      render: function (box) {
        box.innerHTML = histHTML(CANDS, { cut: true, keep: TOP_K }) +
          '<p class="step-desc" style="margin-top:14px">灰色为被 top-k=5 砍掉的候选：请问 / 您好 / 是 / 这。</p>';
      }
    },
    {
      title: 'temperature + top-p 核采样',
      desc: 'temperature 调整分布的陡峭程度；top-p 从高概率往低累加，达到阈值 p 就截断，只从"核"里采样。它按概率质量自适应决定保留多少。',
      concepts: [
        ['temperature', 't<1 更尖锐更确定，t>1 更平坦更随机'],
        ['top-p', '累积概率达 p 即截断，p 越大保留越多'],
        ['自适应', '分布尖锐时少留，平缓时多留']
      ],
      logs: [
        { cls: 'l-samp', text: 'temp: t = 0.7 -> 重缩放 logits' },
        { cls: 'l-samp', text: 'top_p: p = ' + TOP_P + ' -> cumsum 0.382+0.176+0.118 = 0.676 < 0.9, +0.074 = 0.750, +0.052 = 0.802, +0.041 = 0.843 ...' },
        { cls: 'l-samp', text: 'top_p: keep 4 candidates, renormalize' }
      ],
      render: function (box) {
        // 温度 0.7 后重新归一化的示意概率
        var adjusted = [['我', 0.45], ['你', 0.19], ['您', 0.12], ['你好', 0.08], ['请问', 0.05]];
        box.innerHTML =
          '<div style="font-family:var(--mono);font-size:11.5px;color:var(--text-dim);margin-bottom:8px">temperature = 0.7 并 top-p = ' + TOP_P + ' 之后的保留候选</div>' +
          histHTML(adjusted, { sel: -1 }) +
          '<p class="step-desc" style="margin-top:14px">核（top-p）把累积概率收放至 0.9 附近，剩余候选中随机抽一个。</p>';
      }
    },
    {
      title: '选出 token 并解码为文本',
      desc: '采样器选中一个 token id，再把它解码为文本片段（piece）拼接到输出。该 token 又作为下一步的输入，循环往复直到遇到 EOS。',
      concepts: [
        ['token -> piece', 'llama_token_to_piece 把 id 还原成字符串'],
        ['自回归循环', '生成的 token 追加到序列，继续预测下一个'],
        ['EOS', '结束标记（id ' + MODEL.eos + '），遇到即停止']
      ],
      logs: [
        { cls: 'l-samp', text: 'sampled token id = 100000 ("我")' },
        { cls: 'l-samp', text: 'llama_token_to_piece: id -> "我"' },
        { cls: 'l-samp', text: '下一轮: llama_decode(我) -> logits -> 采样 ...' },
        { cls: 'l-samp', text: 'repeat until EOS (' + MODEL.eos + ') or n_predict' }
      ],
      render: function (box) {
        var cs = CANDS.slice();
        box.innerHTML =
          histHTML(cs, { sel: 0 }) +
          '<div class="output-box" style="margin-top:16px">' +
            '<span class="prompt">你是谁</span><span class="gen">我可以帮你回答各种问题</span>' +
            '<span class="cursor"></span>' +
          '</div>' +
          '<div class="rowline" style="margin-top:14px">' +
            node('sampled token', '"我"', 'out') + arrow() +
            node('token_to_piece', 'decode', 'norm') + arrow() +
            node('追加到输出', '循环', 'input') +
          '</div>';
      }
    }
  ];

  // ================================================================ 阶段元数据
  var STAGES = [
    { id: 'load',   title: '模型加载',   sub: 'GGUF -> 权重 -> 上下文', steps: LOAD_STEPS },
    { id: 'graph',  title: '构建计算图', sub: 'token -> 28 层 -> logits', steps: GRAPH_STEPS },
    { id: 'infer',  title: '推理计算',   sub: 'decode -> split -> 执行',  steps: INFER_STEPS },
    { id: 'sample', title: '采样与解码', sub: 'logits -> 概率 -> token', steps: SAMPLE_STEPS }
  ];

  // ================================================================ 控制器
  var App = {
    stage: 0,
    step: 0,
    playing: false,
    timer: null,

    init: function () {
      var self = this;
      Log.init(document.getElementById('log'));
      document.getElementById('btn-next').addEventListener('click', function () { self.next(); });
      document.getElementById('btn-prev').addEventListener('click', function () { self.prev(); });
      document.getElementById('btn-play').addEventListener('click', function () { self.togglePlay(); });
      document.getElementById('btn-reset').addEventListener('click', function () { self.reset(); });
      var mf = document.getElementById('model-file');
      if (mf) mf.addEventListener('change', function (e) {
        var f = e.target.files && e.target.files[0];
        if (f) self.loadModelFile(f);
      });
      window.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowRight') self.next();
        if (e.key === 'ArrowLeft') self.prev();
      });
      this.render();
    },

    // 打开系统文件选择框（由左侧“加载模型”按钮触发）
    openModelPicker: function () {
      var mf = document.getElementById('model-file');
      if (mf) mf.click();
    },

    // 读取并解析所选 GGUF；成功后更新数据源并重置到第一步
    loadModelFile: function (file) {
      var self = this;
      var MAX_HEAD = 32 * 1024 * 1024;
      var fr = new FileReader();
      fr.onload = function () {
        try {
          var info = window.parseGGUF(fr.result);
          setModelFromGGUF(info, file);
          self.stopPlay();
          self.stage = 0;
          self.step = 0;
          self.render();
        } catch (err) {
          var msg = '解析 GGUF 失败：' + (err && err.message ? err.message : err);
          if (typeof alert === 'function') alert(msg); else console.error(msg);
        }
      };
      fr.onerror = function () {
        if (typeof alert === 'function') alert('读取文件失败');
      };
      fr.readAsArrayBuffer(file.slice(0, Math.min(file.size, MAX_HEAD)));
      var mf = document.getElementById('model-file');
      if (mf) mf.value = ''; // 允许重复选择同一文件
    },

    totalSteps: function () {
      return STAGES.reduce(function (a, s) { return a + s.steps.length; }, 0);
    },
    doneSteps: function () {
      var n = 0;
      for (var i = 0; i < this.stage; i++) n += STAGES[i].steps.length;
      return n + this.step + 1;
    },

    next: function () {
      var s = STAGES[this.stage];
      if (this.step < s.steps.length - 1) {
        this.step++;
      } else if (this.stage < STAGES.length - 1) {
        this.stage++;
        this.step = 0;
      } else {
        this.stopPlay();
        return;
      }
      this.render();
    },
    prev: function () {
      if (this.step > 0) {
        this.step--;
      } else if (this.stage > 0) {
        this.stage--;
        this.step = STAGES[this.stage].steps.length - 1;
      } else {
        return;
      }
      this.render();
    },
    reset: function () {
      this.stopPlay();
      this.stage = 0;
      this.step = 0;
      this.render();
    },
    togglePlay: function () {
      if (this.playing) this.stopPlay();
      else this.startPlay();
    },
    startPlay: function () {
      var self = this;
      this.playing = true;
      document.getElementById('btn-play').textContent = '暂停';
      this.timer = setInterval(function () {
        var lastStage = self.stage === STAGES.length - 1;
        var lastStep = self.step === STAGES[self.stage].steps.length - 1;
        if (lastStage && lastStep) { self.stopPlay(); return; }
        self.next();
      }, 2200);
    },
    stopPlay: function () {
      this.playing = false;
      if (this.timer) { clearInterval(this.timer); this.timer = null; }
      var b = document.getElementById('btn-play');
      if (b) b.textContent = '自动播放';
    },

    // 渲染整个界面
    render: function () {
      var stage = STAGES[this.stage];
      var step = stage.steps[this.step];

      // 标签页
      var tabs = document.getElementById('tabs');
      tabs.innerHTML = STAGES.map(function (s, i) {
        var cls = 'stage-tab';
        if (i === App.stage) cls += ' active';
        else if (i < App.stage) cls += ' done';
        return '<div class="' + cls + '" data-i="' + i + '"><span class="num">' + (i + 1) + '</span>' + esc(s.title) + '</div>';
      }).join('');
      Array.prototype.forEach.call(tabs.querySelectorAll('.stage-tab'), function (t) {
        t.addEventListener('click', function () {
          App.stopPlay();
          App.stage = parseInt(t.dataset.i, 10);
          App.step = 0;
          App.render();
        });
      });

      // 左侧
      document.getElementById('stage-sub').textContent = stage.sub;
      document.getElementById('step-title').textContent = step.title;
      document.getElementById('step-desc').innerHTML = step.desc;
      var stepsEl = document.getElementById('steps');
      stepsEl.innerHTML = stage.steps.map(function (st, i) {
        var cls = i === App.step ? 'active' : (i < App.step ? 'done' : '');
        return '<li class="' + cls + '" data-i="' + i + '">' + (i + 1) + '. ' + esc(st.title) + '</li>';
      }).join('');
      // 点击步骤项直接跳到该步（本阶段内），并停止自动播放
      Array.prototype.forEach.call(stepsEl.querySelectorAll('li'), function (li) {
        li.addEventListener('click', function () {
          var target = parseInt(li.dataset.i, 10);
          if (target === App.step) return;
          App.stopPlay();
          App.step = target;
          App.render();
        });
      });

      // “加载模型”入口：仅出现在“模型加载”阶段，位于步骤列表之后，不计入步骤数
      var slot = document.getElementById('load-model-slot');
      if (slot) {
        if (stage.id === 'load') {
          slot.innerHTML = '<button class="btn load-model-btn" id="btn-load-model">' +
            (isLoaded() ? '重新加载模型 (.gguf)' : '加载模型 (.gguf)') + '</button>';
          var lb = slot.querySelector('#btn-load-model');
          if (lb) lb.addEventListener('click', function () { App.openModelPicker(); });
        } else {
          slot.innerHTML = '';
        }
      }

      document.getElementById('concepts').innerHTML = concepts(step.concepts);

      // 中间舞台
      var head = document.getElementById('stage-head');
      head.innerHTML = '<span class="title">' + esc(stage.title) + '</span>' +
        '<span class="badge">' + esc(stage.id) + ' / step ' + (this.step + 1) + '-' + stage.steps.length + '</span>';
      var viz = document.getElementById('viz');
      viz.innerHTML = '';
      step.render(viz);

      // 右侧日志：累积当前阶段到当前 step 的所有日志（logs 可为数组或函数）
      var logs = [];
      for (var i = 0; i <= this.step; i++) {
        var ls = stage.steps[i].logs;
        logs = logs.concat(typeof ls === 'function' ? ls() : (ls || []));
      }
      logs.push({ cls: 'l-hl', text: '--- 步骤 ' + (this.step + 1) + ': ' + step.title + ' ---' });
      Log.setStageLogs(logs);

      // 右侧参数表：跟随 MODEL 单一数据源
      document.getElementById('params').innerHTML = [
        ['arch', MODEL.arch],
        ['n_layer', MODEL.layers],
        ['n_embd', MODEL.hidden],
        ['n_head / n_head_kv', MODEL.heads + ' / ' + MODEL.kvHeads],
        ['head_dim', MODEL.headDim],
        ['n_ff', MODEL.ffn],
        ['n_vocab', MODEL.vocab],
        ['rope_theta', MODEL.ropeTheta],
        ['ctx_length', MODEL.ctxLen],
        ['file_type', MODEL.fileType]
      ].map(function (r) {
        return '<div class="p"><span>' + esc(r[0]) + '</span><span>' + esc(r[1]) + '</span></div>';
      }).join('');

      // 顶部模型徽章
      var badge = document.getElementById('model-badge');
      if (badge) {
        badge.innerHTML = '<span class="dot"></span>' + esc(MODEL.name) + ' · ' + MODEL.layers + ' 层 · GQA ' +
          MODEL.heads + '/' + MODEL.kvHeads +
          (isLoaded() ? ' · <span style="color:var(--green)">已加载</span>' : '');
      }

      // 底部进度
      document.getElementById('progress').style.width =
        Math.round((this.doneSteps() / this.totalSteps()) * 100) + '%';
      document.getElementById('status').textContent =
        '阶段 ' + (this.stage + 1) + '/' + STAGES.length + '  -  步骤 ' +
        (this.step + 1) + '/' + stage.steps.length + '  -  总进度 ' +
        this.doneSteps() + '/' + this.totalSteps();
      document.getElementById('btn-prev').disabled = (this.stage === 0 && this.step === 0);
      document.getElementById('btn-next').disabled =
        (this.stage === STAGES.length - 1 && this.step === stage.steps.length - 1);
    }
  };

  document.addEventListener('DOMContentLoaded', function () { App.init(); });
  window.__App = App;
})();
