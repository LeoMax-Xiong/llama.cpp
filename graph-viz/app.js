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

  // 全局输入文本（用户在“准备输入张量”中输入，后续步骤共用）
  var INPUT_TEXT = '你是谁';
  var TOKENIZER = null; // 加载模型后构建（需要真实词表）

  // 当前展开详情的算子：{ layer: 层号, op: 算子名 } 或 null
  var LAYER_DETAIL = null;

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

  // 用模型的词表/合并表构建分词器
  function buildTokenizer(info) {
    TOKENIZER = null;
    if (!window.GGUFTokenizer) return;
    var t = metaGet(info.metadata, 'tokenizer.ggml.tokens');
    var mg = metaGet(info.metadata, 'tokenizer.ggml.merges');
    var full = function (x) { return x && x.__array && x.full && x.sample; };
    if (full(t) && full(mg)) {
      try {
        TOKENIZER = new window.GGUFTokenizer(t.sample, mg.sample);
      } catch (e) {
        TOKENIZER = null;
      }
    }
  }

  // ------------------------------------------------ 输入文本分词（真实 BPE）
  function tokenizeInfo() {
    if (!TOKENIZER) return { ok: false, reason: 'nolm' };
    try {
      var toks = TOKENIZER.encode(INPUT_TEXT);
      return {
        ok: true,
        toks: toks,
        chars: INPUT_TEXT.length,
        bytes: new TextEncoder().encode(INPUT_TEXT).length
      };
    } catch (e) {
      return { ok: false, reason: 'err', err: e.message };
    }
  }
  // 当前输入的 token 数（无分词器时回退示例值 3）
  function tokenCount() {
    var info = tokenizeInfo();
    return info.ok ? info.toks.length : 3;
  }
  function tokenStatHTML(info) {
    if (!info.ok) {
      return '<p class="step-desc" style="margin:8px 0">' +
        (info.reason === 'nolm' ? '请先加载模型（需要真实词表）以启用分词。'
                                : '分词失败：' + esc(info.err || '')) + '</p>';
    }
    return '<div style="font-family:var(--mono);font-size:11.5px;color:var(--text-dim);margin:8px 0">' +
      'token 数 ' + info.toks.length + ' · 字符数 ' + info.chars + ' · UTF-8 字节 ' + info.bytes + '</div>';
  }
  function tokenTableHTML(toks) {
    if (!toks || !toks.length) return '<p class="step-desc">（空输入）</p>';
    return '<table class="insp-table"><thead><tr>' +
      '<th class="idx">#</th><th>token id</th><th>token 原文</th><th>对应原文片段</th>' +
      '</tr></thead><tbody>' + toks.map(function (t, i) {
        return '<tr><td class="idx">' + (i + 1) + '</td>' +
               '<td class="val">' + t.id + '</td>' +
               '<td class="key">' + esc(t.token) + '</td>' +
               '<td class="desc">' + esc(t.text) + '</td></tr>';
      }).join('') + '</tbody></table>';
  }
  // ------------------------------------------------ 算子详情（数学过程）
  function rmsEps() {
    if (!MODEL.meta) return '1e-06';
    var v = metaGet(MODEL.meta, MODEL.arch + '.attention.layer_norm_rms_epsilon');
    return v != null ? String(v) : '1e-06';
  }

  // MathJax 排版（等其就绪后再排）
  function typesetMath(el) {
    var MJ = window.MathJax;
    if (!MJ || !el) return;
    var run = function () {
      if (MJ.typesetPromise) MJ.typesetPromise([el]).catch(function () {});
    };
    if (MJ.startup && MJ.startup.promise) MJ.startup.promise.then(run).catch(function () {});
    else run();
  }

  // 算子详情面板分发：按算子名选择对应的数学过程说明
  function layerDetailHTML(op, layer) {
    if (op === 'attn_norm') return detailAttnNorm(layer);
    if (op === 'Q proj') return detailQProj(layer);
    if (op === 'K proj') return detailKProj(layer);
    if (op === 'V proj') return detailVProj(layer);
    return '';
  }

  function detailAttnNorm(layer) {
    var d = MODEL.hidden;
    var eps = rmsEps();
    var n = tokenCount();
    var src = (layer != null && layer >= 0) ? '<span class="ldetail-src">来自 blk.' + layer + '.</span> ' : '';
    return '<div class="ldetail" id="layer-detail">' +
      '<div class="ldetail-head">' +
        '<span>' + src + 'attn_norm · RMSNorm（均方根归一化）</span>' +
        '<span class="ldetail-hint">再点一次算子名可收起</span>' +
      '</div>' +
      '<div class="ldetail-body">' +
        '<p class="ldetail-p"><b>作用</b>：进入注意力之前，把每个 token 的向量缩放到「单位均方」，稳定后续数值。</p>' +
        '<p class="ldetail-p"><b>输入 / 输出</b>：$x \\in \\mathbb{R}^{n \\times d}$，其中 $n = ' + n +
          '$（token 数）、$d = ' + d + '$（隐藏维）；输出形状不变。</p>' +

        '<h4>① 平方均值（mean square）</h4>' +
        '$$ \\mathrm{ms}(x) = \\frac{1}{d}\\sum_{i=1}^{d} x_i^2 $$' +
        '<p class="ldetail-p">对这一个 token 的 $d = ' + d + '$ 个分量求平方后取平均。</p>' +

        '<h4>② 均方根（root mean square）</h4>' +
        '$$ \\mathrm{rms}(x) = \\sqrt{\\mathrm{ms}(x) + \\epsilon},\\qquad \\epsilon = 10^{-6} $$' +
        '<p class="ldetail-p">$\\epsilon$ 是防止除以 0 的小量；取自模型元数据 <code>' + MODEL.arch +
          '.attention.layer_norm_rms_epsilon</code> = ' + eps + '（即 $10^{-6}$）。</p>' +

        '<h4>③ 归一化</h4>' +
        '$$ \\hat{x}_i = \\frac{x_i}{\\mathrm{rms}(x)} $$' +

        '<h4>③′ 归一化后的性质：均方 = 1（平方和 = d）</h4>' +
        '<p class="ldetail-p">把 ③ 代入 ② 可得（忽略 $\\epsilon$）：</p>' +
        '$$ \\sum_{i=1}^{d}\\hat{x}_i^2 = \\frac{\\sum_i x_i^2}{\\mathrm{rms}^2} \\approx d $$' +
        '<p class="ldetail-p">也就是：<b>均方 = 1</b>、<b>RMS = 1</b>、L2 范数 $\\|\\hat{x}\\|_2 = \\sqrt{d}$。</p>' +
        '<p class="ldetail-p">小例子（$d = 4$）：$x = [1,\\, 2,\\, 3,\\, 4]$ → $\\mathrm{ms} = 7.5$ → ' +
          '$\\mathrm{rms} \\approx 2.739$ → $\\hat{x} \\approx [0.365,\\, 0.730,\\, 1.095,\\, 1.460]$ → 平方和 $\\approx 4 = d$。</p>' +
        '<p class="ldetail-p">为什么归一化到「均方 = 1」而不是「平方和 = 1」？后者会让单个分量的平均大小 ' +
          '$\\approx 1/\\sqrt{d}$，随维度变化；而均方 = 1 使<b>每个分量的量级与维度 $d$ 解耦</b>' +
          '——无论隐藏维是 1024 还是 4096，输入尺度都稳定。</p>' +

        '<h4>④ 逐元素缩放（可学习参数）</h4>' +
        '$$ y_i = \\hat{x}_i \\cdot w_i $$' +
        '<p class="ldetail-p">$w$ 是权重张量 <code>blk.N.attn_norm.weight</code>（$\\mathbb{R}^{' + d + '}$），逐元素相乘。</p>' +
        '<p class="ldetail-p">注意：乘上 $w$ 之后，输出<b>不再</b>满足平方和 = $d$' +
          '（$\\sum_i y_i^2 = \\sum_i \\hat{x}_i^2 w_i^2$）——归一化负责「定尺度」，$w$ 负责「按维度重新调整尺度与重要性」。</p>' +

        '<h4>合并成一条公式</h4>' +
        '$$ y = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum_{i=1}^{d} x_i^2 + \\epsilon}} \\odot w $$' +

        '<h4>与 LayerNorm 的区别</h4>' +
        '<table class="ldetail-table"><thead><tr><th>项</th><th>LayerNorm</th><th>RMSNorm（本层）</th></tr></thead><tbody>' +
        '<tr><td>是否减均值</td><td>是：$\\dfrac{x-\\mu}{\\sigma}$</td><td><b>否</b>：只除以 RMS</td></tr>' +
        '<tr><td>中心化</td><td>需要（求 $\\mu$）</td><td>不需要</td></tr>' +
        '<tr><td>计算量</td><td>较大（均值 + 方差）</td><td>较小（只求均方）</td></tr>' +
        '<tr><td>可学习参数</td><td>缩放 $\\gamma$ + 偏置 $\\beta$</td><td>仅缩放 $w$</td></tr>' +
        '</tbody></table>' +

        '<h4>为什么这样设计</h4>' +
        '<ul class="ldetail-ul">' +
        '<li><b>尺度不变性</b>：若输入整体放大 $k$ 倍（$x \\to kx$），$\\mathrm{rms}$ 也放大 $k$ 倍，相除后 ' +
          '$\\hat{x}$ 完全不变——结果只看「相对大小」，与输入的绝对量级无关。</li>' +
        '<li><b>数值稳定</b>：无论上一层输出多大，送进 Q/K/V 投影的输入都被压回固定量级，避免 fp16/bf16 溢出或下溢。</li>' +
        '<li><b>训练稳定</b>：防止激活值随层数加深而爆炸/消失，也让梯度不被大激活主导。</li>' +
        '<li><b>参数更少、算得更快</b>：省掉均值与偏置（而「减均值」的收益本身就很小），大模型下收益可观。</li>' +
        '<li><b>让 $w$ 的职责更纯粹</b>：归一化负责定尺度，$w$ 只需学习各维度的重要性与相对尺度。</li>' +
        '</ul>' +

        '<h4>在 llama.cpp 里的实现</h4>' +
        '<p class="ldetail-p"><code>cur = ggml_rms_norm(ctx0, cur);</code><br>' +
          '<code>cur = ggml_mul(ctx0, cur, layer.attn_norm);</code></p>' +

        '<h4>参数量</h4>' +
        '<p class="ldetail-p">每层仅 ' + d + ' 个可学习参数（$w$）；全模型 ' + MODEL.layers + ' 层共 ' +
          (d * MODEL.layers).toLocaleString() + ' 个。</p>' +

        '<h4>出处</h4>' +
        '<p class="ldetail-p">Zhang &amp; Sennrich, 2019, "Root Mean Square Layer Normalization", arXiv:1910.07467。</p>' +
      '</div>' +
    '</div>';
  }

  // Q proj：x_norm 到 Q 的线性投影（多头拼接后再 reshape）
  function detailQProj(layer) {
    var n = tokenCount();
    var din = MODEL.hidden;
    var heads = MODEL.heads;
    var hd = MODEL.headDim;
    var dout = heads * hd;
    var kde = MODEL.kvHeads * hd;
    var src = (layer != null && layer >= 0) ? '<span class="ldetail-src">来自 blk.' + layer + '.</span> ' : '';
    return '<div class="ldetail" id="layer-detail">' +
      '<div class="ldetail-head">' +
        '<span>' + src + 'Q proj · 查询投影（Query projection）</span>' +
        '<span class="ldetail-hint">再点一次算子名可收起</span>' +
      '</div>' +
      '<div class="ldetail-body">' +
        '<p class="ldetail-p"><b>作用</b>：把 attn_norm 的输出 x_norm 线性投影成注意力要用的 Query 向量（每个 token 带上“我在找什么”的查询）。</p>' +
        '<p class="ldetail-p"><b>输入 / 输出</b>：$x_{norm} \\in \\mathbb{R}^{n \\times d_{in}}$，' +
          '$W_q \\in \\mathbb{R}^{d_{in} \\times d_{out}}$，输出 $Q \\in \\mathbb{R}^{n \\times d_{out}}$；' +
          '其中 $n = ' + n + '$、$d_{in} = ' + din + '$、$d_{out} = ' + dout + '$。</p>' +

        '<h4>① 矩阵乘法（行 · 列 做点积）</h4>' +
        '$$ Q = x_{norm} \\cdot W_q $$' +
        '$$ Q_{t,j} = \\sum_{k=1}^{' + din + '} x_{norm}[t,k] \\cdot W_q[k,j] $$' +
        '<p class="ldetail-p">第 $t$ 个 token 的 ' + din + ' 维向量，与 $W_q$ 的第 $j$ 列做点积，得到该 token 的第 $j$ 个输出分量；$j$ 遍历 ' + dout + ' 列。</p>' +

        '<h4>② 形状流转</h4>' +
        '<p class="ldetail-p"><code>[' + n + ', ' + din + '] × [' + din + ', ' + dout + '] -> [' + n + ', ' + dout + ']</code></p>' +
        '<p class="ldetail-p">token 维（$n$）不参与变换：每个 token 独立地做同一次投影。</p>' +

        '<h4>③ 输出为什么是 ' + dout + '，不是 ' + din + '？</h4>' +
        '<p class="ldetail-p">因为 Q 是<b>所有注意力头拼在一起</b>：' +
          '$d_{out} = n_{head} \\times d_{head} = ' + heads + ' \\times ' + hd + ' = ' + dout + '$。</p>' +
        '<p class="ldetail-p">对比 K/V：GQA 下 K/V 只有 $' + MODEL.kvHeads + '$ 个头，故 $d_{out}^{K} = ' + kde + '$。' +
          '另外 Qwen3 的 $d_{head} = ' + hd + '$ 是模型显式设定的，不等于 $d_{in}/n_{head} = ' + (din / heads) + '$。</p>' +

        '<h4>④ 紧接着拆成多头</h4>' +
        '<p class="ldetail-p"><code>[' + n + ', ' + dout + '] -> [' + n + ', ' + heads + ', ' + hd + ']</code></p>' +
        '<p class="ldetail-p">投影只是“把所有头排在一起”，之后要 reshape 成 ' + heads + ' 个头、每头 ' + hd + ' 维，各头独立算注意力 —— 这就是 RoPE 框标注 Q [' + n + ', ' + heads + ', ' + hd + '] 的来源。</p>' +

        '<h4>⑤ 在 llama.cpp 里的实现</h4>' +
        '<p class="ldetail-p"><code>Qcur = ggml_mul_mat(ctx0, layer.wq, cur);</code><br>' +
          '<code>Qcur = ggml_reshape_3d(ctx0, Qcur, ' + hd + ', ' + heads + ', n_tokens);</code></p>' +
        '<p class="ldetail-p">ggml_mul_mat 一次算完整批 token（$n$ 行一起），不是逐 token 循环；权重取自 GGUF 的 ' +
          '<code>blk.' + layer + '.attn_q.weight</code>，形状 [' + din + ', ' + dout + ']。</p>' +

        '<h4>⑥ 参数量</h4>' +
        '<p class="ldetail-p">$' + din + ' \\times ' + dout + ' = ' + (din * dout).toLocaleString() + '$ 个权重；' +
          'Qwen3 的 Q/K/V 投影都<b>没有 bias</b>（无加性项）。</p>' +

        '<h4>⑦ 三个投影的对照</h4>' +
        '<table class="ldetail-table"><thead><tr><th>投影</th><th>权重形状</th><th>输出</th><th>头数</th></tr></thead><tbody>' +
        '<tr><td>Q proj</td><td>[' + din + ', ' + dout + ']</td><td>[' + n + ', ' + dout + ']</td><td>' + heads + '</td></tr>' +
        '<tr><td>K proj</td><td>[' + din + ', ' + kde + ']</td><td>[' + n + ', ' + kde + ']</td><td>' + MODEL.kvHeads + '</td></tr>' +
        '<tr><td>V proj</td><td>[' + din + ', ' + kde + ']</td><td>[' + n + ', ' + kde + ']</td><td>' + MODEL.kvHeads + '</td></tr>' +
        '</tbody></table>' +
        '<p class="ldetail-p">三者共用同一个输入 $x_{norm}$，但各有独立权重矩阵，<b>互不依赖</b>（页面里三条竖列由同一处分叉）。</p>' +
      '</div>' +
    '</div>';
  }

  // MHA / GQA / MQA 对照表：三者只差 K/V 的头数（KV cache 大小正比于它）
  function kvHeadsTableHTML() {
    var nl = MODEL.layers, ctx = KV_DEMO_CTX, hd = MODEL.headDim, nh = MODEL.heads, cur = MODEL.kvHeads;
    var gb = function (kv) { return (2 * nl * ctx * kv * hd * 2 / 1e9).toFixed(2); };
    var tag = function (kv) { return kv === cur ? ' <b>（本模型）</b>' : ''; };
    var row = function (name, kv, note) {
      return '<tr><td>' + name + tag(kv) + '</td><td>' + kv + '</td><td>' + (kv * hd) + '</td>' +
             '<td>约 ' + gb(kv) + ' GB</td><td>' + (kv === 1 ? '1/' + nh : (kv === nh ? '1x' : '1/' + (nh / kv))) + '</td></tr>' +
             (note ? '<tr><td colspan="5" class="kv-note">' + note + '</td></tr>' : '');
    };
    return '<table class="ldetail-table"><thead><tr><th>方案</th><th>n_kv_heads</th><th>K/V 维度</th><th>KV cache（估算）</th><th>相对 MHA</th></tr></thead><tbody>' +
      row('MHA 多头注意力', nh) +
      row('GQA 分组查询', cur, '每 ' + (nh / cur) + ' 个 Q 头共享 1 组 K/V') +
      row('MQA 多查询注意力', 1, '所有 ' + nh + ' 个 Q 头共享同一份 K/V') +
      '</tbody></table>' +
      '<p class="ldetail-p">本模型 <code>' + MODEL.arch + '.attention.head_count_kv</code> = ' + cur +
        '，介于 1 与 ' + nh + ' 之间，属于 <b>GQA</b>（每 ' + (nh / cur) + ' 个 Q 头共享 1 组 K/V）。</p>' +
      '<p class="ldetail-p">判定方法（llama.cpp 只用 head_count_kv 一个字段）：' +
        '<code>kv == head_count</code> 为 MHA、<code>kv == 1</code> 为 MQA、其余为 GQA。' +
        '页面上 Q 输出 ' + (nh * hd) + ' 维而 K/V 只 ' + (cur * hd) + ' 维，比值 ' + (nh / cur) + ' 就是每组共享的头数。</p>';
  }

  // K proj：x_norm 到 K 的线性投影（GQA 下头数少于 Q）
  function detailKProj(layer) {
    var n = tokenCount();
    var din = MODEL.hidden, nh = MODEL.heads, hd = MODEL.headDim, kv = MODEL.kvHeads;
    var dq = nh * hd, dk = kv * hd;
    var src = (layer != null && layer >= 0) ? '<span class="ldetail-src">来自 blk.' + layer + '.</span> ' : '';
    return '<div class="ldetail" id="layer-detail">' +
      '<div class="ldetail-head">' +
        '<span>' + src + 'K proj · 键投影（Key projection）</span>' +
        '<span class="ldetail-hint">再点一次算子名可收起</span>' +
      '</div>' +
      '<div class="ldetail-body">' +
        '<p class="ldetail-p"><b>作用</b>：把 x_norm 投影成 Key —— 每个 token 的“索引标签”，后续用 Q 与它做点积打分（“我有什么可供匹配”）。</p>' +
        '<p class="ldetail-p"><b>输入 / 输出</b>：$x_{norm} \\in \\mathbb{R}^{n \\times ' + din + '}$，$W_k \\in \\mathbb{R}^{' + din + ' \\times ' + dk + '}$，输出 $K \\in \\mathbb{R}^{n \\times ' + dk + '}$。</p>' +

        '<h4>① 矩阵乘法</h4>' +
        '$$ K = x_{norm} \\cdot W_k $$' +
        '$$ K_{t,j} = \\sum_{k=1}^{' + din + '} x_{norm}[t,k] \\cdot W_k[k,j] $$' +

        '<h4>② 形状流转</h4>' +
        '<p class="ldetail-p"><code>[' + n + ', ' + din + '] × [' + din + ', ' + dk + '] -> [' + n + ', ' + dk + ']</code></p>' +
        '<p class="ldetail-p">同样是每个 token 独立做一次投影，token 维 $n$ 不变。</p>' +

        '<h4>③ 为什么 K 是 ' + dk + '，而 Q 是 ' + dq + '？</h4>' +
        '<p class="ldetail-p">因为 K/V 的头数可以和 Q 不同：$d^K_{out} = n_{kv} \\times d_{head} = ' + kv + ' \\times ' + hd + ' = ' + dk + '$，' +
          '而 Q 是 $' + nh + ' \\times ' + hd + ' = ' + dq + '$。这就是下面要讲的 MHA / GQA / MQA。</p>' +

        '<h4>④ 三种头数配置：MHA / GQA / MQA</h4>' +
        '<p class="ldetail-p">三者<b>只差 K/V 的头数</b>，Q 始终是 ' + nh + ' 头。KV cache 大小正比于 K/V 头数，因此这是“质量 vs 显存”的旋钮：</p>' +
        kvHeadsTableHTML() +
        '<p class="ldetail-p">KV cache 公式：$2 \\times n_{layer} \\times n_{ctx} \\times n_{kv} \\times d_{head} \\times \\mathrm{sizeof(dtype)}$' +
          '（K、V 各一份）。这里按 ' + MODEL.layers + ' 层、n_ctx=' + KV_DEMO_CTX + '、fp16 估算，为<b>计算值</b>。</p>' +
        '<p class="ldetail-p">出处：MQA 见 Shazeer 2019, arXiv:1911.02150；GQA 见 Ainslie et al. 2023, arXiv:2305.13245。</p>' +

        '<h4>⑤ 拆成多头 + RoPE</h4>' +
        '<p class="ldetail-p"><code>[' + n + ', ' + dk + '] -> [' + n + ', ' + kv + ', ' + hd + ']</code>，然后 K 也要<b>做 RoPE 旋转</b>（V 不做）。</p>' +

        '<h4>⑥ 写入 KV cache</h4>' +
        '<p class="ldetail-p">K 算完不丢弃：与 V 一起按位置存进 KV cache，后续每生成一个新 token 都复用历史 K/V，避免重算。</p>' +

        '<h4>⑦ 在 llama.cpp 里的实现</h4>' +
        '<p class="ldetail-p"><code>Kcur = ggml_mul_mat(ctx0, layer.wk, cur);</code><br>' +
          '<code>Kcur = ggml_reshape_3d(ctx0, Kcur, ' + hd + ', ' + kv + ', n_tokens);</code><br>' +
          '写入：<code>kv_self.k = ggml_cpy(...)</code>（或 view，取决于后端）</p>' +

        '<h4>⑧ 参数量</h4>' +
        '<p class="ldetail-p">$' + din + ' \\times ' + dk + ' = ' + (din * dk).toLocaleString() + '$ 个权重（无 bias）。' +
          '对比 Q proj 的 ' + (din * dq).toLocaleString() + '，因为 K 头数只有 Q 的 1/' + (nh / kv) + '。</p>' +
      '</div>' +
    '</div>';
  }

  // V proj：x_norm 到 V 的线性投影（与 K 同头数，但不做 RoPE）
  function detailVProj(layer) {
    var n = tokenCount();
    var din = MODEL.hidden, nh = MODEL.heads, hd = MODEL.headDim, kv = MODEL.kvHeads;
    var dv = kv * hd;
    var src = (layer != null && layer >= 0) ? '<span class="ldetail-src">来自 blk.' + layer + '.</span> ' : '';
    return '<div class="ldetail" id="layer-detail">' +
      '<div class="ldetail-head">' +
        '<span>' + src + 'V proj · 值投影（Value projection）</span>' +
        '<span class="ldetail-hint">再点一次算子名可收起</span>' +
      '</div>' +
      '<div class="ldetail-body">' +
        '<p class="ldetail-p"><b>作用</b>：把 x_norm 投影成 Value —— 每个 token 实际携带的“内容”，最终按注意力权重加权求和（“我提供什么”）。</p>' +
        '<p class="ldetail-p"><b>输入 / 输出</b>：$x_{norm} \\in \\mathbb{R}^{n \\times ' + din + '}$，$W_v \\in \\mathbb{R}^{' + din + ' \\times ' + dv + '}$，输出 $V \\in \\mathbb{R}^{n \\times ' + dv + '}$。</p>' +

        '<h4>① 矩阵乘法</h4>' +
        '$$ V = x_{norm} \\cdot W_v $$' +
        '$$ V_{t,j} = \\sum_{k=1}^{' + din + '} x_{norm}[t,k] \\cdot W_v[k,j] $$' +

        '<h4>② 形状流转</h4>' +
        '<p class="ldetail-p"><code>[' + n + ', ' + din + '] × [' + din + ', ' + dv + '] -> [' + n + ', ' + dv + ']</code></p>' +

        '<h4>③ 输出维度与 K 相同</h4>' +
        '<p class="ldetail-p">$d^V_{out} = n_{kv} \\times d_{head} = ' + kv + ' \\times ' + hd + ' = ' + dv + '$，与 K 一致（V 与 K 成对）；' +
          '而 Q 是 ' + nh + ' 头、' + (nh * hd) + ' 维。</p>' +

        '<h4>④ 三种头数配置：MHA / GQA / MQA</h4>' +
        '<p class="ldetail-p">V 与 K 使用<b>同一套头数配置</b>（它们总是一起被缓存，所以头数必须一致）。Q 固定 ' + nh + ' 头：</p>' +
        kvHeadsTableHTML() +
        '<p class="ldetail-p">MQA 见 Shazeer 2019, arXiv:1911.02150；GQA 见 Ainslie et al. 2023, arXiv:2305.13245。</p>' +

        '<h4>⑤ V 与 K 的关键区别：不做 RoPE</h4>' +
        '<p class="ldetail-p">RoPE 是有位置含义的旋转，只作用于需要“按位置匹配”的 Q 和 K；' +
          'V 是纯内容，位置信息已经通过注意力权重（来自 Q·K）体现，因此 <b>V 不旋转</b>。页面里 V 列没有 RoPE 框就是这个原因。</p>' +

        '<h4>⑥ 写入 KV cache</h4>' +
        '<p class="ldetail-p">V 与 K 一起按位置存入 KV cache，供后续 token 复用。</p>' +

        '<h4>⑦ 在 llama.cpp 里的实现</h4>' +
        '<p class="ldetail-p"><code>Vcur = ggml_mul_mat(ctx0, layer.wv, cur);</code><br>' +
          '<code>Vcur = ggml_reshape_3d(ctx0, Vcur, ' + hd + ', ' + kv + ', n_tokens);</code><br>' +
          '写入：<code>kv_self.v = ggml_cpy(...)</code></p>' +

        '<h4>⑧ 参数量</h4>' +
        '<p class="ldetail-p">$' + din + ' \\times ' + dv + ' = ' + (din * dv).toLocaleString() + '$ 个权重（无 bias），与 K proj 相同。</p>' +
      '</div>' +
    '</div>';
  }

  // token_id -> embedding 的查表可视化
  function embLookupHTML(info) {
    var toks = (info && info.ok) ? info.toks : [];
    var n = toks.length;
    var show = Math.min(n, 6);
    var VEC = '▮▮▮▮▮'; // 只用几个方块示意"这是一个向量"

    var idsCol = '', arrCol = '', matCol = '';
    for (var i = 0; i < show; i++) {
      idsCol += '<div class="emb-id-row"><span class="emb-id">' + toks[i].id + '</span>' +
                '<span class="emb-piece">' + esc(toks[i].text || '') + '</span></div>';
      arrCol += '<div class="emb-arrow">──────▶</div>';
      matCol += '<div class="emb-row"><span class="emb-row-id">行 ' + toks[i].id + '</span>' +
                '<span class="emb-vec">' + VEC + '</span></div>';
    }
    if (n > show) {
      idsCol += '<div class="emb-more">… 共 ' + n + ' 个 token</div>';
      arrCol += '<div class="emb-more">&nbsp;</div>';
      matCol += '<div class="emb-more">&nbsp;</div>';
    }
    matCol += '<div class="emb-ellip">⋮ 其余 ' + Math.max(0, MODEL.vocab - n) + ' 行未被取用</div>';

    return '<div class="emb-wrap">' +
      '<div class="emb-title">查表（gather）：按 token id 从权重矩阵里取出对应的行</div>' +
      '<div class="emb-flow">' +
        '<div class="emb-col">' + idsCol + '</div>' +
        '<div class="emb-col arr">' + arrCol + '</div>' +
        '<div class="emb-matrix">' + matCol +
          '<div class="emb-shape">[' + MODEL.vocab + ', ' + MODEL.hidden + ']</div>' +
        '</div>' +
      '</div>' +
      '<div class="emb-note">' +
        'embeddings[i] = W[ token_id[i] ]　·　对应 ggml_get_rows（直接取行）<br>' +
        '数学上等价于 one_hot(token_id) × W（' + MODEL.vocab + ' 维 one-hot，只有一位为 1）<br>' +
        '另：源码里还有 inp_embd 入口 —— 多模态时可直接喂入嵌入向量，跳过查表' +
      '</div>' +
    '</div>';
  }

  function tokenNodesHTML(info) {
    var n = info && info.ok ? info.toks.length : 0;
    var ids = (info && info.ok) ? info.toks.map(function (t) { return t.id; }) : [];
    var preview = INPUT_TEXT.length > 24 ? INPUT_TEXT.slice(0, 24) + '…' : INPUT_TEXT;
    var idsStr;
    if (!ids.length) idsStr = '[]';
    else if (ids.length > 8) idsStr = '[' + ids.slice(0, 8).join(', ') + ', …]';
    else idsStr = '[' + ids.join(', ') + ']';

    // 图一：inp_tokens 数据流（4 个站点，左侧用曲线串联）
    var flow =
      '<div class="flow">' +
        '<div class="flow-station"><div class="rowline">' + node('"' + preview + '"', 'text', 'input') + '</div></div>' +
        '<div class="flow-station"><div class="flow-ids">' + node(idsStr, 'token ids', 'input') +
          '<span class="flow-meta">n_tokens = ' + n + '</span></div></div>' +
        '<div class="flow-station"><div class="inp-box">' +
          '<div class="inp-box-title">inp_tokens</div>' +
          '<div class="inp-box-row"><span>类型</span><span>I32（32 位整数）</span></div>' +
          '<div class="inp-box-row"><span>形状</span><span>[' + n + ']</span></div>' +
          '<div class="inp-box-row"><span>内容</span><span>本批 token id 序列</span></div>' +
          '<div class="inp-box-row"><span>角色</span><span>计算图的输入节点（ggml_set_input）</span></div>' +
        '</div></div>' +
        '<div class="flow-station">' + embLookupHTML(info) + '</div>' +
        '<div class="flow-station"><div class="flow-ids">' + node('embeddings', 'F32 · [' + n + ', ' + MODEL.hidden + ']', 'attn') +
          '<span class="flow-meta">每个 token 变一个向量</span></div></div>' +
      '</div>';

    // 图二：代码实现
    var code =
      '<div class="code-wrap">' +
        '<div class="code-head">' +
          '<span>inp_tokens 的创建 · <code>src/llama-graph.cpp</code></span>' +
          '<a class="code-link" href="https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp#L2316" target="_blank" rel="noreferrer">查看源码 ↗</a>' +
        '</div>' +
        '<pre class="code-body">' +
'<span class="c-fn">llm_graph_context::build_inp_embd()</span>\n' +
'\n' +
'<span class="c-num">1</span>  inp->tokens = <span class="c-fn">ggml_new_tensor_1d</span>(ctx0, GGML_TYPE_I32, ubatch.n_tokens);\n' +
'   <span class="c-cm">// 建一维 I32 张量，长度 = 本批 token 数（当前 ubatch.n_tokens = ' + n + '）</span>\n' +
'\n' +
'<span class="c-num">2</span>  <span class="c-fn">cb</span>(inp->tokens, "inp_tokens", -1);\n' +
'   <span class="c-cm">// 把这枚张量命名为 "inp_tokens"</span>\n' +
'\n' +
'<span class="c-num">3</span>  <span class="c-fn">ggml_set_input</span>(inp->tokens);\n' +
'   <span class="c-cm">// 标记为计算图输入：数据不来自图内计算，而由 set_inputs 从外部填入</span>\n' +
'\n' +
'<span class="c-num">4</span>  res->t_inp_tokens = inp->tokens;\n' +
'   <span class="c-cm">// 保存引用，供 token_embd 查表时取用</span>' +
        '</pre>' +
      '</div>';

    return flow + code;
  }

  // 在数据流图左侧绘制曲线，串联各站点，并在曲线上标注每段的转换说明
  function drawFlowCurve(root) {
    if (!root || typeof root.querySelector !== 'function') return;
    var flow = root.querySelector('.flow');
    if (!flow || typeof flow.getBoundingClientRect !== 'function') return;
    if (typeof document.createElementNS !== 'function' || typeof document.createElement !== 'function') return;

    // 清理旧图层
    var olds = flow.querySelectorAll ? flow.querySelectorAll('.flow-svg, .flow-label') : [];
    for (var d = 0; d < olds.length; d++) {
      if (olds[d].parentNode) olds[d].parentNode.removeChild(olds[d]);
    }
    var stations = flow.querySelectorAll ? flow.querySelectorAll('.flow-station') : [];
    if (!stations || stations.length < 2) return;

    var fb = flow.getBoundingClientRect();
    if (!fb || !fb.height || !fb.width) return; // 未布局（例如无头测试环境）

    var pts = [];
    for (var i = 0; i < stations.length; i++) {
      var r = stations[i].getBoundingClientRect();
      pts.push({ x: r.left - fb.left, y: r.top - fb.top + r.height / 2 });
    }
    var trackX = 44; // 曲线所在 x（左侧轨道区）

    var NS = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('class', 'flow-svg');
    svg.setAttribute('width', String(fb.width));
    svg.setAttribute('height', String(fb.height));

    // 每段（相邻两站之间）各画一条曲线，并各自带圆点（起点）与箭头（终点）
    for (var k = 0; k < pts.length - 1; k++) {
      var a = pts[k], b = pts[k + 1];
      var x0 = a.x, y0 = a.y + 6;  // 本段起点：圆点，贴本站左边缘、偏下
      var x1 = b.x, y1 = b.y - 6;  // 本段终点：箭头，贴下一站左边缘、偏上

      var path = document.createElementNS(NS, 'path');
      path.setAttribute('class', 'flow-curve');
      path.setAttribute('d', 'M ' + x0 + ' ' + y0 +
                             ' C ' + trackX + ' ' + y0 + ', ' + trackX + ' ' + y1 + ', ' + x1 + ' ' + y1);
      svg.appendChild(path);

      var dot = document.createElementNS(NS, 'circle');
      dot.setAttribute('class', 'flow-dot');
      dot.setAttribute('cx', String(x0));
      dot.setAttribute('cy', String(y0));
      dot.setAttribute('r', '3.5');
      svg.appendChild(dot);

      var head = document.createElementNS(NS, 'path');
      head.setAttribute('class', 'flow-head');
      head.setAttribute('d', 'M ' + (x1 - 9) + ' ' + (y1 - 4) + ' L ' + (x1 + 1) + ' ' + y1 +
                              ' L ' + (x1 - 9) + ' ' + (y1 + 4) + ' z');
      svg.appendChild(head);
    }

    flow.appendChild(svg);

    // 曲线上的标签（每段一个）
    var labels = ['BPE 分词', '写入图输入', 'token_embd 查表'];
    for (var t = 0; t < labels.length && t < pts.length - 1; t++) {
      var midY = (pts[t].y + pts[t + 1].y) / 2;
      var lab = document.createElement('div');
      lab.className = 'flow-label';
      lab.style.top = midY + 'px';
      lab.textContent = labels[t];
      flow.appendChild(lab);
    }
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
  function layerBody(i) {
    var n = tokenCount();
    var H = MODEL.hidden;
    var qd = MODEL.heads * MODEL.headDim;    // Q 投影输出维度
    var kvd = MODEL.kvHeads * MODEL.headDim; // K/V 投影输出维度
    var F = MODEL.ffn;
    // 权重矩阵形状：优先取所选模型第 i 层的真实张量形状，未加载模型时用派生值兜底
    var wDims = function (tname, fallback) {
      var ts = MODEL.tensors;
      if (ts && ts.length) {
        var full = 'blk.' + i + '.' + tname;
        for (var k = 0; k < ts.length; k++) {
          if (ts[k].name === full && ts[k].dims && ts[k].dims.length) return ts[k].dims.slice();
        }
      }
      return fallback;
    };

    // 行单元：{name, op, dim, cls} 普通行 | {fork:true} 分叉行 | {tri:[...]} 三列并发行
    var rows = [
      { name: '输入 embeddings', op: '来自上一层输出（第 0 层来自 token 查表）', dim: '[' + n + ', ' + H + ']', cls: 'input' },
      { name: 'attn_norm', op: 'RMSNorm', dim: 'x_norm [' + n + ', ' + H + ']', cls: 'norm' },
      { fork: true },
      { tri: [
        { blocks: [
            { name: 'Q proj', op: 'x_norm · Wq', w: wDims('attn_q.weight', [H, qd]), dim: 'Q [' + n + ', ' + qd + ']', cls: 'attn', detail: 'Q proj' },
            { name: 'RoPE', op: '旋转 Q（前 ' + MODEL.headDim + ' 维）', dim: 'Q [' + n + ', ' + MODEL.heads + ', ' + MODEL.headDim + ']', cls: 'attn' }
          ], tag: '' },
        { blocks: [
            { name: 'K proj', op: 'x_norm · Wk', w: wDims('attn_k.weight', [H, kvd]), dim: 'K [' + n + ', ' + kvd + ']', cls: 'attn', detail: 'K proj' },
            { name: 'RoPE', op: '旋转 K（前 ' + MODEL.headDim + ' 维）', dim: 'K [' + n + ', ' + MODEL.kvHeads + ', ' + MODEL.headDim + ']', cls: 'attn' }
          ], tag: '写入 KV cache' },
        { blocks: [
            { name: 'V proj', op: 'x_norm · Wv', w: wDims('attn_v.weight', [H, kvd]), dim: 'V [' + n + ', ' + kvd + ']', cls: 'attn', detail: 'V proj' }
          ], tag: '写入 KV cache' }
      ] },
      { merge: true },
      { name: 'attention', op: 'Q·Kᵀ → softmax → ·V（' + MODEL.heads + ' 个 Q 头共享 ' + MODEL.kvHeads + ' 个 KV 头）', dim: 'attn_out [' + n + ', ' + qd + ']', cls: 'attn' },
      { name: 'Wo', op: 'MUL_MAT ' + qd + ' → ' + H, dim: 'x_attn [' + n + ', ' + H + ']', cls: 'attn' },
      { name: '⊕ 残差 ①', op: 'x = x_attn + x', dim: 'x [' + n + ', ' + H + ']', cls: 'resid', resid: 1 },
      { name: 'ffn_norm', op: 'RMSNorm', dim: 'x_norm [' + n + ', ' + H + ']', cls: 'norm' },
      { name: 'gate proj', op: 'MUL_MAT ' + H + ' → ' + F, dim: 'gate [' + n + ', ' + F + ']', cls: 'mlp' },
      { name: 'up proj', op: 'MUL_MAT ' + H + ' → ' + F, dim: 'up [' + n + ', ' + F + ']', cls: 'mlp' },
      { name: 'SiLU（SwiGLU）', op: 'gate ⊙ up', dim: '[' + n + ', ' + F + ']', cls: 'mlp' },
      { name: 'down proj', op: 'MUL_MAT ' + F + ' → ' + H, dim: 'x_ffn [' + n + ', ' + H + ']', cls: 'mlp' },
      { name: '⊕ 残差 ②', op: 'x = x_ffn + x', dim: 'x [' + n + ', ' + H + ']', cls: 'resid', resid: 2 },
      { name: '输出', op: '→ 下一层', dim: '[' + n + ', ' + H + ']', cls: 'input' }
    ];

    var H_STEP = 30, H_FORK = 72, H_TRI = 210, H_CONN = 18;
    var y = 0, y0 = 0, yRes1 = 0, yRes2 = 0;

    var html = '<div class="lflow">';
    rows.forEach(function (r, idx) {
      var next = rows[idx + 1];
      // 仅相邻"普通行"之间加圆点/箭头连线（分叉行、三列行、汇总行自带结构）
      var needConn = next && !r.fork && !r.merge && !next.fork && !next.tri && !next.merge;

      var h;
      if (r.fork) {
        h = H_FORK;
        html += '<div class="lstep-fork">' + forkSVG() + '</div>';
      } else if (r.merge) {
        h = H_FORK;
        html += '<div class="lstep-fork">' + mergeSVG() + '</div>';
      } else if (r.tri) {
        h = H_TRI;
        html += '<div class="lstep-tri">' +
          '<div class="ltri-cols">' + r.tri.map(function (col) {
            var inner = '';
            col.blocks.forEach(function (b, bi) {
              if (bi > 0) inner += '<div class="ltri-conn"><i></i></div>';
              inner += '<div class="ltri-box' + (b.detail ? ' clickable' : '') + '"' +
                         (b.detail ? ' data-detail="' + esc(b.detail) + '" data-layer="' + i + '" title="点击查看数学过程"' : '') + '>' +
                         '<span class="ltri-name ' + b.cls + '">' + esc(b.name) +
                           (b.w ? '<span class="ltri-wdim">[' + b.w.join(', ') + ']</span>' : '') + '</span>' +
                         '<span class="ltri-op">' + esc(b.op) + '</span>' +
                         '<span class="ltri-dim">' + esc(b.dim) + '</span>' +
                       '</div>';
            });
            if (col.tag) inner += '<span class="ltri-tag">' + esc(col.tag) + '</span>';
            return '<div class="ltri-col">' + inner + '</div>';
          }).join('') + '</div>' +
        '</div>';
      } else {
        h = H_STEP;
        var clickable = (r.name === 'attn_norm');
        html += '<div class="lstep">' +
          '<span class="lstep-node ' + r.cls + (clickable ? ' clickable' : '') + '"' +
            (clickable ? ' data-detail="attn_norm" data-layer="' + i + '" title="点击查看数学过程"' : '') + '>' +
            esc(r.name) + '</span>' +
          '<span class="lstep-op">' + esc(r.op) + '</span>' +
          '<span class="lstep-dim">' + esc(r.dim) + '</span>' +
        '</div>';
      }

      if (idx === 0)                     y0    = y + h / 2;
      if (r.resid === 1)                 yRes1 = y + h / 2;
      if (r.resid === 2)                 yRes2 = y + h / 2;

      y += h;

      if (needConn) {
        html += '<div class="lconn"><i></i></div>';
        y += H_CONN;
      }
    });
    // 残差旁路（CSS 折线）
    html += '<div class="lpath" style="top:' + y0 + 'px;height:' + (yRes1 - y0) + 'px"></div>';
    html += '<div class="lpath" style="top:' + yRes1 + 'px;height:' + (yRes2 - yRes1) + 'px"></div>';
    html += '</div>';

    return html;
  }

  // 三路并发分叉：直角；主干对准中间列，三支对准三列中心（86 / 272 / 458）
  // 坐标系与三列同宽（172*3 + 14*2 = 544），1:1 映射，可精确对齐
  function forkSVG() {
    return '<svg class="lfork-svg" viewBox="0 0 544 72" aria-hidden="true">' +
      '<circle class="lfork-dot" cx="272" cy="4" r="3.5"></circle>' +
      '<path class="lfork-line" d="M 272 7 L 272 16 M 86 16 L 458 16' +
        ' M 86 16 L 86 62 M 272 16 L 272 62 M 458 16 L 458 62"></path>' +
      '<path class="lfork-head" d="M 82 61 L 86 71 L 90 61 z"></path>' +
      '<path class="lfork-head" d="M 268 61 L 272 71 L 276 61 z"></path>' +
      '<path class="lfork-head" d="M 454 61 L 458 71 L 462 61 z"></path>' +
    '</svg>';
  }

  // 三列汇合：三条竖线（对准三列中心）汇到一条横线，再由中部引出主干到 attention
  function mergeSVG() {
    return '<svg class="lfork-svg" viewBox="0 0 544 72" aria-hidden="true">' +
      '<circle class="lfork-dot" cx="86" cy="3" r="3.5"></circle>' +
      '<circle class="lfork-dot" cx="272" cy="3" r="3.5"></circle>' +
      '<circle class="lfork-dot" cx="458" cy="3" r="3.5"></circle>' +
      '<path class="lfork-line" d="M 86 6 L 86 36 M 272 6 L 272 36 M 458 6 L 458 36' +
        ' M 86 36 L 458 36 M 272 36 L 272 64"></path>' +
      '<path class="lfork-head" d="M 268 63 L 272 71 L 276 63 z"></path>' +
    '</svg>';
  }

  // 整组层块末尾的算子详情（若已展开）
  function layerDetailFor() {
    if (!LAYER_DETAIL) return '';
    return layerDetailHTML(LAYER_DETAIL.op, LAYER_DETAIL.layer);
  }

  function layerBlock(i, open) {
    return '<div class="layer-block' + (open ? ' open' : '') + '">' +
      '<div class="layer-head">' +
        '<span class="idx">blk.' + i + '.</span>' +
        '<span class="name">Transformer 层</span>' +
        '<span class="meta">attn + ffn</span>' +
      '</div>' + layerBody(i) + '</div>';
  }

  var GRAPH_STEPS = [
    {
      title: '准备输入张量',
      desc: '推理从 token 序列开始。文本先被分词（tokenize）成 token id，再通过 token_embd 查表得到每个 token 的向量表示（嵌入）。下方输入框可输入任意文本，实时查看当前模型真实的分词结果。',
      concepts: function () {
        return [
          ['tokenize', '输入文本 -> token id 列表（byte-level BPE）'],
          ['token', '词表里的一个子词片段（可能是整词、部分汉字或字节）'],
          ['embeddings', 'token_embd.weight 是 [vocab, hidden] 的大表，查表即取行'],
          ['batch', '一次送入的 token 序列，形状 [n_tokens, n_embd]']
        ];
      },
      logs: function () {
        var info = tokenizeInfo();
        var n = info.ok ? info.toks.length : 0;
        return [
          { cls: 'l-graph', text: 'graph: build inp_tokens   = [' + n + ', 1] (I32)' },
          { cls: 'l-graph', text: 'graph: build inp_pos      = [' + n + ', 1] (I32)   positions = [0..' + Math.max(0, n - 1) + ']' },
          { cls: 'l-graph', text: 'graph: build token_embd   = MUL_MAT   [' + MODEL.vocab + ',' + MODEL.hidden + '] x [' + n + ',1]' }
        ];
      },
      render: function (box) {
        var info = tokenizeInfo();
        box.innerHTML =
          '<div class="output-box" style="display:flex;align-items:center;gap:8px">' +
            '<span class="prompt">输入文本：</span>' +
            '<input id="in-text" class="text-input" type="text" placeholder="输入任意文本…">' +
          '</div>' +
          '<div id="tok-stat">' + tokenStatHTML(info) + '</div>' +
          '<div id="tok-table">' + (info.ok ? tokenTableHTML(info.toks) : '') + '</div>' +
          '<div id="tok-nodes">' + tokenNodesHTML(info) + '</div>';
        drawFlowCurve(box);
        var inp = box.querySelector('#in-text');
        if (inp) {
          inp.value = INPUT_TEXT;
          inp.addEventListener('input', function () {
            INPUT_TEXT = inp.value;
            var inf = tokenizeInfo();
            var st = box.querySelector('#tok-stat');
            var tb = box.querySelector('#tok-table');
            var nd = box.querySelector('#tok-nodes');
            if (st) st.innerHTML = tokenStatHTML(inf);
            if (tb) tb.innerHTML = inf.ok ? tokenTableHTML(inf.toks) : '';
            if (nd) nd.innerHTML = tokenNodesHTML(inf);
            drawFlowCurve(box);
          });
        }
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
          layerBody(0) + '</div>' +
          '<p class="step-desc" style="margin-top:14px">上方为注意力子层，下方为前馈子层；两条 <code>+ residual</code> 为残差连接。</p>' +
          layerDetailFor();
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
        box.innerHTML = html + layerDetailFor();
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
        ['logits', '形状 [n_tokens, vocab]，每个位置对全部 token 的分数'],
        ['权重共享', 'lm_head 常与 token_embd 共享权重（tie）'],
        ['下一步', 'logits 交给采样器，选出一个 token 作为输出']
      ],
      logs: function () {
        var n = tokenCount();
        return [
          { cls: 'l-graph', text: 'graph: build output_norm   = RMS_NORM' },
          { cls: 'l-graph', text: 'graph: build result_output = MUL_MAT  [' + MODEL.hidden + '] x [' + MODEL.vocab + ',' + MODEL.hidden + ']' },
          { cls: 'l-graph', text: 'graph: build -> logits     = [' + n + ', ' + MODEL.vocab + '] (F32)' }
        ];
      },
      render: function (box) {
        var n = tokenCount();
        box.innerHTML =
          row([
            node('l_out (layer ' + (MODEL.layers - 1) + ')', '[' + n + ', ' + MODEL.hidden + ']', 'attn'),
            node('output_norm', 'RMSNorm', 'norm'),
            node('lm_head', 'MUL_MAT', 'weight'),
            node('logits', '[' + n + ', ' + MODEL.vocab + ']', 'out')
          ]) +
          '<p class="step-desc" style="margin-top:16px">注意 logits 是 ' + n + ' 个位置各自的分数；生成时只关心最后一个位置。</p>' +
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
      logs: function () {
        var n = tokenCount();
        return [
          { cls: 'l-comp', text: 'llama_decode: batch ' + n + ' tokens (1 sequence)' },
          { cls: 'l-comp', text: 'balloc: split -> 1 ubatch of ' + n + ' tokens' }
        ];
      },
      render: function (box) {
        var n = tokenCount();
        box.innerHTML =
          '<div class="rowline">' + node('llama_decode(batch)', 'API', 'input') + arrow() +
          node('balloc->init', 'planner', 'weight') + arrow() +
          node('ubatch[0]', n + ' tokens', 'attn') + '</div>' +
          '<div class="concept-card" style="margin-top:16px">prompt 处理（prefill）阶段：' + n + ' 个 token 可一次并行算完。</div>';
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
      logs: function () {
        var n = tokenCount();
        return [
          { cls: 'l-comp', text: 'llama_kv_cache: init_batch: n_tokens = ' + n + ', n_ctx = ' + KV_DEMO_CTX },
          { cls: 'l-comp', text: 'llama_kv_cache: find_slot: assigned positions [0..' + Math.max(0, n - 1) + ']' }
        ];
      },
      render: function (box) {
        var n = Math.min(tokenCount(), 48);
        var cells = '';
        for (var i = 0; i < 48; i++) {
          cells += '<div class="kv-cell' + (i < n ? ' filled' : '') + '"></div>';
        }
        var posNodes = [];
        for (var k = 0; k < Math.min(n, 6); k++) posNodes.push(node('pos ' + k, 'token ' + k, 'input'));
        box.innerHTML =
          '<div style="font-family:var(--mono);font-size:11.5px;color:var(--text-dim);margin-bottom:8px">KV cache 位置（前 48 个示意）</div>' +
          '<div class="kv-grid">' + cells + '</div>' +
          '<div class="rowline" style="margin-top:14px">' + posNodes.join(arrow()) + '</div>' +
          (n > 6 ? '<p class="step-desc">（仅列出前 6 个位置，共 ' + n + ' 个）</p>' : '');
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
      logs: function () {
        var n = tokenCount();
        return [
          { cls: 'l-comp', text: 'process_ubatch: graph_params (n_tokens=' + n + ', n_seqs=1)' },
          { cls: 'l-comp', text: 'process_ubatch: can_reuse = false (first run) -> build_graph' },
          { cls: 'l-comp', text: 'process_ubatch: set_inputs -> inp_tokens, inp_pos, KQ_mask' }
        ];
      },
      render: function (box) {
        var n = tokenCount();
        box.innerHTML =
          '<div class="rowline">' +
            node('ubatch', n + ' tokens', 'attn') + arrow() +
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
      logs: function () {
        var n = tokenCount();
        return [
          { cls: 'l-comp', text: 'compute_splits: split 0 (CPU)  graph_compute' },
          { cls: 'l-time', text: 'compute_splits: split 0 done (embd)' },
          { cls: 'l-comp', text: 'compute_splits: split 1 (GPU)  graph_compute' },
          { cls: 'l-time', text: 'compute_splits: split 1 done (20 layers)' },
          { cls: 'l-comp', text: 'compute_splits: split 2 (CPU)  graph_compute' },
          { cls: 'l-info', text: 'llama_decode: copy logits to host (' + n + ' x ' + MODEL.vocab + ')' },
          { cls: 'l-time', text: 'llama_perf: eval time = 128.40 ms / ' + n + ' tokens' }
        ];
      },
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
      logs: function () {
        var n = tokenCount();
        return [
          { cls: 'l-samp', text: 'llama_get_logits_ith: idx = -1 (最后一个位置)' },
          { cls: 'l-samp', text: 'logits shape = [' + n + ', ' + MODEL.vocab + '], dtype = F32' }
        ];
      },
      render: function (box) {
        var n = tokenCount();
        box.innerHTML =
          row([
            node('logits', '[' + n + ', ' + MODEL.vocab + ']', 'weight'),
            node('取最后一行', 'idx=-1', 'norm'),
            node('分数向量', '[' + MODEL.vocab + ']', 'input')
          ]) +
          '<div class="concept-card" style="margin-top:16px">' + MODEL.vocab + ' 个候选，每个都有一个分数；分数越大表示模型越"倾向"该 token。</div>';
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
            '<span class="prompt">' + esc(INPUT_TEXT) + '</span><span class="gen">我可以帮你回答各种问题</span>' +
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
      // 算子点击（事件委托：只有 .clickable 会响应）
      var vizEl = document.getElementById('viz');
      if (vizEl && vizEl.addEventListener) {
        vizEl.addEventListener('click', function (e) {
          var t = e.target;
          while (t && t !== vizEl) {
            var cls = t.className ? String(t.className) : '';
            if (cls.indexOf('clickable') !== -1) {
              var op = t.getAttribute ? t.getAttribute('data-detail') : null;
              var lyr = t.getAttribute ? parseInt(t.getAttribute('data-layer'), 10) : NaN;
              if (op) {
                if (LAYER_DETAIL && LAYER_DETAIL.op === op && LAYER_DETAIL.layer === lyr) {
                  LAYER_DETAIL = null; // 再点一次收起
                } else {
                  LAYER_DETAIL = { layer: lyr, op: op };
                }
                self.render();
              }
              return;
            }
            t = t.parentNode;
          }
        });
      }
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
          var info = window.parseGGUF(fr.result, {
            fullKeys: { 'tokenizer.ggml.tokens': 1, 'tokenizer.ggml.merges': 1 }
          });
          setModelFromGGUF(info, file);
          buildTokenizer(info);
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

      // 若有算子详情面板展开，排版其中的 LaTeX
      var detEl = document.getElementById('layer-detail');
      if (detEl) typesetMath(detEl);
    }
  };

  document.addEventListener('DOMContentLoaded', function () { App.init(); });
  window.__App = App;
})();
