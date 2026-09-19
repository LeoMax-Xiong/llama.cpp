// Qwen3-0.6B 计算流程可视化
// 纯静态演示：用一个自包含的状态机，逐步展示
//   1) 模型加载  2) 构建计算图  3) 推理计算  4) 采样与解码
// 参数取自 Qwen3-0.6B 的 config.json / GGUF metadata，其它数值为示意数据。
(function () {
  'use strict';

  // ---------------------------------------------------------------- 模型参数
  var MODEL = {
    name: 'Qwen3-0.6B',
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

  // KV cache 大小：n_ctx * n_layer * n_kv_heads * head_dim * 2(K,V) * 2字节(f16)
  var KV_DEMO_CTX = 4096;
  var kvBytes = KV_DEMO_CTX * MODEL.layers * MODEL.kvHeads * MODEL.headDim * 2 * 2;
  var kvMiB = Math.round(kvBytes / 1024 / 1024);

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
      logs: [
        { cls: 'l-load', text: 'llama_model_loader: loaded meta data with 28 key-value pairs and 291 tensors' },
        { cls: 'l-load', text: 'llama_model_loader: - kv   0: general.architecture       str              = qwen3' },
        { cls: 'l-info', text: 'llama_model_loader: - kv   1: general.name                 str              = Qwen3-0.6B' },
        { cls: 'l-info', text: 'llama_model_loader: - kv   2: general.file_type            u32              = 1 (F16)' },
        { cls: 'l-info', text: 'llama_model_loader: - kv   3: qwen3.block_count            u32              = 28' },
        { cls: 'l-info', text: 'llama_model_loader: - kv   4: qwen3.embedding_length       u32              = 1024' }
      ],
      render: function (box) {
        box.innerHTML =
          '<div class="file-card">' +
            '<div class="file-icon">GGUF</div>' +
            '<div><div class="fname">qwen3-0.6b.gguf</div>' +
            '<div class="fmeta">1.51 GB  -  header: magic=GGUF, version=3, n_tensors=291, n_kv=28</div></div>' +
          '</div>' +
          '<div style="margin-top:16px" class="rowline">' +
            node('magic', 'u32', 'input') + arrow() +
            node('version', 'u32', 'input') + arrow() +
            node('n_tensors', 'u64 = 291', 'input') + arrow() +
            node('n_kv', 'u64 = 28', 'input') +
          '</div>' +
          '<p class="step-desc" style="margin-top:14px">文件头校验通过，准备读取元数据与张量表。</p>';
      }
    },
    {
      title: '解析元数据（超参数）',
      desc: '从文件头之后读出 28 条 key-value 元数据。这些超参数决定了模型结构，也决定了后面计算图如何构建。',
      concepts: [
        ['架构', '<code>qwen3</code>：仅解码器（decoder-only）Transformer，用 GQA 注意力'],
        ['GQA', '16 个 query 头共享 8 个 key/value 头，KV cache 更小'],
        ['head_dim', '128：每个注意力头的维度，16 x 128 = 2048 为 Q 投影输出']
      ],
      logs: [
        { cls: 'l-load', text: 'print_info: arch                  = qwen3' },
        { cls: 'l-load', text: 'print_info: n_layer               = 28' },
        { cls: 'l-load', text: 'print_info: n_embd                = 1024' },
        { cls: 'l-load', text: 'print_info: n_head                = 16' },
        { cls: 'l-load', text: 'print_info: n_head_kv             = 8' },
        { cls: 'l-load', text: 'print_info: n_embd_head_k         = 128' },
        { cls: 'l-load', text: 'print_info: n_ff                  = 3072' },
        { cls: 'l-load', text: 'print_info: n_vocab               = 151936' },
        { cls: 'l-load', text: 'print_info: rope_freq_base        = 1000000' }
      ],
      render: function (box) {
        var rows = [
          ['general.architecture', 'qwen3'],
          ['general.name', 'Qwen3-0.6B'],
          ['qwen3.context_length', MODEL.ctxLen],
          ['qwen3.embedding_length', MODEL.hidden],
          ['qwen3.block_count', MODEL.layers],
          ['qwen3.feed_forward_length', MODEL.ffn],
          ['qwen3.attention.head_count', MODEL.heads],
          ['qwen3.attention.head_count_kv', MODEL.kvHeads],
          ['qwen3.attention.key_length', MODEL.headDim],
          ['qwen3.rope.freq_base', MODEL.ropeTheta],
          ['qwen3.attention.layer_norm_rms_epsilon', '1e-06'],
          ['tokenizer.ggml.model', 'gpt2 (BPE)'],
          ['tokenizer.ggml.tokens', MODEL.vocab]
        ];
        box.innerHTML = '<table class="kv-table">' + rows.map(function (r) {
          return '<tr><td>' + esc(r[0]) + '</td><td>' + esc(r[1]) + '</td></tr>';
        }).join('') + '</table>';
      }
    },
    {
      title: '创建后端并上传权重',
      desc: '根据 -ngl 参数创建后端（CPU / Metal GPU），把 291 个张量分配到各后端缓冲，并把权重数据从磁盘读入设备内存。',
      concepts: [
        ['后端(backend)', '执行计算的目标设备：CPU、Metal、CUDA、Vulkan 等'],
        ['-ngl 99', '把全部 28 层放到 GPU；-ngl 0 则全部在 CPU'],
        ['权重张量', 'token_embd、每层的 attn_q/k/v/o、mlp_gate/up/down 等']
      ],
      logs: [
        { cls: 'l-load', text: 'llama_model_load: load_all_data: buffer size = 1503.02 MiB' },
        { cls: 'l-load', text: 'llama_model_load: model size    = 1400.00 MiB' },
        { cls: 'l-load', text: 'load_tensors: offloading 28 repeating layers to GPU' },
        { cls: 'l-load', text: 'load_tensors: offloaded 29/29 layers to GPU' },
        { cls: 'l-load', text: 'llama_model_load: Metal model buffer size = 1400.00 MiB' }
      ],
      render: function (box) {
        var tensors = [
          ['token_embd.weight', '[151936, 1024]', 'weight'],
          ['blk.0.attn_norm.weight', '[1024]', 'norm'],
          ['blk.0.attn_q.weight', '[2048, 1024]', 'attn'],
          ['blk.0.attn_k.weight', '[1024, 1024]', 'attn'],
          ['blk.0.attn_v.weight', '[1024, 1024]', 'attn'],
          ['blk.0.ffn_gate.weight', '[3072, 1024]', 'mlp'],
          ['...  (第 1..27 层同构)  ...', '', ''],
          ['output_norm.weight', '[1024]', 'norm'],
          ['output.weight', '(tied, 复用 token_embd)', 'out']
        ];
        box.innerHTML =
          '<div class="rowline" style="margin-bottom:14px">' +
            node('CPU', 'backend', 'input') + arrow() +
            node('Metal / GPU', 'backend', 'attn') +
          '</div>' +
          '<div class="layer-stack">' + tensors.map(function (t) {
            if (!t[1]) return '<div class="layer-fold">' + esc(t[0]) + '</div>';
            return '<div class="rowline">' + node(t[0], '', t[2]) +
                   '<span class="arrow" style="margin-left:auto">' + esc(t[1]) + '</span></div>';
          }).join('') + '</div>';
      }
    },
    {
      title: '创建 llama_context（分配 KV cache）',
      desc: '上下文（llama_context）持有推理所需的运行时状态：KV cache、输出缓冲、后端调度器。其中 KV cache 用来缓存历史 token 的 K/V，避免重复计算。',
      concepts: [
        ['KV cache', '缓存每层每步的 Key/Value 张量，是自回归生成的关键'],
        ['GQA 的好处', '只需 8 个 KV 头而非 16 个，KV cache 减半'],
        ['后端调度器', '<code>ggml_backend_sched</code>：把计算图切分到多个后端执行']
      ],
      logs: [
        { cls: 'l-load', text: 'llama_context: n_ctx         = ' + KV_DEMO_CTX },
        { cls: 'l-load', text: 'llama_context: n_batch       = 2048' },
        { cls: 'l-load', text: 'llama_context: flash_attn    = auto' },
        { cls: 'l-load', text: 'llama_kv_cache:      Metal buffer size = ' + kvMiB + ' MiB' },
        { cls: 'l-load', text: 'sched_reserve: reserving ... graph nodes = 2240' }
      ],
      render: function (box) {
        box.innerHTML =
          '<div class="concept-card" style="font-family:var(--mono)">' +
            'KV = n_ctx(' + KV_DEMO_CTX + ') x n_layer(' + MODEL.layers + ') x n_kv_heads(' + MODEL.kvHeads +
            ') x head_dim(' + MODEL.headDim + ') x 2(K,V) x 2B = <b>' + kvMiB + ' MiB</b>' +
          '</div>' +
          '<div class="mem-bar" style="margin-top:14px">' +
            '<div class="mem-seg w"  style="flex:60">weights ~1400 MiB</div>' +
            '<div class="mem-seg kv" style="flex:20">KV cache ' + kvMiB + ' MiB</div>' +
            '<div class="mem-seg out" style="flex:12">output / compute</div>' +
          '</div>' +
          '<div class="rowline" style="margin-top:16px">' +
            node('llama_model', '权重', 'weight') + arrow() +
            node('llama_context', '运行时状态', 'input') + arrow() +
            node('llama_sampler', '采样器链', 'out') +
          '</div>' +
          '<p class="step-desc" style="margin-top:14px">模型加载完成，可以开始推理。</p>';
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
      window.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowRight') self.next();
        if (e.key === 'ArrowLeft') self.prev();
      });
      this.render();
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
      document.getElementById('steps').innerHTML = stage.steps.map(function (st, i) {
        var cls = i === App.step ? 'active' : (i < App.step ? 'done' : '');
        return '<li class="' + cls + '">' + (i + 1) + '. ' + esc(st.title) + '</li>';
      }).join('');
      document.getElementById('concepts').innerHTML = concepts(step.concepts);

      // 中间舞台
      var head = document.getElementById('stage-head');
      head.innerHTML = '<span class="title">' + esc(stage.title) + '</span>' +
        '<span class="badge">' + esc(stage.id) + ' / step ' + (this.step + 1) + '-' + stage.steps.length + '</span>';
      var viz = document.getElementById('viz');
      viz.innerHTML = '';
      step.render(viz);

      // 右侧日志：累积当前阶段到当前 step 的所有日志
      var logs = [];
      for (var i = 0; i <= this.step; i++) {
        logs = logs.concat(stage.steps[i].logs || []);
      }
      logs.push({ cls: 'l-hl', text: '--- 步骤 ' + (this.step + 1) + ': ' + step.title + ' ---' });
      Log.setStageLogs(logs);

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
