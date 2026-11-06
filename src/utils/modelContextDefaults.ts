/**
 * 每模型默认上下文长度人工策展表（2026-08-19 大王裁定；2026-08-21 按真实
 * GGUF 元数据验算重排——小模型长上下文；v2 2026-08-26 设备感知预算重排——
 * 大王裁定「探索上限」，K90 可用内存 8-9GB 挥霍口径）。
 * v3 2026-11-05：>4GB 大模型不再吃 8-9GB 探索口径，改为 4096 上界 +
 * 加载前 resolveLoadBudget（6GB PSS 硬杀线）实时夹紧。
 *
 * 策展原则：模型越小 → 上下文越长（KV 便宜 + 速度快），上限 = GGUF
 * context_length，预算 = 设备感知 resolvePssSafeBudget（8-9GB 口径，
 * fallback 4GB）。本表每个档位都经 memoryEstimator 公式 (权重+KV+compute)×1.1
 * 验算（见 __tests__/modelContextDefaults.test.ts 门禁，预算按 8.5GB 设备档
 * 断言），验算数据来自入选模型真实 GGUF 头部（qwen35 2B: 24层/2KV头×256 →
 * 48KB/token；qwen35 4B: 32层/4KV头×256 → 128KB/token；mistral3: 26层/
 * 8KV头×128 → 106KB/token；lfm2 无 KV 声明，App 保守按 32 头×64 估 →
 * 240KB/token）。设备内存安全由启动 PSS 审计兜底（preset 可自愈）；
 * 用户手调 = 主权，任何迁移/归一都不碰。
 */
import type {Model} from './types';

/**
 * 策展表版本（v3：2026-11-05 大模型加载失败根治）。preset 源档位随表版本
 * 一次性拉齐（normalizePresetNCtxToCuratedDefaults 升档合法化，解除旧版
 * 只降不升的历史钉死）；user 源主权不碰。
 * v3 变更：>4GB 大模型默认 16384 → 4096。9B Q4 权重 ~5.5GB + 16K f16 KV
 * ~2.4GB 超过 MIUI 6GB PSS 硬杀线（进程被系统直接杀掉）；原版 App 默认
 * n_ctx=2048。4096 使 KV 降到 ~0.6GB，加载链再按实时内存夹紧。
 * v4 变更（2026-11-06 真机取证）：v3 的加载链会把 4096 再按预算压到 2048，
 * 而工具 schema 基线 TOOL_BASELINE_TOKENS=4500，2048 首轮即「Context is full」。
 * 版本升 4 自愈已落盘的 2048 → 4096 策展档；加载链不再低于可用下限。
 */
export const CURATED_TABLE_VERSION = 4;

// 文件名/名匹配规则（先于尺寸档）。工具 schema 注入后提示词基线 ~4.5K
// （TOOL_BASELINE_TOKENS，bannerVariantResolver 沉底常量），每档 nCtx 必须
// 覆盖基线 + 对话窗 + 生成预留（策展门禁断言）：
// 1B 级 32768 为管家舒适档（KV 极小，封顶 GGUF 原生 32768）；2B 级 32768 为
// 写作/聊天主力档（Q8 近无损 + KV 48KB/token 极省）；3B 级 24576
// （Ministral KV 106KB/token）；4B 级 16384 为 KV 大户（128KB/token，
// f16 16K ≈ 2GB KV，8-9GB 预算内余量充足——大王实测 16K 可用）。
const CURATED_RULES: Array<{match: RegExp; nCtx: number}> = [
  {match: /minicpm/i, nCtx: 32768},
  {match: /qwen3\.5-2b/i, nCtx: 32768},
  {match: /qwen3\.5-4b/i, nCtx: 16384},
  // 8B/9B 级：权重已占 5GB 上下，长上下文 KV 会把进程推过厂商 PSS 硬杀线
  // （MIUI/HyperOS 实测 6GB）。4096 只作上界，加载前还会按实时可用内存
  // 降档（原版 App 默认 2048 正是能稳定加载的口径）。
  {match: /qwen3(\.5)?[-_ ](8b|9b)/i, nCtx: 4096},
  {match: /ministral/i, nCtx: 24576},
  {match: /gemma-3-4b/i, nCtx: 32768},
  {match: /lfm2\.5-8b/i, nCtx: 4096},
  {match: /lfm2\.5-2\.6b/i, nCtx: 16384},
];

/**
 * 模型策展默认 n_ctx：规则表优先，未命中按尺寸分档
 * （≤1.5GB→24576；≤4GB→32768；更大→4096）。
 * >4GB 大模型（7B/9B/14B Q4）权重本身就接近 6GB PSS 硬杀线，KV 每加 1K
 * 都是风险；4096 为大模型的安全上界，加载前还会按实时可用内存再夹紧。
 */
export function defaultNCtxForModel(
  model: Pick<Model, 'filename' | 'name'> & {size?: number},
): number {
  const hay = `${model.filename ?? ''} ${model.name ?? ''}`;
  for (const rule of CURATED_RULES) {
    if (rule.match.test(hay)) {
      return rule.nCtx;
    }
  }
  if (model.size && model.size <= 1.5e9) {
    return 24576;
  }
  if (model.size && model.size <= 4e9) {
    return 32768;
  }
  return 4096;
}
