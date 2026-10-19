import json
import jsonlines
import argparse
from collections import Counter, defaultdict
from random import sample
import random

class Sampler:
    def __init__(self, file_path, total_samples_needed=None, percentage=None,
                 output_file='../DATAsets/h_sampled_10000.json',
                 stats_before_file='../OUTPUTs/type_distribution_before.json',
                 stats_after_file='../OUTPUTs/type_distribution_after.json',
                 use_reservoir=False, random_seed=42):
        """
        初始化Sampler类。新增参数:
        :param use_reservoir: True 使用 reservoir（无偏），False 使用 greedy（先到先得，偏倚但更快，可提前停止）
        :param random_seed: 随机种子（可选）
        """
        self.file_path = file_path
        self.total_samples_needed = total_samples_needed
        self.percentage = percentage
        self.output_file = output_file
        self.stats_before_file = stats_before_file
        self.stats_after_file = stats_after_file
        self.sample_ids = {}  # {id: type} 映射（轻量）
        self.total_samples = 0
        self.use_reservoir = use_reservoir
        self.random_seed = random_seed
        random.seed(self.random_seed)

    def load_sample_sum(self):
        """
        流式读取JSONL文件，只加载id和type字段，统计总数和type分布。
        返回统计信息。
        """
        try:
            self.sample_ids = {}
            self.total_samples = 0
            with jsonlines.open(self.file_path, 'r') as reader:
                for item in reader:
                    sample_id = item.get('id')
                    if sample_id is not None:
                        typ = item.get('type', 'Unknown')
                        self.sample_ids[sample_id] = typ
                        self.total_samples += 1
            self.type_counts = Counter(self.sample_ids.values())
            print(f"Total samples: {self.total_samples}")
            print("Type分布:", self.type_counts)
            return self.type_counts, self.total_samples
        except Exception as e:
            print(f"Error loading file: {e}")
            return None, None

    def stratified_sample_by_type(self):
        """
        计算各 type 的 quota（按比例，至少1），再做单次流式抽样：
        - use_reservoir=True: 对每个 type 做 reservoir sampling（无偏，须扫描整个文件）
        - use_reservoir=False: greedy 填满 quota，达到总数可提前结束（偏倚但最快）
        """
        if not self.sample_ids:
            print("Error: No stats loaded. Call load_sample_sum() first.")
            return

        # 计算 total_samples_needed
        if self.percentage is not None:
            if not (0.001 <= self.percentage <= 0.08):
                print("Error: Percentage must be between 0.1% (0.001) and 8% (0.08).")
                return
            self.total_samples_needed = int(self.percentage * self.total_samples)
            print(f"Calculated total_samples_needed: {self.total_samples_needed} based on {self.percentage*100}% of {self.total_samples}")

        if not self.total_samples_needed:
            print("Error: total_samples_needed not set.")
            return

        # 保存抽样前type分布
        with open(self.stats_before_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.type_counts), f, ensure_ascii=False, indent=4)
        print(f"抽样前type分布保存到 {self.stats_before_file}")

        # 1) 计算按比例的初始 quota（并考虑可用样本上限），使用"最大1个"策略
        total = self.total_samples
        desired = {}
        quota = {}
        for typ, cnt in self.type_counts.items():
            d = (cnt / total) * self.total_samples_needed
            desired[typ] = d
            quota[typ] = max(1, int(d)) if cnt > 0 else 0

        # cap by available counts
        for typ in list(quota.keys()):
            if quota[typ] > self.type_counts[typ]:
                quota[typ] = self.type_counts[typ]

        # 分配余数（按小数部分从大到小分配），直到满足 total_samples_needed 或没有可分配空间
        current_sum = sum(quota.values())
        remain = self.total_samples_needed - current_sum
        if remain > 0:
            # 计算小数部分并按降序分配
            remainders = [(typ, desired[typ] - int(desired[typ])) for typ in desired]
            remainders.sort(key=lambda x: x[1], reverse=True)
            idx = 0
            while remain > 0:
                typ, _ = remainders[idx % len(remainders)]
                if quota[typ] < self.type_counts[typ]:
                    quota[typ] += 1
                    remain -= 1
                idx += 1
                # 若无法再分配（所有类型已到上限），停止
                if idx >= len(remainders) * 2 and all(quota[t] >= self.type_counts[t] for t in quota):
                    break

        # 最终检查：如果仍然不足（极少见，类型数量少），将总数降到可获取的最大值
        max_available = sum(self.type_counts.values())
        if self.total_samples_needed > max_available:
            print(f"警告: 请求样本数 {self.total_samples_needed} 大于可用 {max_available}，将降为可用数量。")
            self.total_samples_needed = max_available

        print("Quota per type sample count:", {k: v for k, v in quota.items() if v > 0})

        # 2) 单次流式抽样：reservoir 或 greedy
        sampled_data = []
        if self.use_reservoir:
            # 为每个 type 准备 reservoir 容器和计数器
            reservoirs = {typ: [] for typ in quota if quota[typ] > 0}
            seen = defaultdict(int)
            with jsonlines.open(self.file_path, 'r') as reader:
                for item in reader:
                    typ = item.get('type', 'Unknown')
                    if typ not in reservoirs:
                        continue
                    seen[typ] += 1
                    k = quota[typ]
                    r = reservoirs[typ]
                    n = seen[typ]
                    if len(r) < k:
                        r.append(item)
                    else:
                        # 以 k/n 的概率替换
                        j = random.randint(0, n - 1)
                        if j < k:
                            r[j] = item
            # 合并结果
            for typ, r in reservoirs.items():
                sampled_data.extend(r)
        else:
            # greedy: 先到先得，达到每个 type quota 即不再接受；达到总数可提前停止（最快但有偏）
            collected_per_type = defaultdict(int)
            with jsonlines.open(self.file_path, 'r') as reader:
                for item in reader:
                    typ = item.get('type', 'Unknown')
                    if quota.get(typ, 0) <= 0:
                        continue
                    if collected_per_type[typ] < quota[typ]:
                        sampled_data.append(item)
                        collected_per_type[typ] += 1
                        if len(sampled_data) >= self.total_samples_needed:
                            break

        # 如果合并后数量仍与期望不符，进行补足（从已读取样本中随机补足或重新扫描）
        if len(sampled_data) < self.total_samples_needed:
            need = self.total_samples_needed - len(sampled_data)
            print(f"注意: 实际抽到 {len(sampled_data)}，需要补足 {need} 条。将在文件中补充（快速方案）。")
            # 快速补足：再次流式查找尚未选中的样本并补足（用 set 加速）
            selected_ids = set(item.get('id') for item in sampled_data if item.get('id') is not None)
            with jsonlines.open(self.file_path, 'r') as reader:
                for item in reader:
                    if len(sampled_data) >= self.total_samples_needed:
                        break
                    sid = item.get('id')
                    if sid is None or sid in selected_ids:
                        continue
                    sampled_data.append(item)
                    selected_ids.add(sid)

        print(f"抽样后的样本数: {len(sampled_data)}")

        # 检查抽样后的type分布并保存
        sampled_type_counts = Counter(item.get('type', 'Unknown') for item in sampled_data)
        print("抽样后的type分布:", sampled_type_counts)
        with open(self.stats_after_file, 'w', encoding='utf-8') as f:
            json.dump(dict(sampled_type_counts), f, ensure_ascii=False, indent=4)
        print(f"抽样后type分布保存到 {self.stats_after_file}")

        # 保存抽样结果为JSONL
        with jsonlines.open(self.output_file, 'w') as writer:
            for item in sampled_data:
                writer.write(item)
        print(f"保存到 {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description="分层抽样脚本")
    parser.add_argument('--file_path', type=str, default='../DATAsets/samples_1xx.jsonl', help='输入文件路径（JSONL格式）')
    parser.add_argument('--total_samples_needed', type=int, default=10000, help='需要的总样本数')
    parser.add_argument('--percentage', type=float, help='抽样百分比 (0.001 to 0.08)')
    parser.add_argument('--output_file', type=str, default='../DATAsets/sampled1xx_10000.jsonl', help='输出文件路径（JSONL格式）')
    parser.add_argument('--stats_before_file', type=str, default='../logs/type_distribution_before.json', help='抽样前type分布保存路径')
    parser.add_argument('--stats_after_file', type=str, default='../logs/type_distribution_after.json', help='抽样后type分布保存路径')
    
    args = parser.parse_args()
    
    # 创建Sampler实例
    sampler = Sampler(
        file_path=args.file_path,
        total_samples_needed=args.total_samples_needed,
        percentage=args.percentage,
        output_file=args.output_file,
        stats_before_file=args.stats_before_file,
        stats_after_file=args.stats_after_file
    )
    
    # 加载数据
    stats = sampler.load_sample_sum()
    if stats is None:
        return

    # 执行分层抽样
    sampler.stratified_sample_by_type()

if __name__ == "__main__":
    main()