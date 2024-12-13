from collections import defaultdict


# 计算频繁项集的支持度
def calculate_support(itemset, transactions):
    return sum(1 for transaction in transactions if itemset.issubset(transaction))


# 获取频繁项集
def get_frequent_itemsets(candidates, transactions, min_support):
    frequent_itemsets = []
    itemsets_support = defaultdict(int)

    for candidate in candidates:
        support = calculate_support(candidate, transactions)
        itemsets_support[candidate] = support
        if support >= min_support:
            frequent_itemsets.append(candidate)

    return frequent_itemsets, itemsets_support


# 生成候选项集
def generate_candidates(itemsets, k):
    candidates = set()
    itemsets = list(itemsets)
    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            candidate = itemsets[i].union(itemsets[j])
            if len(candidate) == k:
                candidates.add(candidate)

    return candidates


# AprioriTID算法实现
def apriori_tid(transactions, min_support):
    # 将事务转换为集合形式
    transactions = [set(transaction) for transaction in transactions]

    # 生成单一项集的候选项
    single_items = set(item for transaction in transactions for item in transaction)
    candidate_1 = [frozenset([item]) for item in single_items]

    # 获取频繁项集
    frequent_itemsets_1, itemsets_support_1 = get_frequent_itemsets(candidate_1, transactions, min_support)

    # 存储所有的频繁项集
    frequent_itemsets = frequent_itemsets_1
    k = 2

    while frequent_itemsets:
        # 生成k项集候选项
        candidate_k = generate_candidates(frequent_itemsets, k)
        frequent_itemsets_k, _ = get_frequent_itemsets(candidate_k, transactions, min_support)

        if frequent_itemsets_k:
            frequent_itemsets.extend(frequent_itemsets_k)
            k += 1
        else:
            break

    return frequent_itemsets


# 示例：输入一个事务数据库和最小支持度
transactions = [
    [1,3,4],
    [2,3,5],
    [1,2,3,5],
    [2,5],
    [1,2,3,5]
]

min_support = 4

# 调用AprioriTID算法
frequent_itemsets = apriori_tid(transactions, min_support)

# 输出结果
for itemset in frequent_itemsets:
    print(itemset)
