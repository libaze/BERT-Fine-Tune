# -*- coding: utf-8 -*-
'''
    CMRC 2018 评估脚本
'''
import re
import nltk


# 中英文拆分函数
def mixed_segmentation(in_str, rm_punc=False):
    """
    对输入字符串进行中英文拆分，可以去除标点符号。
    参数:
        in_str: 输入的字符串。
        rm_punc: 是否去除标点符号，默认为False。
    返回:
        拆分后的字符串列表。
    """
    in_str = str(in_str).lower().strip()  # 转换为小写并去除首尾空格
    segs_out = []  # 存储拆分后的结果
    temp_str = ""  # 临时存储英文字符串
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']  # 特殊字符列表
    for char in in_str:
        if rm_punc and char in sp_char:  # 如果需要去除标点符号且当前字符是标点
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:  # 如果是中文字符或特殊字符
            if temp_str != "":  # 如果临时字符串不为空，则进行英文分词
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)  # 将中文字符或特殊字符添加到结果列表
        else:
            temp_str += char  # 将英文字符添加到临时字符串

    # 处理最后的英文字符串
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


# 删除标点符号函数
def remove_punctuation(in_str):
    """
    去除输入字符串中的标点符号。
    参数:
        in_str: 输入的字符串。
    返回:
        去除标点后的字符串。
    """
    in_str = str(in_str).lower().strip()  # 转换为小写并去除首尾空格
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']  # 特殊字符列表
    out_segs = []  # 存储去除标点后的字符
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# 查找最长公共字符串函数
def find_lcs(s1, s2):
    """
    查找两个字符串列表的最长公共子序列。
    参数:
        s1: 字符串列表1。
        s2: 字符串列表2。
    返回:
        最长公共子序列及其长度。
    """
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 初始化动态规划表
    mmax = 0  # 最长公共子序列长度
    p = 0  # 最长公共子序列的结束位置
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:  # 如果字符相同
                m[i + 1][j + 1] = m[i][j] + 1  # 更新动态规划表
                if m[i + 1][j + 1] > mmax:  # 如果找到更长的公共子序列
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax  # 返回最长公共子序列及其长度


# 计算F1分数
def calc_f1_score(answers, prediction):
    """
    计算单个预测答案与一组真实答案之间的最大F1分数。
    参数:
        answers: 真实答案的列表。
        prediction: 预测答案。
    返回:
        max_f1: 最大F1分数。
    """
    f1_scores = []  # 存储所有答案对的F1分数
    for ans in answers:
        # 对真实答案和预测答案进行分词并去除标点
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        # 找到最长公共子序列及其长度
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)  # 如果没有公共子序列，F1分数为0
            continue
        # 计算精确度、召回率和F1分数
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)  # 返回最大F1分数


# 计算精确匹配(EM)分数
def calc_em_score(answers, prediction):
    """
    计算单个预测答案与一组真实答案之间的精确匹配(EM)分数。
    参数:
        answers: 真实答案的列表。
        prediction: 预测答案。
    返回:
        em: 精确匹配分数（1表示精确匹配，0表示不匹配）。
    """
    em = 0  # 初始化精确匹配分数
    for ans in answers:
        # 去除真实答案和预测答案中的标点
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:  # 如果精确匹配
            em = 1
            break  # 找到匹配后退出循环
    return em


def evaluate(ground_truth_list, prediction_list):
    """
    计算预测列表与真实答案列表之间的F1分数和精确匹配(EM)分数。
    参数:
        ground_truth_list: 真实答案的列表。
        prediction_list: 预测答案的列表。
    返回:
        f1_score: 平均F1分数。
        em_score: 平均精确匹配分数。
        total_count: 答案的总对数。
        skip_count: 跳过的对数（在此实现中始终为0）。
    """
    # 确保真实答案列表和预测列表长度相同
    if len(ground_truth_list) != len(prediction_list):
        raise ValueError("The length of ground truth list and prediction list must be the same.")

    f1 = 0  # 初始化F1分数总和
    em = 0  # 初始化精确匹配分数总和
    total_count = len(ground_truth_list)  # 答案对的总数
    skip_count = 0  # 跳过的对数（此处未使用）

    # 遍历每一对真实答案和预测答案
    for i in range(total_count):
        answers = ground_truth_list[i]  # 当前真实答案
        prediction = prediction_list[i]  # 当前预测答案

        # 计算并累加F1分数
        f1 += calc_f1_score(answers, prediction)
        # 计算并累加精确匹配分数
        em += calc_em_score(answers, prediction)

    # 计算平均F1分数和平均精确匹配分数
    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count


if __name__ == '__main__':
    ground_truth_list = [['I am Tom!'], ['Tom 你好啊！'], ['你是谁？']]
    prediction_list = ['I am Tom!', 'Tom 你你好！', '是谁啊？']
    F1, EM, TOTAL, SKIP = evaluate(ground_truth_list, prediction_list)
    AVG = (EM + F1) * 0.5
    print('EM score:', EM)
    print('F1 score:', F1)
    print('TOTAL score:', TOTAL)
    print('SKIP score:', SKIP)
    print('AVG score:', AVG)
