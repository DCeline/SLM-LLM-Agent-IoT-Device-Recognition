# -*- coding:utf-8 -*-
# @Data: 2024 10/31
# @Author: cingkwang
# @Summary: 对banner做文本预处理
import re
from inspect import getmembers
from json import loads as parse_json, JSONDecodeError
from urllib.parse import unquote
from warnings import catch_warnings, simplefilter
import json
import jsonlines  # 新增导入

from bs4 import BeautifulSoup as HMTLParser
from bs4 import SoupStrainer, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning

from utils import strify, HTMLTAGS


def sanitize_http(banner):
    patterns = {
        '<uuid>': re.compile(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'),
        '<timestamp>': re.compile(
            r"(\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}:\d{2})|" # 
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?)|'  # ISO 8601
            r'((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s\d{2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}\s\d{2}:\d{2}:\d{2}\sGMT)|' # RFC 1123
            r'(\b\d{10,13}\b)' # Unix Timestamp
        ),
        '<long_random>': re.compile(r'\b[a-fA-F0-9]{32,}\b'),
    }
    
    for placeholder, pattern in patterns.items():
        banner = pattern.sub(placeholder, banner)
    return banner


class Cleaner():
    '''针对banner进行文本预处理'''
    def __init__(self):
        funcs = [m for m in getmembers(self) if m[0].startswith('clean_')]
        self.funcs = {func[0].split('_')[-1]: func[1] for func in funcs}
        self.protocols = list(self.funcs.keys())
    
    def clean(self, banner, protocol=None):
        if protocol:
            if protocol == 'https':
                protocol = 'http'
            if protocol in self.protocols:
                banner = self.funcs[protocol](banner)

        # ①清除中文字符： \u4e00-\u9fa5 ②清除无语义的字符：BOM(\ufeff)
        banner = re.sub(r'[\u4e00-\u9fa5\ufeff]', '', banner)
        # ③清除多余空格
        banner = re.sub(r'\s+', ' ', banner)
        return banner.strip().lower()

    @staticmethod
    def clean_http(banner):
        parts = re.split(r'[\r\n]+', banner.lower())
        idx = None
        for i, part in enumerate(parts):
            if part.startswith('http'):
                continue
            if not re.match(r'^[\w\.\- {}_"]+[:：]\s?.*', part):
                idx = i
                break
        if idx is None:
            headers, html = parts, ''
        elif idx > 0:
            headers, html = parts[:idx], '\n'.join(parts[idx:])
        elif idx == 0:
            try:
                return strify(parse_json(banner))
            except JSONDecodeError:
                headers, html = '', '\n'.join(parts)
        else:
            headers, html = parts, ''
            
        if headers:
            # ①清除banner中无信息的header
            irrelevant = (
                'http/1', 'date', 'last-modified', 'content', 'cache-control', 'expires',
                'etag', 'connection', 'transfer-encoding', 'access-control-allow-origin'
            )
            headers = '\n'.join([h for h in headers if h and not h.startswith(irrelevant)])
        else:
            headers = ''

        # 清除 <!DOCTYPE> 声明
        html = re.sub(r'<!DOCTYPE [^>]+>', ' ', html).strip()

        if html:
            with catch_warnings(record=True) as ws:
                simplefilter("always", category=MarkupResemblesLocatorWarning)
                simplefilter("always", category=XMLParsedAsHTMLWarning)
                dom = HMTLParser(
                    unquote(html),
                    'html5lib',
                    parse_only=SoupStrainer(name=HTMLTAGS)
                )
                for w in ws:
                    if w.category is MarkupResemblesLocatorWarning:
                        return headers + '\n' + html
                    elif w.category is XMLParsedAsHTMLWarning:
                        has_html_tag = False
                        for tag in HTMLTAGS:
                            if f'<{tag}>' in html:
                                has_html_tag = True
                                break
                        if has_html_tag:
                            pass
                        else:
                            dom = HMTLParser(unquote(html), 'xml')
            if str(dom):
                # 移除所有 <script> 标签及其内部内容
                for script in dom.find_all('script'):
                    script.decompose()
                # ②清除无信息的html版本声明、meta标签、css部分
                for element in dom.find_all('style'):
                    element.decompose()
                for tag in dom.find_all(True):
                    if tag.name in ('img', 'info'):
                        continue
                    # ③清除无信息的的标签内部属性
                    semantic_attrs = {
                        'class', 'id', 'name', 'alt', 'title', 'role', 'href',
                        'placeholder', 'src', 'content', 'value', 'ng-bind'
                    }
                    for attr in dict(tag.attrs):
                        if not (attr in semantic_attrs or attr.startswith(('data-', 'detect'))):
                            del tag.attrs[attr]
                    # ④清除自闭和标签：如 <span></span>
                    if not (tag.contents or tag.attrs):
                        tag.decompose()

                html = str(dom)
            else:
                return headers + '\n' + html

        return sanitize_http(headers + '\n' + html)


cleaner = Cleaner()

if __name__ == "__main__":

    # 初始化 Cleaner 实例
    cleaner = Cleaner()

    # 输入和输出文件路径
    input_file = "../DATAsets/final_test_set.jsonl"
    output_file = "../DATAsets/testset_cleaned.jsonl"

    try:
        # 读取 JSONL 文件
        with jsonlines.open(input_file, 'r') as reader:
            samples = list(reader)

        # 对每条样本的 banner 字段进行清洗（仅当存在 banner 字段时）
        for idx, sample in enumerate(samples, start=1):
            if 'banner' in sample:
                banner = sample['banner']
                cleaned_banner = cleaner.clean(banner, protocol='http')
                if cleaned_banner:  # 仅当清洗后不为空时保存
                    sample['banner'] = cleaned_banner
                else:
                    del sample['banner']  # 如果清洗为空，删除字段，避免空的 banner 字段
            print(f"清洗进度: {idx}/{len(samples)}")

        # 保存清洗后的样本到新的 JSONL 文件
        with jsonlines.open(output_file, 'w') as writer:
            for sample in samples:
                writer.write(sample)

        print(f"清洗完成！清洗后的样本已保存到 {output_file}")

    except FileNotFoundError:
        print(f"Error: 文件 {input_file} 不存在。")
    except json.JSONDecodeError:
        print(f"Error: 文件 {input_file} 不是有效的 JSON 格式。")
