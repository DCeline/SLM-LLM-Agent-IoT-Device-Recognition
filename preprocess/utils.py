HTMLTAGS = [
    'html', 'a', 'abbr', 'address', 'area', 'article', 'aside', 'audio', 'b', 'base',
    'bdi', 'bdo', 'blockquote', 'body', 'br', 'button', 'canvas', 'caption', 'cite',
    'code', 'col', 'colgroup', 'data', 'datalist', 'dd', 'del', 'details', 'dfn',
    'dialog', 'div', 'dl', 'dt', 'em', 'embed', 'fieldset', 'figcaption', 'figure',
    'footer', 'form', 'h1,', 'head', 'header', 'hgroup', 'hr', 'html', 'i', 'iframe',
    'img', 'input', 'ins', 'kbd', 'label', 'legend', 'li', 'link', 'main', 'map', 'mark',
    'mathml', 'menu', 'meta', 'meter', 'nav', 'noscript', 'object', 'ol', 'optgroup',
    'option', 'output', 'p', 'picture', 'pre', 'progress', 'q', 'rp', 'rt', 'ruby', 's',
    'samp', 'script', 'search', 'section', 'select', 'slot', 'small', 'source', 'span',
    'strong', 'style', 'sub', 'summary', 'sup', 'svg', 'table', 'tbody', 'td', 'template',
    'textarea', 'tfoot', 'th', 'thead', 'time', 'title', 'tr', 'track', 'u', 'ul', 'var',
    'video', 'wbr', 'autonomous', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'info'
]

def strify(obj):
    # 字典/列表对象递归转换为字符串
    if isinstance(obj, dict):
        return '\n'.join([f'{key}: {strify(value)}' for key, value in obj.items()])
    elif isinstance(obj, list):
        return '\n'.join([strify(item) for item in obj])
    else:
        return str(obj)