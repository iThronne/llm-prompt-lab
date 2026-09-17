// 只规范化查找键，绝不改写显示/复制/保存的原文。与 src/text_matching.py 对齐。
function normalizeMatchText(value) {
    return String(value ?? '').normalize('NFC')
        .replace(/[\uff1f\ufe56]/g, '?').replace(/[\uff01\ufe57]/g, '!')
        .replace(/[\u200b\ufeff\u2060]/g, '')
        .replace(/[\t\n\v\f\r \u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/g, ' ')
        .trim().replace(/ +([?!])/g, '$1');
}
function questionMatchKey(value) {
    return normalizeMatchText(value).replace(/^\u00bf+/, '').replace(/\?+$/, '').trim();
}
function matchesSearchText(value, needle) {
    const haystack = normalizeMatchText(value).toLowerCase();
    const query = normalizeMatchText(needle).toLowerCase();
    if (!query || haystack.includes(query)) return true;
    const question = questionMatchKey(query);
    return Boolean(question) && questionMatchKey(haystack).includes(question);
}
