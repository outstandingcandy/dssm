#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""汉字处理的工具:判断unicode是否是汉字，数字，英文，或者其他字符。全角符号转半角符号。"""


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def is_chinese_all(ustring):
    """判断一个unicode串是否是汉字串"""
    for uchar in ustring:
        if not is_chinese(uchar):
            return False
    return True

def is_number(uchar, but = u''):
    """判断一个unicode是否是数字"""
    if uchar == but or uchar >= u'\u0030' and uchar<=u'\u0039':
        return True
    else:
       return False
 
def is_number_all(ustring, but = u''):
    """判断一个unicode串是否是数字串"""
    for uchar in ustring:
        if not is_number(uchar, but):
            return False
    return True

def is_alphabet(uchar, but = u''):
    """判断一个unicode是否是英文字母"""
    if uchar == but or (uchar >= u'\u0061' and uchar<=u'\u007a') or (uchar >= u'\u0041' and uchar<=u'\u005a'):
        return True
    else:
        return False
 
def is_alphabet_all(ustring, but = u''):
    """判断一个unicode串是否是英文字母串"""
    for uchar in ustring:
        if not is_alphabet(uchar, but):
            return False
    return True

def is_alphanum(uchar, but = u''):
    """判断一个unicode是否是英文字母或数字"""
    if is_number(uchar, but) or is_alphabet(uchar, but):
        return True
    else:
        return False
 
def is_alpha_or_num_all(ustring, but = u''):
    """判断一个unicode串是否是英文字母或数字串"""
    for uchar in ustring:
        if not is_alphanum(uchar, but):
            return False
    return True

def is_alpha_and_num_all(ustring, but = u''):
    """判断一个unicode串是否是英文字母及数字串"""
    alphabet = 0
    number = 0
    for uchar in ustring:
        if is_alphabet(uchar, but):
            alphabet += 1
        elif is_number(uchar, but):
            number += 1
        else:
            return False
    if alphabet > 0 and number > 0:
        return True
    else:
        return False

def is_other(uchar, but = u''):
    """判断是否非汉字、数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar, but) or is_alphabet(uchar, but)):
        return True
    else:
        return False
 
def is_other_all(ustring, but = u''):
    """判断是否非汉字、数字和英文字符串"""
    for uchar in ustring:
        if not is_other(uchar, but):
            return False
    return True

def exist_chinese(ustring):
    for uchar in ustring:
        if is_chinese(uchar):
            return True
    return False

def exist_number(ustring):
    for uchar in ustring:
        if is_number(uchar):
            return True
    return False

def exist_alphabet(ustring):
    for uchar in ustring:
        if is_alphabet(uchar):
            return True
    return False

def exist_other(ustring, but = u''):
    for uchar in ustring:
        if is_other(uchar, but):
            return True
    return False

def B2Q(uchar):
    """半角转全角"""
    inside_code=ord(uchar)
    if inside_code<0x0020 or inside_code>0x7e:      #不是半角字符就返回原来的字符
        return uchar
    if inside_code==0x0020:       #除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code=0x3000
    else:
        inside_code+=0xfee0
    return unichr(inside_code)

def Q2B(uchar):
    """全角转半角"""
    if uchar == u'’':
		return u'\''
    inside_code=ord(uchar)
    if inside_code==0x3000:
        inside_code=0x0020
    else:
        inside_code-=0xfee0
    if inside_code<0x0020 or inside_code>0x7e:      #转完之后不是半角字符返回原来的字符
        return uchar
    return unichr(inside_code)

def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])

def uniform(ustring):
    """格式化字符串，完成全角转半角，大写转小写的工作"""
    return stringQ2B(ustring).lower()

def string2List(ustring):
    """将ustring按照中文，字母，数字分开"""
    retList=[]
    utmp=[]
    for uchar in ustring:
        if is_other(uchar):
            if len(utmp)==0:
                continue
            else:
                retList.append("".join(utmp))
                utmp=[]
        else:
            utmp.append(uchar)
    if len(utmp)!=0:
        retList.append("".join(utmp))
    return retList
